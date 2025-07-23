"""Model loading and device management for local inference."""

import torch
import logging
from typing import List, Dict, Tuple, Optional, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    BitsAndBytesConfig
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        logging.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        return torch.device("cuda")
    logging.info("CUDA not available. Using CPU.")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_path: str,
    quantization: Optional[str] = None,
    device_map: str = "auto"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str, torch.device]:
    """Load model and tokenizer with optimizations."""
    device = get_device()
    logging.info(f"Loading model: {model_path}")
    
    # Configure quantization if requested
    quantization_config = None
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with appropriate settings
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "trust_remote_code": True
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    
    # Configure padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    model.padding_side = "left"
    tokenizer.padding_side = "left"
    
    model_name = model_path.split("/")[-1]
    logging.info(f"Model {model_name} loaded successfully")
    
    return model, tokenizer, model_name, device


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for specific tokens."""
    
    def __init__(self, tokenizer, stop_strings: List[str]):
        self.stop_ids_list = []
        for stop_string in stop_strings:
            stop_ids = tokenizer(stop_string, add_special_tokens=False).input_ids
            self.stop_ids_list.append(stop_ids)
    
    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_ids_list:
            if input_ids.shape[1] < len(stop_ids):
                continue
            tail = input_ids[0, -len(stop_ids):]
            if torch.equal(tail, torch.tensor(stop_ids, device=input_ids.device)):
                return True
        return False


def generate_completion_batch(
    model,
    tokenizer,
    device,
    prompts: List[Dict],
    batch_size: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    stop_strings: Optional[List[str]] = None
) -> List[Dict]:
    """Generate completions for a batch of prompts."""
    
    results = []
    
    # Set up stopping criteria
    stopping_criteria = None
    if stop_strings:
        stopping_criteria = StoppingCriteriaList([StopOnTokens(tokenizer, stop_strings)])
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        
        # Extract conversation format
        conversations = []
        question_ids = []
        
        for prompt_dict in batch:
            if "messages" in prompt_dict:
                conversations.append(prompt_dict["messages"])
            else:
                # Convert single prompt to messages format
                conversations.append([{"role": "user", "content": prompt_dict.get("prompt", "")}])
            question_ids.append(prompt_dict.get("question_id", i))
        
        logging.info(f"Processing batch {i//batch_size+1}/{(len(prompts)+batch_size-1)//batch_size}")
        
        # Format prompts using chat template
        formatted_prompts = []
        for conv in conversations:
            formatted = tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted)
        
        # Tokenize batch
        encoded = tokenizer(
            formatted_prompts,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )
        
        # Get model and device (handle distributed models)
        gen_model = model.module if hasattr(model, "module") else model
        gen_device = next(gen_model.parameters()).device
        
        input_ids = encoded["input_ids"].to(gen_device)
        attention_mask = encoded["attention_mask"].to(gen_device)
        
        # Generate
        with torch.no_grad():
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature if temperature > 0 else None,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            if stopping_criteria:
                generation_kwargs["stopping_criteria"] = stopping_criteria
            
            outputs = gen_model.generate(**generation_kwargs)
        
        # Decode completions (remove input prompt)
        input_length = input_ids.shape[1]
        generated_tokens = outputs[:, input_length:]
        
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Collect results
        for qid, completion in zip(question_ids, decoded):
            results.append({
                "question_id": qid,
                "completion": completion.strip()
            })
    
    return results