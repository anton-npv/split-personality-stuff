#!/usr/bin/env python3
"""Generate follow-up responses asking models about their hint usage.

This script takes the switched conversations and adds a follow-up turn where we ask
the model whether it used the hint. Uses local models with chat template support.

Usage:
    # Single GPU with accelerate
    python scripts/evals/hint_self_report/01_generate_followup_responses.py
    
    # With custom config
    python scripts/evals/hint_self_report/01_generate_followup_responses.py --config config_test.yaml
"""

import argparse
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.utils.paths import get_dataset_path
from src.utils.jsonl import read_jsonl, write_jsonl
from src.utils.logging_setup import setup_logging


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_export_data(config: Dict[str, Any]) -> tuple:
    """Load the exported switched conversations and metadata."""
    export_dir = get_dataset_path(config['data']['dataset']) / "exports" / \
                 config['data']['source_model'] / config['data']['hint_type']
    
    conversations_path = export_dir / "switched_conversations.jsonl"
    metadata_path = export_dir / "switched_metadata.json"
    
    conversations = list(read_jsonl(conversations_path))
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return conversations, metadata


def create_followup_messages(conversation: Dict, hint_info: Dict, question_template: str) -> List[Dict]:
    """Create the messages list for the follow-up conversation."""
    # Get the hint text from the metadata
    hint_text = hint_info.get("hint_text", "the provided hint")
    
    # Format the question with the hint text
    question = question_template.replace("{hint_text}", f'"{hint_text}"')
    
    # Get original messages
    messages = conversation["messages"]
    
    # Create proper message format for Gemma
    chat_messages = [
        {"role": "user", "content": messages[0]['content']},
        {"role": "assistant", "content": messages[1]['content']},
        {"role": "user", "content": question}
    ]
    
    return chat_messages


def generate_responses_batch(
    model,
    tokenizer,
    conversations: List[Dict],
    metadata: Dict,
    config: Dict,
    accelerator: Accelerator,
    device
) -> List[Dict]:
    """Generate follow-up responses for conversations."""
    logger = logging.getLogger(__name__)
    
    # Get the selected question template
    question_templates = {q['id']: q['template'] for q in config['followup_questions']}
    selected_template = question_templates[config['selected_question']]
    
    # Create a mapping of question_id to hint_info
    hint_info_by_id = {q["question_id"]: q["hint_info"] for q in metadata["questions"]}
    
    results = []
    batch_size = config['generation']['batch_size']
    
    # Process in batches
    for i in tqdm(range(0, len(conversations), batch_size), desc="Generating responses"):
        batch_conversations = conversations[i:i+batch_size]
        batch_messages = []
        
        for conv in batch_conversations:
            qid = conv["question_id"]
            hint_info = hint_info_by_id.get(qid, {})
            messages = create_followup_messages(conv, hint_info, selected_template)
            batch_messages.append(messages)
        
        # Apply chat template to each conversation
        batch_prompts = []
        for messages in batch_messages:
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            batch_prompts.append(prompt)
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            truncation=False,  # Don't truncate for generation
            padding=True
        ).to(device)
        
        # Generate
        with torch.no_grad():
            temperature = config['generation']['temperature']
            generation_kwargs = {
                "input_ids": inputs['input_ids'],
                "attention_mask": inputs['attention_mask'],
                "max_new_tokens": config['generation']['max_new_tokens'],
                "do_sample": temperature > 0,
                "temperature": temperature if temperature > 0 else None,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            # Unwrap model if it's wrapped by accelerator
            if hasattr(model, 'module'):
                outputs = model.module.generate(**generation_kwargs)
            else:
                outputs = model.generate(**generation_kwargs)
        
        # Decode responses
        for j, (conv, output, messages) in enumerate(zip(batch_conversations, outputs, batch_messages)):
            # Decode only the new tokens
            input_length = inputs['input_ids'][j].shape[0]
            response = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            
            # Create extended conversation
            hint_info = hint_info_by_id.get(conv["question_id"], {})
            extended_conv = {
                "question_id": conv["question_id"],
                "messages": conv["messages"] + [
                    {"role": "user", "content": selected_template.replace(
                        "{hint_text}", 
                        f'"{hint_info.get("hint_text", "the provided hint")}"'
                    )},
                    {"role": "assistant", "content": response}
                ],
                "followup_metadata": {
                    "question_template_id": config['selected_question'],
                    "generation_model": config['generation']['model']
                }
            }
            results.append(extended_conv)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate follow-up responses about hint usage")
    parser.add_argument("--config", default="scripts/evals/hint_self_report/config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load configuration
    config = load_config(Path(args.config))
    logger.info(f"Loaded configuration from {args.config}")
    
    # Load export data
    logger.info("Loading exported switched conversations...")
    conversations, metadata = load_export_data(config)
    logger.info(f"Loaded {len(conversations)} switched conversations")
    
    # Set up output path
    output_dir = get_dataset_path(config['data']['dataset']) / \
                 config['output']['followup_dir'].format(
                     source_model=config['data']['source_model'],
                     hint_type=config['data']['hint_type']
                 )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config['output']['extended_conversations']
    
    # Initialize model and tokenizer
    logger.info(f"Loading model: {config['generation']['model']}")
    
    # Import model loading utilities
    from src.local.model_handler import load_model_and_tokenizer
    
    # Load model and tokenizer using the same approach as working scripts
    # Don't use device_map with accelerate
    model, tokenizer, model_name, device = load_model_and_tokenizer(
        model_path="google/gemma-3-4b-it",
        quantization="4bit"
    )
    
    # Prepare model with accelerator
    model, tokenizer = accelerator.prepare(model, tokenizer)
    device = accelerator.device
    
    logger.info(f"Using device: {device}")
    
    # Generate responses
    logger.info("Generating follow-up responses...")
    results = generate_responses_batch(
        model, tokenizer, conversations, metadata, config, accelerator, device
    )
    
    # Save results (only on main process, but with single GPU this is always true)
    write_jsonl(results, output_path)
    logger.info(f"Saved {len(results)} extended conversations to {output_path}")
    
    # Save metadata about this run
    question_templates = {q['id']: q['template'] for q in config['followup_questions']}
    run_metadata = {
        "dataset": config['data']['dataset'],
        "source_model": config['data']['source_model'],
        "hint_type": config['data']['hint_type'],
        "generation_model": config['generation']['model'],
        "question_template_id": config['selected_question'],
        "question_template": question_templates[config['selected_question']],
        "total_conversations": len(results)
    }
    
    metadata_path = output_dir / "generation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(run_metadata, f, indent=2)
    logger.info(f"Saved generation metadata to {metadata_path}")
    
    logger.info("Follow-up response generation complete!")


if __name__ == "__main__":
    main()