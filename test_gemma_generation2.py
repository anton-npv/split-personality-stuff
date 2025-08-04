#!/usr/bin/env python3
"""Test Gemma generation with proper settings."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
tokenizer.padding_side = "left"  # Important for generation

# Simple test
prompt = "What is 2+2? Answer: "

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Input tokens: {inputs['input_ids'][0][:20]}")  # First 20 tokens

# Generate with different settings
print("\nGenerating...")
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=50,
        min_new_tokens=5,  # Force at least some generation
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )

# Decode
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nFull response:")
print(response)

# Try with chat template
print("\n" + "="*50 + "\n")
print("Testing with chat template:")
messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=50,
        min_new_tokens=5,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Response with chat template:")
print(response)