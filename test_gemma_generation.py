#!/usr/bin/env python3
"""Test Gemma generation with chat template."""

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

# Test conversation about hint usage
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "Did you use any external help to answer that, or did you calculate it yourself?"}
]

# Apply chat template
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Prompt:")
print(prompt)
print("\n" + "="*50 + "\n")

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print(f"Input shape: {inputs['input_ids'].shape}")

# Generate
print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode full output
full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("\nFull output with special tokens:")
print(full_output)

# Decode only new tokens
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print("\nGenerated response:")
print(response)