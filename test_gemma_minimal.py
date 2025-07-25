#!/usr/bin/env python3
"""Minimal test of Gemma model generation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

# Simple test
messages = [{"role": "user", "content": "What is 2+2?"}]

print("\nFormatting prompt...")
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Prompt: {repr(prompt)}")

print("\nTokenizing...")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"Input shape: {inputs.input_ids.shape}")

print("\nGenerating (max 20 tokens)...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

print("\nDecoding...")
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")

# Extract just the model's response
model_response = response.split("<start_of_turn>model\n")[-1]
print(f"\nModel said: {model_response}")