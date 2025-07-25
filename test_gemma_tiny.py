#!/usr/bin/env python3
"""Ultra-minimal test of Gemma model."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print("Loading model...")
start = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"  # Force single GPU
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
print(f"Model loaded in {time.time() - start:.1f}s")

# Even simpler test
prompt = "<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n"

print(f"\nPrompt: {repr(prompt)}")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

print("\nGenerating just 5 tokens...")
start = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
print(f"Generation took {time.time() - start:.1f}s")

response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(f"Model said: {response}")