#!/usr/bin/env python3
"""Test Gemma chat template."""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

# Test conversation
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "Are you sure about that?"}
]

# Apply chat template
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Chat template format:")
print(prompt)
print("\nTokenizer special tokens:")
print(f"BOS token: {tokenizer.bos_token}")
print(f"EOS token: {tokenizer.eos_token}")
print(f"PAD token: {tokenizer.pad_token}")