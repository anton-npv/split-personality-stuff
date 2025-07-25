#!/usr/bin/env python3
"""Test Gemma tokenizer chat template."""

from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

# Test message
messages = [{"role": "user", "content": "What is 2+2?"}]

print("1. Original messages:")
print(messages)
print()

# Apply chat template
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("2. After apply_chat_template:")
print(repr(formatted))
print()

# Check what happens with double-nested
double_nested = [{"role": "user", "content": messages}]
print("3. Double-nested messages:")
print(double_nested)
print()

try:
    formatted_bad = tokenizer.apply_chat_template(
        double_nested,
        tokenize=False,
        add_generation_prompt=True
    )
    print("4. After apply_chat_template on double-nested:")
    print(repr(formatted_bad))
except Exception as e:
    print(f"4. Error with double-nested: {e}")