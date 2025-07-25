#!/usr/bin/env python3
"""Find where the double-nesting happens."""

import sys
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Let's trace through the exact code path used in the script
# First, load a question
from src.datasets import DatasetRegistry
dataset_registry = DatasetRegistry()
questions = dataset_registry.load_dataset("mmlu", "test")
question = questions[0]

print("1. Original question:")
print(f"  Type: {type(question.question)}")
print(f"  Content: {question.question[:50]}...")
print()

# Build prompt
from src.prompts import build_baseline_prompt
messages = build_baseline_prompt(question)

print("2. Messages from build_baseline_prompt:")
print(f"  Type: {type(messages)}")
print(f"  Length: {len(messages)}")
print(f"  First message type: {type(messages[0])}")
print(f"  First message content type: {type(messages[0]['content'])}")
print(f"  Messages: {json.dumps(messages, indent=2)}")
print()

# Now let's check what happens when we save this
# Following the exact pattern from completion_utils.py
result = {
    "question_id": question.question_id,
    "hint_type": "none",
    "prompt": messages,
    "completion": "Test",
    "model": "test"
}

# Check the saved format
print("3. Result dict prompt field:")
prompt_field = result["prompt"]
print(f"  Type: {type(prompt_field)}")
print(f"  Content: {json.dumps(prompt_field, indent=2)}")

# Now let's see if something weird happens when we serialize/deserialize
json_str = json.dumps(result)
loaded = json.loads(json_str)

print("\n4. After JSON round-trip:")
print(f"  Prompt type: {type(loaded['prompt'])}")
print(f"  First message content type: {type(loaded['prompt'][0]['content'])}")
print(f"  Content: {loaded['prompt'][0]['content'][:50]}...")