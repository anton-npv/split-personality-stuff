#!/usr/bin/env python3
"""Debug local client prompt handling."""

import sys
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.base import MCQRecord
from src.prompts import build_baseline_prompt
from src.clients import create_client
from src.utils.config import load_model_config

# Create a test question
question = MCQRecord(
    question_id=0,
    question="What is 2+2?",
    choices={"A": "3", "B": "4", "C": "5", "D": "6"},
    correct="B"
)

# Build prompt using the same method as the pipeline
messages = build_baseline_prompt(question)

print("1. Original messages from build_baseline_prompt:")
print(json.dumps(messages, indent=2))
print()

# Now let's see what happens when we save it as the completion utils does
result = {
    "question_id": question.question_id,
    "hint_type": "none",
    "prompt": messages,
    "completion": "Test completion",
    "model": "test-model"
}

print("2. Result dict as it would be saved:")
print(json.dumps(result, indent=2))
print()

# Now let's check what the prompt field looks like when loaded
print("3. The prompt field from the result:")
print(json.dumps(result["prompt"], indent=2))