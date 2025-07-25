#!/usr/bin/env python3
"""Test prompt format to debug double-nesting issue."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.base import MCQRecord
from src.prompts import build_baseline_prompt

# Create a test question
question = MCQRecord(
    question_id=0,
    question="What is 2+2?",
    choices={"A": "3", "B": "4", "C": "5", "D": "6"},
    correct="B"
)

# Build prompt
messages = build_baseline_prompt(question)

print(f"Type of messages: {type(messages)}")
print(f"Number of messages: {len(messages)}")
print(f"Messages content:")
import json
print(json.dumps(messages, indent=2))