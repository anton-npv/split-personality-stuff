#!/usr/bin/env python3
"""Check MMLU question structure."""

import sys
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import DatasetRegistry

# Load dataset
dataset_registry = DatasetRegistry()
questions = dataset_registry.load_dataset("mmlu", "test")

# Check first question
first_q = questions[0]
print(f"Question type: {type(first_q)}")
print(f"Question fields: {first_q.__dict__.keys() if hasattr(first_q, '__dict__') else 'N/A'}")
print()
print("First question:")
print(f"  ID: {first_q.question_id}")
print(f"  Question: {first_q.question[:50]}...")
print(f"  Question type: {type(first_q.question)}")

# Check if question might already be in a weird format
if isinstance(first_q.question, list):
    print("WARNING: Question is a list!")
    print(json.dumps(first_q.question, indent=2))