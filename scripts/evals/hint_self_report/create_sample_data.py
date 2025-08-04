#!/usr/bin/env python3
"""Create sample data for testing the hint self-report pipeline."""

import json
import random
from pathlib import Path

# Set paths
source_export_dir = Path("data/mmlu/exports/gemma-3-4b-local/sycophancy")
target_export_dir = Path("data/mmlu/exports/gemma-3-4b-test/sycophancy")

# Create target directory
target_export_dir.mkdir(parents=True, exist_ok=True)

# Load source data
print("Loading source data...")
with open(source_export_dir / "switched_metadata.json", 'r') as f:
    metadata = json.load(f)

with open(source_export_dir / "switched_conversations.jsonl", 'r') as f:
    conversations = [json.loads(line) for line in f]

# Sample 20 questions
print(f"Total questions available: {len(conversations)}")
sample_size = min(20, len(conversations))
random.seed(42)  # For reproducibility
sampled_indices = random.sample(range(len(conversations)), sample_size)

# Extract sampled data
sampled_conversations = [conversations[i] for i in sampled_indices]
sampled_question_ids = [conv["question_id"] for conv in sampled_conversations]

# Filter metadata to match sampled questions
sampled_metadata = {
    "dataset": metadata["dataset"],
    "model": "gemma-3-4b-test",  # Changed model name
    "hint_type": metadata["hint_type"],
    "total_switched": sample_size,
    "questions": [q for q in metadata["questions"] if q["question_id"] in sampled_question_ids]
}

# Save sampled data
print(f"Saving {sample_size} sampled conversations...")
with open(target_export_dir / "switched_conversations.jsonl", 'w') as f:
    for conv in sampled_conversations:
        f.write(json.dumps(conv) + '\n')

with open(target_export_dir / "switched_metadata.json", 'w') as f:
    json.dump(sampled_metadata, f, indent=2)

print(f"Sample data created successfully in {target_export_dir}")
print(f"- {sample_size} conversations saved")
print(f"- Metadata updated with model name 'gemma-3-4b-test'")