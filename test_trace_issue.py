#!/usr/bin/env python3
"""Trace the double-nesting issue."""

import sys
import json
import asyncio
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.base import MCQRecord
from src.prompts import build_baseline_prompt
from src.clients import create_client
from src.utils.config import load_model_config

async def main():
    # Create a test question
    question = MCQRecord(
        question_id=0,
        question="What is 2+2?",
        choices={"A": "3", "B": "4", "C": "5", "D": "6"},
        correct="B"
    )

    # Build prompt
    messages = build_baseline_prompt(question)
    print("1. Messages from build_baseline_prompt:")
    print(json.dumps(messages, indent=2))
    print()

    # Create client
    all_model_configs = load_model_config()
    model_config = all_model_configs["gemma-3-4b-local"].copy()
    provider = model_config.pop("provider")
    
    # Let's check what's being passed to create_client
    print("2. Model config being passed to create_client:")
    print(json.dumps(model_config, indent=2))
    print()
    
    client = create_client(
        provider=provider,
        model_id="gemma-3-4b-local",
        **model_config
    )
    
    # Now let's trace what happens when we call complete
    # First, let's patch the client to see what it receives
    original_complete = client.complete
    
    async def traced_complete(msgs):
        print("3. Messages received by client.complete:")
        print(f"Type: {type(msgs)}")
        print(json.dumps(msgs, indent=2))
        print()
        return await original_complete(msgs)
    
    client.complete = traced_complete
    
    # Call complete
    try:
        response = await client.complete(messages)
        print("4. Response from client:")
        print(f"Content: {response.content[:100]}...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())