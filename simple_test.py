#!/usr/bin/env python3
"""Simple test: generate and print completions for 1 question."""

import asyncio
import json
from dotenv import load_dotenv
from src.datasets import DatasetRegistry
from src.datasets.base import MCQRecord, HintRecord
from src.clients import create_client
from src.prompts import build_baseline_prompt, build_hinted_prompt
from src.utils.config import load_model_config

load_dotenv()

async def simple_test():
    """Generate completions for 1 question and print everything."""
    
    # Load data
    registry = DatasetRegistry()
    questions = registry.load_dataset("mmlu", "dev")
    
    with open("data/mmlu/hints/sycophancy.json", 'r') as f:
        hints_data = json.load(f)
        hints = [HintRecord(**h) for h in hints_data]
    
    hints_by_id = {h.question_id: h for h in hints}
    
    # Get first question
    question = questions[0]
    hint = hints_by_id[question.question_id]
    
    # Create client
    model_config = load_model_config()
    model_cfg = model_config["claude-3-5-sonnet"]
    
    client = create_client(
        provider=model_cfg["provider"],
        model_id=model_cfg["model_id"],
        temperature=0,
        max_tokens=500
    )
    
    print("="*80)
    print("QUESTION")
    print("="*80)
    print(f"ID: {question.question_id}")
    print(f"Question: {question.question}")
    print(f"Choices: {question.choices}")
    print(f"Correct answer: {question.correct}")
    print()
    print(f"Hint points to: {hint.hint_option}")
    print(f"Hint is correct: {hint.is_correct_option}")
    print(f"Hint text: {hint.hint_text}")
    
    # BASELINE
    print("\n" + "="*80)
    print("BASELINE PROMPT")
    print("="*80)
    baseline_messages = build_baseline_prompt(question)
    print(baseline_messages[0]["content"])
    
    print("\n" + "="*80) 
    print("BASELINE COMPLETION")
    print("="*80)
    baseline_response = await client.complete(baseline_messages)
    print(baseline_response.content)
    
    # HINTED
    print("\n" + "="*80)
    print("HINTED PROMPT")
    print("="*80)
    hinted_messages = build_hinted_prompt(question, hint)
    print(hinted_messages[0]["content"])
    
    print("\n" + "="*80)
    print("HINTED COMPLETION") 
    print("="*80)
    hinted_response = await client.complete(hinted_messages)
    print(hinted_response.content)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Baseline usage: {baseline_response.usage}")
    print(f"Hinted usage: {hinted_response.usage}")

if __name__ == "__main__":
    asyncio.run(simple_test())