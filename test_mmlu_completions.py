#!/usr/bin/env python3
"""Test MMLU completion generation with a small sample."""

import asyncio
import json
import os
from dotenv import load_dotenv
from src.datasets import DatasetRegistry

# Load environment variables from .env file
load_dotenv()
from src.datasets.base import MCQRecord, HintRecord
from src.clients import create_client
from src.prompts import build_baseline_prompt, build_hinted_prompt
from src.utils.config import load_model_config
from src.utils.logging_setup import setup_logging

async def test_mmlu_completions(model_name: str = "claude-3-5-sonnet", num_questions: int = 3):
    """Test completion generation on MMLU questions."""
    print(f"\n🧪 Testing MMLU Completion Generation")
    print(f"Model: {model_name}")
    print(f"Questions: {num_questions}")
    print(f"{'='*60}")
    
    try:
        # Check API key
        model_config = load_model_config()
        model_cfg = model_config[model_name]
        provider = model_cfg["provider"]
        key_var = f"API_KEY_{provider.upper()}"
        
        if not os.getenv(key_var):
            print(f"❌ API key {key_var} not set")
            print(f"Please add your API key to .env file:")
            print(f"{key_var}=your_actual_key_here")
            return
        
        # Load dataset
        print("📁 Loading MMLU data...")
        registry = DatasetRegistry()
        questions = registry.load_dataset("mmlu", "dev")
        
        # Load hints
        with open("data/mmlu/hints/sycophancy.json", 'r') as f:
            hints_data = json.load(f)
            hints = [HintRecord(**h) for h in hints_data]
        
        hints_by_id = {h.question_id: h for h in hints}
        
        # Limit to sample
        sample_questions = questions[:num_questions]
        print(f"✅ Loaded {len(sample_questions)} sample questions")
        
        # Create client
        print(f"🔌 Creating {model_name} client...")
        client = create_client(
            provider=model_cfg["provider"],
            model_id=model_cfg["model_id"],
            temperature=0,
            max_tokens=500  # Short for testing
        )
        print("✅ Client created successfully")
        
        # Test each question
        for i, question in enumerate(sample_questions):
            print(f"\n" + "="*40)
            print(f"QUESTION {i+1}/{len(sample_questions)}")
            print(f"ID: {question.question_id}")
            print(f"="*40)
            
            # Show the question
            print(f"Q: {question.question[:100]}...")
            choices_str = ", ".join([f"{k}: {v[:20]}..." for k, v in question.choices.items()])
            print(f"Choices: {choices_str}")
            print(f"Correct: {question.correct}")
            
            # Get hint
            hint = hints_by_id.get(question.question_id)
            if hint:
                print(f"Hint → {hint.hint_option} ({'✅' if hint.is_correct_option else '❌'})")
                print(f"Hint: {hint.hint_text[:50]}...")
                switch_case = hint.hint_option != question.correct
                print(f"Switch case: {'Yes' if switch_case else 'No'}")
            
            # Test baseline completion
            print(f"\n🔵 Testing BASELINE completion...")
            baseline_messages = build_baseline_prompt(question)
            baseline_response = await client.complete(baseline_messages)
            baseline_answer = baseline_response.content.strip()[-1] if baseline_response.content.strip() else "?"
            print(f"Baseline answer: {baseline_answer}")
            
            # Test hinted completion
            if hint:
                print(f"\n🟡 Testing HINTED completion...")
                hinted_messages = build_hinted_prompt(question, hint)
                hinted_response = await client.complete(hinted_messages)
                hinted_answer = hinted_response.content.strip()[-1] if hinted_response.content.strip() else "?"
                print(f"Hinted answer: {hinted_answer}")
                
                # Check for switch
                if baseline_answer != hinted_answer:
                    print(f"🔄 SWITCH DETECTED: {baseline_answer} → {hinted_answer}")
                    if hinted_answer == hint.hint_option:
                        print(f"✨ Switched TO hint option!")
                else:
                    print(f"➡️  No switch: both answered {baseline_answer}")
            
            print(f"\nUsage - Baseline: {baseline_response.usage}")
            if hint:
                print(f"Usage - Hinted: {hinted_response.usage}")
            
            # Pause between questions
            await asyncio.sleep(1)
        
        print(f"\n🎉 MMLU completion test completed successfully!")
        print(f"The system is ready for full experiments.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    setup_logging()
    
    print("This script tests MMLU completion generation on a small sample.")
    print("Make sure you have:")
    print("1. Run: python scripts/00_download_format.py --dataset mmlu --split dev --hint-families sycophancy")
    print("2. Added API keys to .env file")
    print()
    
    # Check if data exists
    if not os.path.exists("data/mmlu/processed/dev.json"):
        print("❌ MMLU data not found. Please run:")
        print("python scripts/00_download_format.py --dataset mmlu --split dev --hint-families sycophancy")
        return
    
    # Run test
    await test_mmlu_completions(
        model_name="claude-3-5-sonnet",  # Change this to test different models
        num_questions=3
    )

if __name__ == "__main__":
    asyncio.run(main())