#!/usr/bin/env python3
"""Preview MMLU prompts without calling APIs."""

import json
import os
from src.datasets import DatasetRegistry
from src.datasets.base import MCQRecord, HintRecord
from src.prompts import build_baseline_prompt, build_hinted_prompt

def preview_mmlu_prompts(num_questions: int = 3):
    """Show what the MMLU prompts look like."""
    print(f"🔍 MMLU Prompt Preview")
    print(f"Showing {num_questions} sample questions")
    print(f"{'='*80}")
    
    try:
        # Check if data exists
        if not os.path.exists("data/mmlu/processed/dev.json"):
            print("❌ MMLU data not found. Please run:")
            print("make download")
            return
        
        # Load dataset
        registry = DatasetRegistry()
        questions = registry.load_dataset("mmlu", "dev")
        
        # Load hints
        with open("data/mmlu/hints/sycophancy.json", 'r') as f:
            hints_data = json.load(f)
            hints = [HintRecord(**h) for h in hints_data]
        
        hints_by_id = {h.question_id: h for h in hints}
        
        # Sample questions
        sample_questions = questions[:num_questions]
        
        for i, question in enumerate(sample_questions):
            hint = hints_by_id.get(question.question_id)
            
            print(f"\n{'='*80}")
            print(f"SAMPLE {i+1}: Question ID {question.question_id}")
            print(f"Correct Answer: {question.correct}")
            if hint:
                print(f"Hint Points To: {hint.hint_option} ({'Correct' if hint.is_correct_option else 'Incorrect'})")
                switch_potential = hint.hint_option != question.correct
                print(f"Switch Potential: {'YES - hint differs from correct' if switch_potential else 'NO - hint matches correct'}")
            print(f"{'='*80}")
            
            # Baseline prompt
            print(f"\n🔵 BASELINE PROMPT:")
            print(f"-" * 40)
            baseline_messages = build_baseline_prompt(question)
            print(baseline_messages[0]["content"])
            
            # Hinted prompt
            if hint:
                print(f"\n🟡 HINTED PROMPT:")
                print(f"-" * 40)
                hinted_messages = build_hinted_prompt(question, hint)
                print(hinted_messages[0]["content"])
            
            print(f"\n" + "="*80)
        
        print(f"\n✨ Prompt Preview Complete!")
        print(f"\nNext steps:")
        print(f"1. Add API keys to .env file")
        print(f"2. Run: python test_llm_basic.py (to test basic connectivity)")
        print(f"3. Run: python test_mmlu_completions.py (to test MMLU with real API calls)")
        print(f"4. Run: make test (to run full pipeline with 3 questions)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    preview_mmlu_prompts()