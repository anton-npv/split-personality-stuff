#!/usr/bin/env python3
"""Export switched completions in conversational format for fine-tuning.

This script extracts completions where the model switched to the hint and exports them
in a conversational format suitable for fine-tuning, along with associated metadata.

Usage:
    python scripts/utils/export_switched_data.py --dataset mmlu --model gemma-3-4b-local --hint sycophancy
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from src.utils.paths import get_completion_path, get_evaluation_path, get_dataset_path
from src.utils.jsonl import read_jsonl
from src.utils.logging_setup import setup_logging


def load_switched_data(dataset: str, model: str, hint: str) -> tuple:
    """Load all necessary data files for switched completions."""
    eval_dir = get_evaluation_path(dataset, model, hint)
    completion_dir = get_completion_path(dataset, model, hint)
    
    # Load switches data
    switches_path = eval_dir / "switches.json"
    with open(switches_path, 'r') as f:
        switches_data = json.load(f)
    
    # Load CoT verification data
    cot_path = eval_dir / "cot_verification.json"
    with open(cot_path, 'r') as f:
        cot_data = json.load(f)
    
    # Load completions
    completions_path = completion_dir / "completions.jsonl"
    completions_by_id = {}
    for comp in read_jsonl(completions_path):
        completions_by_id[comp["question_id"]] = comp
    
    # Load hints data
    hints_path = get_dataset_path(dataset) / "hints" / f"{hint}.json"
    with open(hints_path, 'r') as f:
        hints_data = json.load(f)
    hints_by_id = {h["question_id"]: h for h in hints_data}
    
    return switches_data, cot_data, completions_by_id, hints_by_id


def export_conversational_format(
    switches_data: Dict,
    cot_data: Dict, 
    completions_by_id: Dict,
    hints_by_id: Dict,
    output_dir: Path
) -> None:
    """Export switched completions in conversational format."""
    logger = logging.getLogger(__name__)
    
    # Get list of question IDs that switched to hint
    switched_ids = switches_data["switched_to_hint"]
    
    # Prepare conversational data
    conversations = []
    question_metadata = []
    
    for qid in switched_ids:
        qid_str = str(qid)
        
        if qid not in completions_by_id:
            logger.warning(f"Missing completion for question {qid}")
            continue
            
        completion = completions_by_id[qid]
        
        # Extract user message from prompt
        prompt_messages = completion.get("prompt", [])
        if prompt_messages and isinstance(prompt_messages, list):
            user_content = "\n".join([msg.get("content", "") for msg in prompt_messages if msg.get("role") == "user"])
        else:
            user_content = str(prompt_messages)
        
        # Create conversation format (without system prompt)
        conversation = {
            "question_id": qid,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": completion["completion"]}
            ]
        }
        conversations.append(conversation)
        
        # Compile metadata for this question
        switch_detail = switches_data["switch_details"].get(qid_str, {})
        cot_verification = cot_data["verification_results"].get(qid_str, {})
        hint_info = hints_by_id.get(qid, {})
        
        metadata = {
            "question_id": qid,
            
            # Switch information
            "switch_info": {
                "baseline_answer": switch_detail.get("baseline_answer"),
                "hinted_answer": switch_detail.get("hinted_answer"),
                "hint_option": switch_detail.get("hint_option"),
                "correct_answer": switch_detail.get("correct_answer"),
                "baseline_is_correct": switch_detail.get("baseline_is_correct"),
                "hinted_is_correct": switch_detail.get("hinted_is_correct"),
            },
            
            # CoT verification information
            "cot_verification": {
                "mentions_hint": cot_verification.get("mentions_hint", False),
                "verbalizes_hint": cot_verification.get("verbalizes_hint", False),
                "depends_on_hint": cot_verification.get("depends_on_hint", False),
                "uses_hint_only_for_verification": cot_verification.get("uses_hint_only_for_verification", False),
                "quartiles": cot_verification.get("quartiles", []),
                "explanation": cot_verification.get("explanation", "")
            },
            
            # Hint information
            "hint_info": {
                "hint_text": hint_info.get("hint_text", ""),
                "hint_option": hint_info.get("hint_option", ""),
                "is_correct_option": hint_info.get("is_correct_option", False)
            }
        }
        question_metadata.append(metadata)
    
    # Save conversations as JSONL only
    conversations_jsonl_path = output_dir / "switched_conversations.jsonl"
    with open(conversations_jsonl_path, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')
    logger.info(f"Saved {len(conversations)} conversations to {conversations_jsonl_path}")
    
    # Save metadata with common fields at top level
    metadata_structure = {
        "dataset": switches_data["dataset"],
        "model": switches_data["model"],
        "hint_type": switches_data["hint_type"],
        "total_switched": len(conversations),
        "questions": question_metadata
    }
    
    metadata_path = output_dir / "switched_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_structure, f, indent=2)
    logger.info(f"Saved metadata for {len(question_metadata)} questions to {metadata_path}")
    
    # Print summary statistics
    logger.info("\n=== Export Summary ===")
    logger.info(f"Total switched completions: {len(conversations)}")
    
    # Count how many verbalized the hint
    verbalized_count = sum(1 for m in question_metadata if m["cot_verification"]["verbalizes_hint"])
    logger.info(f"Verbalizes hint: {verbalized_count} ({verbalized_count/len(question_metadata)*100:.1f}%)")
    
    # Count accuracy changes
    correct_to_incorrect = sum(1 for m in question_metadata 
                              if m["switch_info"]["baseline_is_correct"] 
                              and not m["switch_info"]["hinted_is_correct"])
    incorrect_to_correct = sum(1 for m in question_metadata 
                              if not m["switch_info"]["baseline_is_correct"] 
                              and m["switch_info"]["hinted_is_correct"])
    
    logger.info(f"Correct → Incorrect: {correct_to_incorrect}")
    logger.info(f"Incorrect → Correct: {incorrect_to_correct}")


def main():
    parser = argparse.ArgumentParser(description="Export switched completions for fine-tuning")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--hint", required=True, help="Hint type")
    parser.add_argument("--output-dir", help="Output directory (default: data/{dataset}/exports/{model}/{hint}/)")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"data/{args.dataset}/exports/{args.model}/{args.hint}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load all data
    logger.info("Loading data files...")
    switches_data, cot_data, completions_by_id, hints_by_id = load_switched_data(
        args.dataset, args.model, args.hint
    )
    
    # Export data
    logger.info("Exporting switched completions...")
    export_conversational_format(
        switches_data, cot_data, completions_by_id, hints_by_id, output_dir
    )
    
    logger.info("Export complete!")


if __name__ == "__main__":
    main()