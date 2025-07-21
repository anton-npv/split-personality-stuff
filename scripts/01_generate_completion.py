#!/usr/bin/env python3
"""Generate LLM completions for MCQ questions with baseline and hinted prompts.

Usage:
    python scripts/01_generate_completion.py --dataset mmlu --model claude-3-5-sonnet --hint none
    python scripts/01_generate_completion.py --dataset mmlu --model claude-3-5-sonnet --hint sycophancy
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.datasets import DatasetRegistry
from src.datasets.base import MCQRecord, HintRecord
from src.clients import create_client
from src.prompts import build_baseline_prompt, build_hinted_prompt
from src.utils.config import load_model_config
from src.utils.paths import get_completion_path, ensure_dir
from src.utils.jsonl import append_jsonl
from src.utils.logging_setup import setup_logging


async def generate_completions(
    questions: List[MCQRecord],
    hints: List[HintRecord],
    client,
    output_path: Path,
    hint_type: str,
    resume: bool = False
) -> None:
    """Generate completions for questions."""
    logger = logging.getLogger(__name__)
    
    # Create output directory
    ensure_dir(output_path.parent)
    
    # Check if resuming from existing file
    completed_ids = set()
    if resume and output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    completed_ids.add(data['question_id'])
        logger.info(f"Resuming: found {len(completed_ids)} completed questions")
    
    # Create lookup for hints
    hints_by_id = {h.question_id: h for h in hints} if hints else {}
    
    total_questions = len(questions)
    completed_count = len(completed_ids)
    
    logger.info(f"Generating completions: {total_questions - completed_count} remaining")
    
    for i, question in enumerate(questions):
        if question.question_id in completed_ids:
            continue
            
        try:
            # Build prompt based on hint type
            if hint_type == "none":
                messages = build_baseline_prompt(question)
                hint_info = None
            else:
                hint = hints_by_id.get(question.question_id)
                if not hint:
                    logger.warning(f"No hint found for question {question.question_id}, skipping")
                    continue
                messages = build_hinted_prompt(question, hint)
                hint_info = {
                    "hint_option": hint.hint_option,
                    "is_correct_option": hint.is_correct_option,
                    "hint_text": hint.hint_text
                }
            
            # Generate completion
            logger.info(f"Generating completion {i+1}/{total_questions} (question_id: {question.question_id})")
            response = await client.complete(messages)
            
            # Save result
            result = {
                "question_id": question.question_id,
                "hint_type": hint_type,
                "prompt": messages,
                "completion": response.content,
                "model": response.model,
                "hint_info": hint_info
            }
            
            append_jsonl(result, output_path)
            completed_count += 1
            
            logger.info(f"Progress: {completed_count}/{total_questions} completed")
            
            # Small delay to be respectful to APIs
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error generating completion for question {question.question_id}: {e}")
            continue
    
    logger.info(f"Completion generation finished: {completed_count}/{total_questions} successful")


async def main():
    parser = argparse.ArgumentParser(description="Generate LLM completions")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True, help="Model name from configs/models.yaml")
    parser.add_argument("--hint", required=True, help="Hint type ('none' for baseline, or hint family)")
    parser.add_argument("--split", default="dev", help="Dataset split")
    parser.add_argument("--resume", action="store_true", help="Resume from existing completions file")
    parser.add_argument("--max-questions", type=int, help="Limit number of questions for testing")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configurations
    registry = DatasetRegistry()
    model_config = load_model_config()
    
    if args.model not in model_config:
        raise ValueError(f"Model '{args.model}' not found in config. Available: {list(model_config.keys())}")
    
    model_cfg = model_config[args.model]
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset (split: {args.split})")
    questions = registry.load_dataset(args.dataset, args.split)
    
    if args.max_questions:
        questions = questions[:args.max_questions]
        logger.info(f"Limited to {len(questions)} questions for testing")
    
    # Load hints if not baseline
    hints = []
    if args.hint != "none":
        from src.utils.paths import get_dataset_path
        hint_file = get_dataset_path(args.dataset, "hints", f"{args.hint}.json")
        if not hint_file.exists():
            raise FileNotFoundError(f"Hint file not found: {hint_file}")
        
        with open(hint_file, 'r') as f:
            hint_data = json.load(f)
            hints = [HintRecord(**h) for h in hint_data]
        
        logger.info(f"Loaded {len(hints)} hints for '{args.hint}' family")
    
    # Create LLM client
    logger.info(f"Creating {args.model} client")
    client = create_client(
        provider=model_cfg["provider"],
        model_id=model_cfg["model_id"],
        temperature=model_cfg.get("temperature", 0),
        max_tokens=model_cfg.get("max_tokens", 4096)
    )
    
    # Set output path
    output_dir = get_completion_path(args.dataset, args.model, args.hint)
    output_file = output_dir / "completions.jsonl"
    
    logger.info(f"Output file: {output_file}")
    
    # Generate completions
    await generate_completions(
        questions=questions,
        hints=hints,
        client=client,
        output_path=output_file,
        hint_type=args.hint,
        resume=args.resume
    )
    
    logger.info("Completion generation complete!")


if __name__ == "__main__":
    asyncio.run(main())