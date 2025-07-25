"""Utility functions for completion generation."""

import asyncio
import json
import logging
from pathlib import Path
from typing import List

from src.datasets.base import MCQRecord, HintRecord
from src.prompts import build_baseline_prompt, build_hinted_prompt
from src.utils.paths import ensure_dir
from src.utils.jsonl import append_jsonl

logger = logging.getLogger(__name__)


async def generate_completions_async(
    questions: List[MCQRecord],
    hints: List[HintRecord],
    client,
    output_path: Path,
    hint_type: str,
    resume: bool = False
) -> None:
    """Generate completions for questions asynchronously."""
    
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
            logger.debug(f"Messages type: {type(messages)}, content: {messages}")
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