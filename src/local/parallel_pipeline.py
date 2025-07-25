"""Parallel pipeline for local model inference using accelerate."""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object

from ..datasets.base import MCQRecord, HintRecord
from ..prompts import build_baseline_prompt, build_hinted_prompt
from .model_handler import generate_completion_batch

logger = logging.getLogger(__name__)


def load_data(path: Path) -> List[Dict]:
    """Load JSON data from file."""
    if not path.exists():
        logger.error(f"Data file not found: {path}")
        return []
    
    try:
        with path.open() as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return []


def save_results(
    results: List[Dict],
    output_path: Path,
    dataset: str,
    hint_type: str,
    model_name: str
) -> None:
    """Save completion results to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    logger.info(f"Saved {len(results)} completions to {output_path}")


def generate_parallel_completions(
    accelerator: Accelerator,
    model,
    tokenizer,
    model_name: str,
    device,
    questions: List[MCQRecord],
    hints: List[HintRecord],
    hint_type: str,
    output_path: Path,
    batch_size: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    resume: bool = False
) -> None:
    """Generate completions in parallel across GPUs."""
    
    start_time = time.time()
    
    if accelerator.is_main_process:
        logger.info(f"Running parallel completion on {accelerator.num_processes} GPU(s)")
        logger.info(f"Processing {len(questions)} questions with hint type: {hint_type}")
    
    # Check for resume
    completed_ids = set()
    if resume and output_path.exists():
        with output_path.open("r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_ids.add(record.get("question_id"))
                    except:
                        continue
        
        if accelerator.is_main_process:
            logger.info(f"Resuming: found {len(completed_ids)} completed questions")
    
    # Filter out completed questions
    if completed_ids:
        questions = [q for q in questions if q.question_id not in completed_ids]
        hints = [h for h in hints if h.question_id not in completed_ids]
        
        if accelerator.is_main_process:
            logger.info(f"Processing remaining {len(questions)} questions")
    
    if not questions:
        if accelerator.is_main_process:
            logger.info("No questions to process")
        return
    
    # Distribute data across processes
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    
    # Split data evenly across processes
    questions_per_process = questions[rank::world_size]
    hints_per_process = hints[rank::world_size]
    
    logger.info(f"Process {rank}: processing {len(questions_per_process)} questions")
    
    # Create hint lookup
    hint_dict = {h.question_id: h for h in hints_per_process}
    
    # Build prompts
    prompts = []
    for question in questions_per_process:
        hint_record = hint_dict.get(question.question_id)
        
        # Build prompt based on hint type
        if hint_type == "none":
            messages = build_baseline_prompt(question)
            hint_info = None
        else:
            if hint_record is None:
                logger.warning(f"No hint found for question {question.question_id}")
                continue
            
            messages = build_hinted_prompt(question, hint_record)
            hint_info = {
                "hint_option": hint_record.hint_option,
                "is_correct_option": hint_record.hint_option == question.correct,
                "hint_text": hint_record.hint_text
            }
        
        prompts.append({
            "question_id": question.question_id,
            "messages": messages,
            "hint_type": hint_type,
            "hint_info": hint_info
        })
    
    # Generate completions with periodic checkpointing
    logger.info(f"Process {rank}: generating completions for {len(prompts)} prompts")
    
    # Process in smaller chunks for checkpointing
    checkpoint_interval = 100  # Save every 100 completions
    final_results = []
    
    for chunk_start in range(0, len(prompts), checkpoint_interval):
        chunk_end = min(chunk_start + checkpoint_interval, len(prompts))
        chunk_prompts = prompts[chunk_start:chunk_end]
        
        # Convert prompts to format expected by generate_completion_batch
        batch_prompts = []
        for prompt_data in chunk_prompts:
            batch_prompts.append({
                "question_id": prompt_data["question_id"],
                "messages": prompt_data["messages"]
            })
        
        completion_results = generate_completion_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=batch_prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Combine completion results with metadata
        for i, completion_result in enumerate(completion_results):
            original_prompt = chunk_prompts[i]
            
            final_result = {
                "question_id": completion_result["question_id"],
                "hint_type": hint_type,
                "prompt": original_prompt["messages"],
                "completion": completion_result["completion"],
                "model": model_name
            }
            
            if original_prompt["hint_info"]:
                final_result["hint_info"] = original_prompt["hint_info"]
            
            final_results.append(final_result)
        
        # Checkpoint: gather and save current results
        if (chunk_end % checkpoint_interval == 0 or chunk_end == len(prompts)) and len(final_results) > 0:
            accelerator.wait_for_everyone()
            gathered_chunk = gather_object(final_results)
            
            if accelerator.is_main_process:
                # Flatten chunk results
                chunk_to_save = []
                for process_results in gathered_chunk:
                    if isinstance(process_results, list):
                        chunk_to_save.extend(process_results)
                    else:
                        chunk_to_save.append(process_results)
                
                # Append to output file
                with output_path.open("a") as f:
                    for result in chunk_to_save:
                        f.write(json.dumps(result) + "\n")
                
                logger.info(f"Checkpoint: saved {len(chunk_to_save)} completions (total processed: {chunk_end * world_size})")
            
            # Clear results for next chunk
            final_results = []
            accelerator.wait_for_everyone()
    
    logger.info(f"Process {rank}: generated all completions")
    
    # Final synchronization
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # Count total completions saved
        total_saved = 0
        if output_path.exists():
            with output_path.open("r") as f:
                total_saved = sum(1 for line in f if line.strip())
        
        elapsed = time.time() - start_time
        logger.info(f"Parallel completion finished in {elapsed:.2f}s")
        logger.info(f"Total completions: {total_saved}")
        logger.info(f"Throughput: {total_saved/elapsed:.2f} completions/sec")