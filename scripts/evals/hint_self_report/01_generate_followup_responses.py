#!/usr/bin/env python3
"""Generate follow-up responses asking models about their hint usage.

This script takes the switched conversations and adds a follow-up turn where we ask
the model whether it used the hint. Uses local models with multi-GPU support.

Usage:
    # Single GPU
    python scripts/evals/hint_self_report/01_generate_followup_responses.py
    
    # Multi-GPU parallel (recommended)
    accelerate launch scripts/evals/hint_self_report/01_generate_followup_responses.py --parallel
"""

import argparse
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any
import torch
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.utils.paths import get_dataset_path
from src.utils.jsonl import read_jsonl, write_jsonl
from src.utils.logging_setup import setup_logging
from src.local.model_handler import LocalModelHandler


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_export_data(config: Dict[str, Any]) -> tuple:
    """Load the exported switched conversations and metadata."""
    export_dir = get_dataset_path(config['data']['dataset']) / "exports" / \
                 config['data']['source_model'] / config['data']['hint_type']
    
    conversations_path = export_dir / "switched_conversations.jsonl"
    metadata_path = export_dir / "switched_metadata.json"
    
    conversations = list(read_jsonl(conversations_path))
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return conversations, metadata


def create_followup_prompt(conversation: Dict, hint_info: Dict, question_template: str) -> str:
    """Create the follow-up prompt asking about hint usage."""
    # Get the hint text from the metadata
    hint_text = hint_info.get("hint_text", "the provided hint")
    
    # Format the question with the hint text
    question = question_template.replace("{hint_text}", f'"{hint_text}"')
    
    # Build the full conversation with the follow-up
    messages = conversation["messages"]
    
    # Create a prompt that includes the original question and response
    prompt = f"{messages[0]['content']}\n\n"
    prompt += f"Assistant: {messages[1]['content']}\n\n"
    prompt += f"Human: {question}\n\n"
    prompt += "Assistant:"
    
    return prompt


def generate_responses_batch(
    model_handler: LocalModelHandler,
    conversations: List[Dict],
    metadata: Dict,
    config: Dict,
    accelerator: Accelerator,
    start_idx: int = 0
) -> List[Dict]:
    """Generate follow-up responses for a batch of conversations."""
    logger = logging.getLogger(__name__)
    
    # Get the selected question template
    question_templates = {q['id']: q['template'] for q in config['followup_questions']}
    selected_template = question_templates[config['selected_question']]
    
    # Create a mapping of question_id to hint_info
    hint_info_by_id = {q["question_id"]: q["hint_info"] for q in metadata["questions"]}
    
    results = []
    batch_size = config['generation']['batch_size']
    
    # Process in batches
    for i in tqdm(range(start_idx, len(conversations), batch_size), 
                  desc="Generating responses", disable=not accelerator.is_main_process):
        batch_conversations = conversations[i:i+batch_size]
        batch_prompts = []
        
        for conv in batch_conversations:
            qid = conv["question_id"]
            hint_info = hint_info_by_id.get(qid, {})
            prompt = create_followup_prompt(conv, hint_info, selected_template)
            batch_prompts.append(prompt)
        
        # Generate responses
        try:
            responses = model_handler.generate_batch(
                batch_prompts,
                max_new_tokens=config['generation']['max_new_tokens'],
                temperature=config['generation']['temperature']
            )
            
            # Create extended conversations
            for j, (conv, response) in enumerate(zip(batch_conversations, responses)):
                extended_conv = {
                    "question_id": conv["question_id"],
                    "messages": conv["messages"] + [
                        {"role": "user", "content": selected_template.replace(
                            "{hint_text}", 
                            f'"{hint_info_by_id.get(conv["question_id"], {}).get("hint_text", "the provided hint")}"'
                        )},
                        {"role": "assistant", "content": response}
                    ],
                    "followup_metadata": {
                        "question_template_id": config['selected_question'],
                        "generation_model": config['generation']['model']
                    }
                }
                results.append(extended_conv)
                
        except Exception as e:
            logger.error(f"Error generating batch starting at index {i}: {e}")
            # Skip this batch and continue
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate follow-up responses about hint usage")
    parser.add_argument("--config", default="scripts/evals/hint_self_report/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--parallel", action="store_true",
                       help="Use multi-GPU parallel generation")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous run")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(Path(args.config))
    logger.info(f"Loaded configuration from {args.config}")
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load export data
    logger.info("Loading exported switched conversations...")
    conversations, metadata = load_export_data(config)
    logger.info(f"Loaded {len(conversations)} switched conversations")
    
    # Set up output path
    output_dir = get_dataset_path(config['data']['dataset']) / \
                 config['output']['followup_dir'].format(
                     source_model=config['data']['source_model'],
                     hint_type=config['data']['hint_type']
                 )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config['output']['extended_conversations']
    
    # Check for existing results if resuming
    start_idx = 0
    existing_results = []
    if args.resume and output_path.exists():
        existing_results = list(read_jsonl(output_path))
        existing_ids = {r["question_id"] for r in existing_results}
        # Filter out already processed conversations
        conversations = [c for c in conversations if c["question_id"] not in existing_ids]
        logger.info(f"Resuming: {len(existing_results)} already processed, {len(conversations)} remaining")
    
    if not conversations:
        logger.info("All conversations already processed!")
        return
    
    # Initialize model handler
    logger.info(f"Initializing model: {config['generation']['model']}")
    model_handler = LocalModelHandler(
        model_name=config['generation']['model'],
        device=accelerator.device,
        load_in_4bit=True  # Use 4-bit quantization for efficiency
    )
    
    # Distribute conversations across processes if parallel
    if args.parallel and accelerator.num_processes > 1:
        # Split conversations across processes
        conversations_per_process = len(conversations) // accelerator.num_processes
        start = accelerator.process_index * conversations_per_process
        end = start + conversations_per_process if accelerator.process_index < accelerator.num_processes - 1 else len(conversations)
        process_conversations = conversations[start:end]
        logger.info(f"Process {accelerator.process_index}: Processing {len(process_conversations)} conversations")
    else:
        process_conversations = conversations
    
    # Generate responses
    logger.info("Generating follow-up responses...")
    results = generate_responses_batch(
        model_handler, process_conversations, metadata, config, accelerator
    )
    
    # Gather results from all processes if parallel
    if args.parallel and accelerator.num_processes > 1:
        logger.info("Gathering results from all processes...")
        all_results = gather_object(results)
        
        # Only save on main process
        if accelerator.is_main_process:
            # Flatten the list of lists
            all_results = [item for sublist in all_results for item in sublist]
            # Combine with existing results if resuming
            all_results = existing_results + all_results
            
            # Save results
            write_jsonl(output_path, all_results)
            logger.info(f"Saved {len(all_results)} extended conversations to {output_path}")
            
            # Save metadata about this run
            run_metadata = {
                "dataset": config['data']['dataset'],
                "source_model": config['data']['source_model'],
                "hint_type": config['data']['hint_type'],
                "generation_model": config['generation']['model'],
                "question_template_id": config['selected_question'],
                "question_template": question_templates[config['selected_question']],
                "total_conversations": len(all_results)
            }
            
            metadata_path = output_dir / "generation_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(run_metadata, f, indent=2)
            logger.info(f"Saved generation metadata to {metadata_path}")
    else:
        # Single process - combine with existing results if resuming
        all_results = existing_results + results
        
        # Save results
        write_jsonl(output_path, all_results)
        logger.info(f"Saved {len(all_results)} extended conversations to {output_path}")
        
        # Save metadata about this run
        question_templates = {q['id']: q['template'] for q in config['followup_questions']}
        run_metadata = {
            "dataset": config['data']['dataset'],
            "source_model": config['data']['source_model'],
            "hint_type": config['data']['hint_type'],
            "generation_model": config['generation']['model'],
            "question_template_id": config['selected_question'],
            "question_template": question_templates[config['selected_question']],
            "total_conversations": len(all_results)
        }
        
        metadata_path = output_dir / "generation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(run_metadata, f, indent=2)
        logger.info(f"Saved generation metadata to {metadata_path}")
    
    logger.info("Follow-up response generation complete!")


if __name__ == "__main__":
    main()