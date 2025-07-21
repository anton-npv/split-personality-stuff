#!/usr/bin/env python3
"""Download and format dataset into canonical MCQ format with hint generation.

Usage:
    python scripts/00_download_format.py --dataset mmlu --split dev
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List

from src.datasets import DatasetRegistry
from src.datasets.base import HintRecord
from src.utils.config import load_hint_config
from src.utils.paths import get_dataset_path, ensure_dir
from src.utils.logging_setup import setup_logging


def download_mmlu(root_path: Path, split: str) -> None:
    """Download MMLU dataset from HuggingFace."""
    from datasets import load_dataset
    import json
    
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading MMLU dataset to {root_path}")
    
    # Create directories
    raw_dir = ensure_dir(root_path / "raw")
    
    # Download from HuggingFace datasets - use "all" config to get combined dataset
    dataset = load_dataset("cais/mmlu", "all", split=split)
    
    # Save as JSON to preserve list structure
    json_file = raw_dir / f"{split}.json"
    with open(json_file, 'w') as f:
        json.dump(list(dataset), f, indent=2)
    
    logger.info(f"MMLU {split} dataset saved to {json_file}")
    logger.info("MMLU dataset downloaded successfully")


def generate_hints(questions: List, hint_families: List[str], hint_config: dict) -> dict:
    """Generate hints for each question and hint family."""
    logger = logging.getLogger(__name__)
    hints_by_family = {}
    
    for family in hint_families:
        if family not in hint_config:
            logger.warning(f"Hint family '{family}' not found in config")
            continue
            
        family_hints = []
        templates = hint_config[family]
        
        for question in questions:
            # Randomly select hint option from all available options
            available_options = question.get_option_letters()
            hint_option = random.choice(available_options)
            is_correct = hint_option == question.correct
            
            # Select random template and format
            template = random.choice(templates)
            hint_text = template.format(option=hint_option)
            
            hint_record = HintRecord(
                question_id=question.question_id,
                hint_option=hint_option,
                is_correct_option=is_correct,
                hint_text=hint_text
            )
            
            family_hints.append(hint_record)
        
        hints_by_family[family] = family_hints
        logger.info(f"Generated {len(family_hints)} hints for family '{family}'")
    
    return hints_by_family


def main():
    parser = argparse.ArgumentParser(description="Download and format dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--split", default="dev", help="Dataset split")
    parser.add_argument("--hint-families", nargs="+", 
                       default=["sycophancy", "visual_pattern", "metadata"],
                       help="Hint families to generate")
    parser.add_argument("--force-download", action="store_true",
                       help="Force re-download even if data exists")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    setup_logging()
    random.seed(args.seed)
    
    logger = logging.getLogger(__name__)
    
    # Load configurations
    registry = DatasetRegistry()
    hint_config = load_hint_config()
    
    # Get dataset paths
    dataset_root = get_dataset_path(args.dataset)
    processed_dir = ensure_dir(dataset_root / "processed") 
    hints_dir = ensure_dir(dataset_root / "hints")
    
    # Download raw data if needed
    if args.dataset == "mmlu":
        raw_dir = dataset_root / "raw"
        if not raw_dir.exists() or args.force_download:
            download_mmlu(dataset_root, args.split)
    
    # Load and process dataset
    logger.info(f"Processing {args.dataset} dataset (split: {args.split})")
    questions = registry.load_dataset(args.dataset, args.split)
    
    # Save processed questions
    processed_file = processed_dir / f"{args.split}.json"
    with open(processed_file, 'w') as f:
        questions_data = [q.model_dump() for q in questions]
        json.dump(questions_data, f, indent=2)
    
    logger.info(f"Saved {len(questions)} processed questions to {processed_file}")
    
    # Generate hints
    hints_by_family = generate_hints(questions, args.hint_families, hint_config)
    
    # Save hint files
    for family, hints in hints_by_family.items():
        hint_file = hints_dir / f"{family}.json"
        with open(hint_file, 'w') as f:
            hints_data = [h.model_dump() for h in hints]
            json.dump(hints_data, f, indent=2)
        
        logger.info(f"Saved {len(hints)} {family} hints to {hint_file}")
    
    logger.info("Dataset processing complete!")


if __name__ == "__main__":
    main()