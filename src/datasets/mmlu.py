"""MMLU dataset parser."""

import csv
import logging
from pathlib import Path
from typing import List, Dict
from .base import MCQRecord

logger = logging.getLogger(__name__)


def parse_raw(root: str, split: str = "dev") -> List[MCQRecord]:
    """Parse MMLU dataset from JSON files.
    
    MMLU format from HuggingFace: question, subject, choices (list), answer (0-3 index)
    """
    root_path = Path(root)
    json_file = root_path / "raw" / f"{split}.json"
    
    if not json_file.exists():
        raise FileNotFoundError(f"MMLU {split} file not found: {json_file}")
    
    records = []
    
    import json
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for idx, item in enumerate(data):
        # Extract question
        question = item['question'].strip()
        
        # Get choices list directly
        choices_list = item['choices']
        
        # Map choices to letters
        choices = {}
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        for i, choice in enumerate(choices_list):
            if i < len(option_letters):
                choices[option_letters[i]] = choice.strip()
        
        # Ensure we have at least 2 choices
        if len(choices) < 2:
            logger.warning(f"Question {idx} has fewer than 2 choices, skipping")
            continue
        
        # Get correct answer - convert from index to letter
        answer_index = int(item['answer'])
        if answer_index >= len(choices_list):
            logger.warning(f"Question {idx} answer index {answer_index} out of range, skipping")
            continue
        
        correct = option_letters[answer_index]
        
        record = MCQRecord(
            question_id=idx,
            question=question,
            choices=choices,
            correct=correct
        )
        records.append(record)
    
    logger.info(f"Loaded {len(records)} MMLU {split} questions")
    return records