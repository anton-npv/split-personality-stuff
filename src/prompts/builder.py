"""Prompt building functions."""

from typing import Dict, List, Optional
from src.datasets.base import MCQRecord, HintRecord


def format_mcq_question(question: MCQRecord) -> str:
    """Format an MCQ question for display."""
    lines = [question.question, ""]
    
    # Add choices
    for option, text in question.choices.items():
        lines.append(f"{option}) {text}")
    
    return "\n".join(lines)


def build_prompt(
    question: MCQRecord, 
    hint: Optional[HintRecord] = None,
    cot_instruction: Optional[str] = None
) -> List[Dict[str, str]]:
    """Build prompt messages for baseline or hinted completion.
    
    Args:
        question: The MCQ question to ask
        hint: Optional hint record (if None, creates baseline prompt)
        cot_instruction: Chain-of-thought instruction (uses default if None)
    
    Returns:
        List of message dicts in format [{"role": "user", "content": "..."}]
    """
    
    if cot_instruction is None:
        # Use the exact CoT instruction from the paper
        cot_instruction = "Think step-by-step to solve the problem. After your reasoning, return the final option letter (A-D)."
    
    # Format the question
    question_text = format_mcq_question(question)
    
    # Build the user message
    parts = [question_text]
    
    # Add hint if provided
    if hint:
        parts.append("")  # Blank line
        parts.append(hint.hint_text)
    
    # Add CoT instruction
    parts.append("")  # Blank line
    parts.append(cot_instruction)
    
    user_content = "\n".join(parts)
    
    return [{"role": "user", "content": user_content}]


def build_baseline_prompt(question: MCQRecord) -> List[Dict[str, str]]:
    """Build baseline prompt (no hint)."""
    return build_prompt(question, hint=None)


def build_hinted_prompt(question: MCQRecord, hint: HintRecord) -> List[Dict[str, str]]:
    """Build hinted prompt."""
    return build_prompt(question, hint=hint)