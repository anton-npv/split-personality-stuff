"""Base classes and types for dataset handling."""

from typing import Dict, List
from pydantic import BaseModel


class MCQRecord(BaseModel):
    """Standard multiple-choice question record."""
    question_id: int
    question: str
    choices: Dict[str, str]  # Option letter -> choice text (A, B, C, D, E, etc.)
    correct: str  # Correct option letter
    
    def get_choices(self) -> Dict[str, str]:
        """Get choices as option letter -> text mapping."""
        return self.choices
    
    def get_choice_text(self, option: str) -> str:
        """Get text for a specific option."""
        return self.choices[option]
    
    def get_option_letters(self) -> List[str]:
        """Get all available option letters."""
        return list(self.choices.keys())


class HintRecord(BaseModel):
    """Hint record for a specific question."""
    question_id: int
    hint_option: str  # Can be A, B, C, D, E, etc.
    is_correct_option: bool
    hint_text: str