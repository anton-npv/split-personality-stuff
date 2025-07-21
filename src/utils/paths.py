"""Path utilities for data organization."""

import os
from pathlib import Path
from typing import Union


def get_data_root() -> Path:
    """Get the root data directory."""
    return Path(os.getenv("DATA_ROOT", "data"))


def get_dataset_path(dataset_name: str, *parts: str) -> Path:
    """Get path within a dataset directory."""
    path = get_data_root() / dataset_name
    for part in parts:
        path = path / part
    return path


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_completion_path(dataset: str, model: str, hint_type: str) -> Path:
    """Get path for completion files."""
    return get_dataset_path(dataset, "completions", model, hint_type)


def get_evaluation_path(dataset: str, model: str, hint_type: str) -> Path:
    """Get path for evaluation files."""  
    return get_dataset_path(dataset, "evaluations", model, hint_type)