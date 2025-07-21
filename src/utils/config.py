"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_file = Path("configs") / f"{config_path}.yaml"
    if not config_file.exists():
        config_file = Path(config_path)
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_dataset_config() -> Dict[str, Any]:
    """Load dataset configuration."""
    return load_config("datasets")


def load_model_config() -> Dict[str, Any]:
    """Load model configuration."""
    return load_config("models")


def load_hint_config() -> Dict[str, Any]:
    """Load hint configuration."""
    return load_config("hints")