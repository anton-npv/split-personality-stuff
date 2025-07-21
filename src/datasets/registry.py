"""Dataset registry for dynamic loading."""

import importlib
import logging
from pathlib import Path
from typing import List, Dict, Any
from ..utils.config import load_dataset_config
from .base import MCQRecord

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Registry for loading datasets dynamically."""
    
    def __init__(self):
        self.config = load_dataset_config()
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        return list(self.config.keys())
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get configuration for a specific dataset."""
        if dataset_name not in self.config:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        return self.config[dataset_name]
    
    def load_dataset(self, dataset_name: str, split: str = None) -> List[MCQRecord]:
        """Load a dataset using its registered parser."""
        dataset_config = self.get_dataset_info(dataset_name)
        
        # Use default split if not specified
        if split is None:
            split = dataset_config.get("default_split", "dev")
        
        # Validate split
        available_splits = dataset_config.get("splits", [])
        if split not in available_splits:
            raise ValueError(
                f"Split '{split}' not available for {dataset_name}. "
                f"Available splits: {available_splits}"
            )
        
        # Get parser function
        parser_path = dataset_config["parser"]
        module_name, function_name = parser_path.rsplit(".", 1)
        
        try:
            # Import module dynamically  
            module = importlib.import_module(f"src.{module_name}")
            parser_func = getattr(module, function_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Cannot import parser {parser_path}: {e}")
        
        # Load dataset
        root_path = dataset_config["root"]
        logger.info(f"Loading {dataset_name} dataset (split: {split})")
        
        return parser_func(root_path, split)