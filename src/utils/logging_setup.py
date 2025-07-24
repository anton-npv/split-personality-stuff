"""Logging configuration."""

import logging
import sys
from pathlib import Path
from typing import Optional


class FlushFileHandler(logging.FileHandler):
    """File handler that flushes after every log entry for real-time updates."""
    
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """Set up logging configuration."""
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Handle both string level names and integer levels
    if isinstance(level, str):
        log_level = getattr(logging, level.upper())
    else:
        log_level = level
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Create file handler with immediate flushing
        file_handler = FlushFileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=log_level,
        format=format_string,
        handlers=handlers,
    )
    
    # Reduce noise from HTTP libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)