#!/usr/bin/env python3
"""Generate LLM completions using local models with multi-GPU parallelization.

This script uses HuggingFace's accelerate library for distributed inference across multiple GPUs.
It's designed for local model inference and provides significant speedup over API-based inference.

Setup:
    1. Install dependencies: pip install accelerate bitsandbytes
    2. Configure accelerate: accelerate config
       - Select: this machine, multi-gpu, 1 node, default settings, N GPUs, bf16
    3. Set environment variable if needed: export NCCL_TIMEOUT=1800

Usage:
    # Single process (compatibility mode)
    python scripts/01_generate_completion_parallel.py --dataset mmlu --model gemma-3-4b-local --hint none
    
    # Multi-GPU parallel mode
    accelerate launch scripts/01_generate_completion_parallel.py --dataset mmlu --model gemma-3-4b-local --hint sycophancy --parallel
    
    # With custom batch size and resume
    accelerate launch scripts/01_generate_completion_parallel.py --dataset mmlu --model gemma-3-4b-local --hint sycophancy --parallel --batch-size 32 --resume
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.datasets import DatasetRegistry
from src.utils.config import load_model_config
from src.utils.paths import get_completion_path
from src.utils.logging_setup import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Generate completions with local models")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., mmlu)")
    parser.add_argument("--model", required=True, help="Model name from configs/models.yaml")
    parser.add_argument("--hint", required=True, help="Hint type (none, sycophancy, etc.)")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--n-questions", type=int, help="Limit number of questions")
    parser.add_argument("--batch-size", type=int, help="Override batch size per GPU")
    parser.add_argument("--parallel", action="store_true", help="Use multi-GPU parallelization")
    parser.add_argument("--resume", action="store_true", help="Resume from existing completions")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)
    
    # Load model configuration
    all_model_configs = load_model_config()
    if args.model not in all_model_configs:
        logger.error(f"Model {args.model} not found in configs/models.yaml")
        return 1
    
    model_config = all_model_configs[args.model]
    if model_config["provider"] != "local":
        logger.error(f"Model {args.model} is not a local model (provider: {model_config['provider']})")
        logger.error("Use scripts/01_generate_completion.py for non-local models")
        return 1
    
    # Check if parallel mode is requested
    if args.parallel:
        try:
            from accelerate import Accelerator
            from accelerate.utils import InitProcessGroupKwargs
            from datetime import timedelta
            
            # Initialize accelerator with timeout
            timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
            accelerator = Accelerator(kwargs_handlers=[timeout_kwargs])
            
            if accelerator.is_main_process:
                logger.info(f"Using parallel mode with {accelerator.num_processes} GPU(s)")
            
            return run_parallel_completion(args, accelerator, logger)
            
        except ImportError:
            logger.error("Accelerate not available. Install with: pip install accelerate")
            return 1
        except Exception as e:
            logger.error(f"Failed to initialize accelerate: {e}")
            logger.info("Falling back to single-process mode")
            return run_single_process_completion(args, logger)
    else:
        return run_single_process_completion(args, logger)


def run_parallel_completion(args, accelerator, logger):
    """Run completion generation in parallel mode."""
    from src.local.model_handler import load_model_and_tokenizer
    from src.local.parallel_pipeline import generate_parallel_completions
    
    # Load model configuration
    all_model_configs = load_model_config()
    model_config = all_model_configs[args.model]
    
    # Override batch size if specified
    if args.batch_size:
        model_config["batch_size"] = args.batch_size
    
    # Load dataset
    dataset_registry = DatasetRegistry()
    questions = dataset_registry.load_dataset(args.dataset, args.split)
    
    # Load hints if not baseline
    hints = []
    if args.hint != "none":
        from src.utils.paths import get_dataset_path
        from src.datasets.base import HintRecord
        hint_file = get_dataset_path(args.dataset, "hints", f"{args.hint}.json")
        if not hint_file.exists():
            raise FileNotFoundError(f"Hint file not found: {hint_file}")
        
        with open(hint_file, 'r') as f:
            hint_data = json.load(f)
            hints = [HintRecord(**h) for h in hint_data]
        
        logger.info(f"Loaded {len(hints)} hints for '{args.hint}' family")
    
    # Limit questions if specified
    if args.n_questions:
        questions = questions[:args.n_questions]
        hints = hints[:args.n_questions]
    
    if accelerator.is_main_process:
        logger.info(f"Loaded {len(questions)} questions and {len(hints)} hints")
    
    # Load model and tokenizer
    model_path = model_config.get("model_path", model_config.get("model_id", args.model))
    quantization = model_config.get("quantization")
    
    if accelerator.is_main_process:
        logger.info(f"Loading model: {model_path}")
    
    # Don't use device_map with accelerate/DDP
    model, tokenizer, model_name, device = load_model_and_tokenizer(
        model_path=model_path,
        quantization=quantization
    )
    
    # Prepare model with accelerator
    model, tokenizer = accelerator.prepare(model, tokenizer)
    device = accelerator.device
    
    if accelerator.is_main_process:
        logger.info(f"Model loaded on {accelerator.num_processes} GPU(s)")
    
    # Set up output path
    output_path = get_completion_path(args.dataset, args.model, args.hint) / "completions.jsonl"
    
    # Generate completions
    generate_parallel_completions(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        device=device,
        questions=questions,
        hints=hints,
        hint_type=args.hint,
        output_path=output_path,
        batch_size=model_config.get("batch_size", 8),
        max_new_tokens=model_config.get("max_tokens", 2048),
        temperature=model_config.get("temperature", 0.0),
        resume=args.resume
    )
    
    if accelerator.is_main_process:
        logger.info("Parallel completion generation finished")
    
    return 0


def run_single_process_completion(args, logger):
    """Run completion generation in single-process mode."""
    import asyncio
    from src.clients import create_client
    from src.datasets import DatasetRegistry
    
    logger.info("Using single-process mode")
    
    # Load model configuration and create client
    all_model_configs = load_model_config()
    model_config = all_model_configs[args.model].copy()
    
    # Extract provider to avoid duplication in kwargs
    provider = model_config.pop("provider")
    client = create_client(
        provider=provider,
        model_id=args.model,
        **model_config
    )
    
    # Load dataset
    dataset_registry = DatasetRegistry()
    questions = dataset_registry.load_dataset(args.dataset, args.split)
    
    # Load hints if not baseline
    hints = []
    if args.hint != "none":
        from src.utils.paths import get_dataset_path
        from src.datasets.base import HintRecord
        hint_file = get_dataset_path(args.dataset, "hints", f"{args.hint}.json")
        if not hint_file.exists():
            raise FileNotFoundError(f"Hint file not found: {hint_file}")
        
        with open(hint_file, 'r') as f:
            hint_data = json.load(f)
            hints = [HintRecord(**h) for h in hint_data]
        
        logger.info(f"Loaded {len(hints)} hints for '{args.hint}' family")
    
    # Limit questions if specified
    if args.n_questions:
        questions = questions[:args.n_questions]
        hints = hints[:args.n_questions]
    
    logger.info(f"Loaded {len(questions)} questions and {len(hints)} hints")
    
    # Set up output path
    output_path = get_completion_path(args.dataset, args.model, args.hint) / "completions.jsonl"
    
    # Import and run the standard completion generation
    from scripts.utils.completion_utils import generate_completions_async
    
    asyncio.run(generate_completions_async(
        questions=questions,
        hints=hints,
        client=client,
        output_path=output_path,
        hint_type=args.hint,
        resume=args.resume
    ))
    
    logger.info("Single-process completion generation finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())