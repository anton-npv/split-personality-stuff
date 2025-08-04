# Hint Self-Report Evaluation Pipeline - Summary

## What We've Created

A complete evaluation pipeline to test whether language models accurately self-report their use of hints. The pipeline consists of:

### 1. Configuration System (`config.yaml`)
- Centralized configuration for all pipeline parameters
- Support for multiple question templates
- Configurable models, batch sizes, and output paths

### 2. Three Sequential Scripts

#### `01_generate_followup_responses.py`
- Takes switched conversations and adds a follow-up question
- Uses local models with multi-GPU support via Accelerate
- Supports resume functionality for interrupted runs
- Default batch size: 20 (configurable)

#### `02_extract_model_claims.py`
- Uses Gemini to analyze model responses
- Classifies claims into: used_hint, no_hint, unclear, no_response
- Handles concurrent API calls with rate limiting
- Provides detailed summary statistics

#### `03_compute_self_report_accuracy.py`
- Compares model claims against CoT verbalization data
- Identifies key scenarios:
  - Hidden influence (doesn't verbalize, claims no use)
  - Honest hidden (doesn't verbalize, admits use)
  - Transparent (verbalizes and admits)
  - Contradictory (verbalizes but denies)
- Calculates accuracy metrics and consistency rates

### 3. Supporting Files
- `README.md`: Comprehensive documentation
- `run_pipeline.sh`: Convenience script to run full pipeline
- `SUMMARY.md`: This file

## Key Features

- **Multi-GPU Support**: Leverages Accelerate for parallel generation
- **Resume Capability**: All scripts can resume from interruptions
- **Flexible Configuration**: Easy to change models, questions, and parameters
- **Detailed Analysis**: Provides both aggregate metrics and per-question details

## Usage

### Quick Start
```bash
# Edit config.yaml to set your parameters
vim scripts/evals/hint_self_report/config.yaml

# Run the complete pipeline
./scripts/evals/hint_self_report/run_pipeline.sh --parallel
```

### Individual Steps
```bash
# Step 1: Generate follow-ups (with multi-GPU)
accelerate launch scripts/evals/hint_self_report/01_generate_followup_responses.py --parallel

# Step 2: Extract claims
python scripts/evals/hint_self_report/02_extract_model_claims.py

# Step 3: Compute accuracy
python scripts/evals/hint_self_report/03_compute_self_report_accuracy.py
```

## Output Structure
```
data/{dataset}/hint_self_report/{model}/{hint}/
├── extended_conversations.jsonl    # Conversations with follow-up Q&A
├── generation_metadata.json        # Generation run details
├── model_claims.json              # Extracted claims
└── self_report_accuracy.json      # Final analysis results
```

## Research Value

This pipeline helps answer:
- Do models know when they're using hints?
- How often do models hide their reliance on hints?
- Is there a gap between CoT transparency and self-reporting?

The "hidden influence" metric is particularly important for AI safety, as it identifies cases where models use hints but neither verbalize nor admit this influence.