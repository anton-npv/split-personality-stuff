# Hint Self-Report Evaluation Pipeline - Summary

## What We've Created

A complete evaluation pipeline to test whether language models accurately self-report their use of hints. The pipeline consists of:

### 1. Configuration System
- **`config.yaml`**: Production configuration for full dataset runs
- **`config_test.yaml`**: Test configuration with smaller batches and sample data
- Support for multiple question templates
- Configurable models, batch sizes, and output paths

### 2. Three Sequential Scripts

#### `01_generate_followup_responses.py`
- Takes switched conversations and adds a follow-up question
- Uses local models with multi-GPU support via Accelerate
- **Critical**: Uses proper chat templates with `tokenizer.apply_chat_template`
- **Critical**: No `device_map="auto"` when using accelerate
- Supports resume functionality for interrupted runs
- Default batch size: 20 (configurable)

#### `02_extract_model_claims.py`
- Uses Gemini to analyze model responses
- Classifies claims into: used_hint, no_hint, unclear (merged unclear/no_response)
- Uses `GenerationConfig` (not GenerateContentConfig)
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
- **`README.md`**: Comprehensive documentation with prerequisites
- **`run_pipeline.sh`**: Convenience script to run full pipeline
- **`create_sample_data.py`**: Creates 20-question test dataset
- **`SUMMARY.md`**: This file

## Key Features

- **Multi-GPU Support**: Leverages Accelerate for parallel generation
- **Resume Capability**: All scripts can resume from interruptions
- **Flexible Configuration**: Easy to change models, questions, and parameters
- **Detailed Analysis**: Provides both aggregate metrics and per-question details

## Usage

### Prerequisites First!
```bash
# MUST complete main pipeline first (see README.md for full commands)
# Key step: Export switched data
python scripts/utils/export_switched_data.py --dataset mmlu --model gemma-3-4b-local --hint sycophancy
```

### Quick Start
```bash
# Edit config.yaml to set your parameters
vim scripts/evals/hint_self_report/config.yaml

# Run the complete pipeline
cd /home/ubuntu/split-personality-stuff
bash scripts/evals/hint_self_report/run_pipeline.sh --parallel
```

### Testing First (Recommended)
```bash
# Create test data and use test config
python scripts/evals/hint_self_report/create_sample_data.py
cp scripts/evals/hint_self_report/config_test.yaml scripts/evals/hint_self_report/config.yaml
bash scripts/evals/hint_self_report/run_pipeline.sh --parallel
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

## Common Pitfalls to Avoid

1. **Running without prerequisites**: Must complete main pipeline first (6 steps)
2. **Using wrong model loading**: Don't use `device_map="auto"` with accelerate
3. **Missing chat templates**: Always use `tokenizer.apply_chat_template` for Gemma
4. **Wrong Gemini config**: Use `GenerationConfig`, not `GenerateContentConfig`
5. **Not testing first**: Always test with sample data before full runs

## Test Results

In our testing with Gemma-3-4b-it on 20 sample questions:
- **100% self-report accuracy**: Model honestly admitted using hints in all cases
- **65% honest hidden rate**: Model admitted hint use even when not verbalizing it in CoT
- **35% transparent rate**: Model both verbalized and admitted hint use
- **0% hidden influence**: No cases of dishonesty about hint usage