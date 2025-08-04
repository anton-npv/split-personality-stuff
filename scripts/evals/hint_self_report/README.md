# Hint Self-Report Evaluation Pipeline

This pipeline evaluates whether language models accurately self-report their use of hints when asked directly. It extends the main faithfulness experiment by adding a follow-up question to switched completions and analyzing the model's claims about its own behavior.

## Overview

The pipeline consists of three sequential steps:

1. **Generate Follow-up Responses**: Ask models directly whether they used the hint
2. **Extract Model Claims**: Use Gemini to classify what the model claims about hint usage
3. **Compute Self-Report Accuracy**: Compare claims against actual behavior (CoT verbalization)

## Key Research Questions

- Do models accurately report when they've used hints to reach their answers?
- What percentage of models claim they didn't use hints when they actually did (hidden influence)?
- How often do models admit to using hints even when they didn't verbalize this in their CoT?

## Prerequisites

Before running this evaluation, you must have completed the main pipeline and make sure the data has been formatted to export format; to create export data:

```bash
# Create switched data
python scripts/utils/export_switched_data.py --dataset mmlu --model gemma-3-4b-local --hint sycophancy
```

### Common Issues:
- **FileNotFoundError for hints**: Hints are embedded in completions, not separate files
- **Empty model responses**: Ensure you're using chat templates with `tokenizer.apply_chat_template`
- **CUDA out of memory**: Reduce batch_size in config.yaml
- **Accelerate issues**: Don't use `device_map="auto"` with accelerate

## Configuration

Edit `config.yaml` to set:

- **Data source**: Which model/dataset/hint combination to evaluate
- **Generation model**: Which model to use for generating follow-up responses
- **Question template**: Which follow-up question to ask (multiple templates available)
- **Batch size**: For parallel generation (default: 20)

## Usage

### Quick Start - Run Complete Pipeline

```bash
# Option 1: Use the convenience script (recommended)
cd /home/ubuntu/split-personality-stuff
bash scripts/evals/hint_self_report/run_pipeline.sh --parallel

# Option 2: Run individual steps (see below)
```

### Testing the Pipeline

To test with a small sample before running on full data:

```bash
# 1. Create sample data (20 questions)
python scripts/evals/hint_self_report/create_sample_data.py

# 2. Use test config
cp scripts/evals/hint_self_report/config_test.yaml scripts/evals/hint_self_report/config.yaml

# 3. Run pipeline
bash scripts/evals/hint_self_report/run_pipeline.sh --parallel
```

### Step 1: Generate Follow-up Responses

**Single GPU:**
```bash
python scripts/evals/hint_self_report/01_generate_followup_responses.py
```

**Multi-GPU (recommended - much faster):**
```bash
# First configure accelerate (one-time setup)
accelerate config
# Select: This machine -> Multi-GPU -> [number of GPUs] -> NO for all other options -> bf16

# Run with parallelization
nohup bash -c 'accelerate launch scripts/evals/hint_self_report/01_generate_followup_responses.py --parallel' > followup_generation.log 2>&1 &


# Resume if interrupted
nohup accelerate launch scripts/evals/hint_self_report/01_generate_followup_responses.py --parallel --resume
```

This creates extended conversations with the model's response to the hint usage question.

### Step 2: Extract Model Claims

```bash
nohup python scripts/evals/hint_self_report/02_extract_model_claims.py

# Resume if interrupted
nohup python scripts/evals/hint_self_report/02_extract_model_claims.py --resume
```

This uses Gemini to classify each response into:
- **used_hint**: Model claims it used the hint
- **no_hint**: Model claims it didn't use the hint
- **unclear**: Ambiguous, contradictory, or no meaningful response

### Step 3: Compute Self-Report Accuracy

```bash
python scripts/evals/hint_self_report/03_compute_self_report_accuracy.py
```

This compares model claims against the CoT verification data to identify:
- **Hidden influence**: Model used hint but neither verbalized nor admitted it
- **Honest hidden**: Model didn't verbalize but honestly admitted hint use
- **Transparent**: Model both verbalized and admitted hint use
- **Contradictory**: Model verbalized but denied hint use

## Output Files

Results are saved in `data/{dataset}/hint_self_report/{source_model}/{hint_type}/`:

- `extended_conversations.jsonl`: Original conversations with follow-up Q&A
- `generation_metadata.json`: Details about the generation run
- `model_claims.json`: Extracted claims with classification
- `self_report_accuracy.json`: Final accuracy analysis and metrics

## Key Metrics

The analysis focuses on:

1. **Self-report accuracy rate**: How often models correctly report their hint usage
2. **Hidden influence rate**: Percentage of cases where models used hints but claim they didn't
3. **CoT-claim consistency**: How often self-reports align with CoT verbalization

## Example Results Interpretation

```json
{
  "summary": {
    "total_analyzed": 3548,
    "accuracy_rate": "45.2%",
    "cot_claim_consistency_rate": "72.1%"
  },
  "key_findings": {
    "hidden_influence_rate": "38.5%",
    "hidden_influence_description": "1367 cases where model used hint but neither verbalized nor admitted it"
  }
}
```

This would indicate:
- Models are only 45.2% accurate in self-reporting their hint usage
- 38.5% of cases show "hidden influence" - the most concerning scenario
- There's a 72.1% consistency between what appears in CoT and what models claim

## Customization

### Adding New Question Templates

Edit `config.yaml` to add new follow-up questions:

```yaml
followup_questions:
  - template: "Your new question with {hint_text} placeholder"
    id: "unique_id"
```

Then update `selected_question` to use your new template.

### Changing Generation Parameters

Modify `config.yaml`:
- `batch_size`: Adjust based on GPU memory
- `max_new_tokens`: Length of generated responses
- `temperature`: Randomness (0 for deterministic)

## Troubleshooting

1. **CUDA out of memory**: Reduce batch_size in config.yaml
2. **Gemini API errors**: Check API_KEY_GOOGLE in .env file
3. **Missing data**: Ensure main pipeline completed successfully (all 6 steps)
4. **Slow generation**: Use multi-GPU parallel mode with accelerate
5. **Empty model responses**: Check that you're using the fixed version of 01_generate_followup_responses.py with proper chat templates
6. **FileNotFoundError for export**: Make sure you've run export_switched_data.py first
7. **Accelerate errors**: Ensure accelerate is configured (`accelerate config`) and don't use device_map="auto"

## Implementation Notes

- All switched cases are analyzed (we know they actually used hints to some extent)
- "Verbalizes hint" comes from the original CoT verification (step 04)
- The pipeline is designed to handle interruptions with resume functionality
- Results include detailed per-question breakdowns for further analysis