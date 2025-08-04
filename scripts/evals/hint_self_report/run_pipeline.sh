#!/bin/bash
# Run the complete hint self-report evaluation pipeline

set -e  # Exit on error

echo "=== Hint Self-Report Evaluation Pipeline ==="
echo

# Check if using parallel mode
PARALLEL_FLAG=""
if [ "$1" == "--parallel" ]; then
    PARALLEL_FLAG="--parallel"
    echo "Using multi-GPU parallel mode"
else
    echo "Using single GPU mode (use --parallel for multi-GPU)"
fi

# Step 1: Generate follow-up responses
echo "Step 1: Generating follow-up responses..."
if [ -n "$PARALLEL_FLAG" ]; then
    accelerate launch scripts/evals/hint_self_report/01_generate_followup_responses.py $PARALLEL_FLAG
else
    python scripts/evals/hint_self_report/01_generate_followup_responses.py
fi
echo "✓ Follow-up responses generated"
echo

# Step 2: Extract model claims
echo "Step 2: Extracting model claims..."
python scripts/evals/hint_self_report/02_extract_model_claims.py
echo "✓ Model claims extracted"
echo

# Step 3: Compute self-report accuracy
echo "Step 3: Computing self-report accuracy..."
python scripts/evals/hint_self_report/03_compute_self_report_accuracy.py
echo "✓ Self-report accuracy computed"
echo

echo "=== Pipeline Complete ==="
echo "Results saved in data/{dataset}/hint_self_report/{model}/{hint}/"
echo "Check self_report_accuracy.json for final metrics"