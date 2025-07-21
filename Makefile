# Makefile for reasoning faithfulness replica pipeline

# Default values
DATASET ?= mmlu
MODEL ?= claude-3-5-sonnet
HINT ?= sycophancy
SPLIT ?= dev

# Full pipeline
run:
	@echo "Running full pipeline: $(DATASET) + $(MODEL) + $(HINT)"
	python scripts/00_download_format.py --dataset $(DATASET) --split $(SPLIT) --hint-families $(HINT)
	python scripts/01_generate_completion.py --dataset $(DATASET) --model $(MODEL) --hint none --split $(SPLIT)
	python scripts/01_generate_completion.py --dataset $(DATASET) --model $(MODEL) --hint $(HINT) --split $(SPLIT)
	@echo "Pipeline complete! Check data/$(DATASET)/completions/$(MODEL)/"

# Individual steps
download:
	python scripts/00_download_format.py --dataset $(DATASET) --split $(SPLIT) --hint-families $(HINT)

baseline:
	python scripts/01_generate_completion.py --dataset $(DATASET) --model $(MODEL) --hint none --split $(SPLIT)

hinted:
	python scripts/01_generate_completion.py --dataset $(DATASET) --model $(MODEL) --hint $(HINT) --split $(SPLIT)

# Testing with small subset
test:
	python scripts/00_download_format.py --dataset $(DATASET) --split $(SPLIT) --hint-families $(HINT)
	python scripts/01_generate_completion.py --dataset $(DATASET) --model $(MODEL) --hint none --split $(SPLIT) --max-questions 3
	python scripts/01_generate_completion.py --dataset $(DATASET) --model $(MODEL) --hint $(HINT) --split $(SPLIT) --max-questions 3

# Clean up generated data
clean:
	rm -rf data/

# Resume interrupted completion generation
resume-baseline:
	python scripts/01_generate_completion.py --dataset $(DATASET) --model $(MODEL) --hint none --split $(SPLIT) --resume

resume-hinted:
	python scripts/01_generate_completion.py --dataset $(DATASET) --model $(MODEL) --hint $(HINT) --split $(SPLIT) --resume

# Help
help:
	@echo "Available targets:"
	@echo "  run          - Full pipeline (download + baseline + hinted completions)"
	@echo "  download     - Download and format dataset with hints"  
	@echo "  baseline     - Generate baseline completions (no hints)"
	@echo "  hinted       - Generate hinted completions"
	@echo "  test         - Run pipeline with 3 questions for testing"
	@echo "  clean        - Remove all generated data"
	@echo "  resume-*     - Resume interrupted completion generation"
	@echo ""
	@echo "Variables:"
	@echo "  DATASET=$(DATASET)  - Dataset to use"
	@echo "  MODEL=$(MODEL)    - Model to use" 
	@echo "  HINT=$(HINT)       - Hint family to use"
	@echo "  SPLIT=$(SPLIT)         - Dataset split to use"
	@echo ""
	@echo "Examples:"
	@echo "  make test"
	@echo "  make run DATASET=mmlu MODEL=gpt-4o HINT=metadata"
	@echo "  make baseline MODEL=claude-3-5-sonnet"

.PHONY: run download baseline hinted test clean resume-baseline resume-hinted help