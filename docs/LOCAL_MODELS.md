# Local Model Inference Guide

This guide covers setting up and using local GPU models for faithfulness experiments, providing significant speedup over API-based inference.

## Prerequisites

- **GPU**: NVIDIA GPU with at least 8GB VRAM (16GB+ recommended for larger models)
- **CUDA**: Compatible CUDA installation
- **Python**: 3.9+ with pip

## Setup

### 1. Install Dependencies

```bash
# Install base package (if not already done)
pip install -e .

# Install local model dependencies
pip install -r requirements-local.txt
```

This installs:
- `torch` - PyTorch with CUDA support
- `transformers` - HuggingFace model library
- `accelerate` - Multi-GPU coordination
- `bitsandbytes` - Quantization support

### 2. Configure Accelerate

```bash
accelerate config
```

Choose:
- **Compute environment**: This machine
- **Distributed training**: Multi-GPU
- **Number of machines**: 1
- **Number of GPUs**: (your GPU count, e.g., 2)
- **Mixed precision**: bf16
- **Use DeepSpeed**: No (unless you need it)

### 3. HuggingFace Authentication (Optional)

For gated models like Gemma:

```bash
pip install huggingface_hub
huggingface-cli login
```

Enter your HuggingFace token when prompted.

## Available Models

### Configured Models

See `configs/models.yaml` for the full list:

```yaml
gemma-3-4b-local:
  provider: local
  model_path: "google/gemma-3-4b-it"
  batch_size: 16  # per-GPU
  quantization: null

gemma-3-8b-local:
  provider: local
  model_path: "google/gemma-3-8b-it"
  batch_size: 8
  quantization: "4bit"  # Saves memory

llama-3.1-8b-local:
  provider: local
  model_path: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  batch_size: 8
  quantization: "4bit"
```

### Memory Requirements

| Model | Size | No Quantization | 4-bit Quantization |
|-------|------|----------------|-------------------|
| Gemma-3-4B | 4B params | ~8GB VRAM | ~3GB VRAM |
| Gemma-3-8B | 8B params | ~16GB VRAM | ~6GB VRAM |
| Llama-3.1-8B | 8B params | ~16GB VRAM | ~6GB VRAM |

## Usage

### Single Process Mode

Works on single GPU or as fallback:

```bash
python scripts/01_generate_completion_parallel.py \
  --dataset mmlu \
  --model gemma-3-4b-local \
  --hint sycophancy \
  --n-questions 100
```

### Multi-GPU Parallel Mode

**Recommended for speed:**

```bash
accelerate launch scripts/01_generate_completion_parallel.py \
  --dataset mmlu \
  --model gemma-3-4b-local \
  --hint sycophancy \
  --parallel \
  --batch-size 32 \
  --resume
```

### Parameters

- `--parallel`: Enable multi-GPU mode (requires `accelerate launch`)
- `--batch-size`: Override per-GPU batch size (default from config)
- `--resume`: Resume from existing completions file
- `--n-questions`: Limit questions for testing

## Performance

### Expected Throughput

With 2x RTX 4090 GPUs:

| Model | Single GPU | Multi-GPU | API Comparison |
|-------|------------|-----------|----------------|
| Gemma-3-4B | ~15 q/min | ~25 q/min | ~50x faster than API |
| Gemma-3-8B | ~8 q/min | ~14 q/min | ~30x faster than API |

### Optimization Tips

1. **Use appropriate batch sizes**: Start with config defaults, increase if memory allows
2. **Enable quantization**: 4-bit quantization saves memory with minimal quality loss
3. **Multi-GPU**: Always use `accelerate launch` for multiple GPUs
4. **Monitor VRAM**: Use `nvidia-smi` to check memory usage

## Integration with Pipeline

Local models integrate seamlessly with the existing pipeline:

```bash
# Step 1: Generate completions (local model)
accelerate launch scripts/01_generate_completion_parallel.py \
  --dataset mmlu --model gemma-3-4b-local --hint none --parallel

accelerate launch scripts/01_generate_completion_parallel.py \
  --dataset mmlu --model gemma-3-4b-local --hint sycophancy --parallel

# Steps 2-5: Use standard pipeline
python scripts/02_extract_answer.py --dataset mmlu --model gemma-3-4b-local --hint none
python scripts/02_extract_answer.py --dataset mmlu --model gemma-3-4b-local --hint sycophancy
python scripts/03_detect_switch.py --dataset mmlu --model gemma-3-4b-local --hint sycophancy --baseline none
python scripts/04_verify_cot.py --dataset mmlu --model gemma-3-4b-local --hint sycophancy
python scripts/05_compute_faithfulness.py --dataset mmlu --model gemma-3-4b-local --hint sycophancy
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
- Reduce `batch_size` in model config or via `--batch-size`
- Enable quantization: set `quantization: "4bit"` in config
- Try a smaller model

**Model Access Denied**
```
401 Client Error: Access to model ... is restricted
```
- Run `huggingface-cli login` with a valid token
- Ensure you have access to gated models (Gemma, Llama)

**Accelerate Not Found**
```
ModuleNotFoundError: No module named 'accelerate'
```
- Install: `pip install -r requirements-local.txt`
- Run: `accelerate config`

**Slow Performance**
- Verify GPU utilization: `nvidia-smi`
- Use multi-GPU mode: `accelerate launch --parallel`
- Increase batch size if memory allows

### Getting Help

1. Check GPU memory: `nvidia-smi`
2. Verify accelerate config: `accelerate env`
3. Test basic loading: `python test_local_client.py`

## Adding New Models

To add a new local model:

1. **Add to config** (`configs/models.yaml`):
```yaml
my-new-model-local:
  provider: local
  model_path: "organization/model-name"
  batch_size: 8
  quantization: "4bit"
  max_tokens: 2048
  temperature: 0
```

2. **Test the model**:
```bash
python test_local_client.py  # Update model name in script
```

3. **Run experiment**:
```bash
accelerate launch scripts/01_generate_completion_parallel.py \
  --dataset mmlu --model my-new-model-local --hint sycophancy --parallel
```

The system supports any HuggingFace Transformers model that works with `AutoModelForCausalLM`.