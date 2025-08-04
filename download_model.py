#!/usr/bin/env python3
"""Download the Gemma model from HuggingFace."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "google/gemma-3-4b-it"
print(f"Downloading model: {model_name}")

# Download tokenizer
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("Tokenizer downloaded successfully")

# Download model with 4-bit quantization to save memory
print("Downloading model (this may take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)
print("Model downloaded successfully")

print(f"Model loaded on device: {model.device}")
print("Ready for inference!")