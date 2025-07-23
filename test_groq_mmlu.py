#!/usr/bin/env python3
"""Quick test of Groq with MMLU questions."""
import os
import sys
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.clients.groq_client import GroqClient
from src.prompts import build_prompt

def test_groq_mmlu():
    """Test Groq with a few MMLU questions."""
    
    # API key should be set in .env file
    # os.environ["GROQ_API_KEY"] = "your_key_here"
    
    # Load a few MMLU questions
    with open("data/mmlu/processed/test.json", 'r') as f:
        questions = json.load(f)[:5]  # First 5 questions
    
    # Create client
    print("🚀 Testing Groq Llama 3.1 8B on MMLU questions\n")
    client = GroqClient(
        model_id="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1000
    )
    
    # Test each question
    for i, q in enumerate(questions):
        print(f"📝 Question {i+1}:")
        print(f"   {q['question']}")
        print(f"   Options: {q['choices']}")
        
        # Build prompt manually for test
        prompt = f"{q['question']}\n\n"
        for letter, choice in sorted(q['choices'].items()):
            prompt += f"{letter}. {choice}\n"
        prompt += "\nThink step-by-step to solve the problem. After your reasoning, write **only** the final option letter (A-D)."
        
        messages = [{"role": "user", "content": prompt}]
        
        # Time the request
        start = time.time()
        response = client.complete_sync(messages)
        elapsed = time.time() - start
        
        print(f"🤖 Response (in {elapsed:.2f}s):")
        print(f"   {response.content[-100:]}")  # Last 100 chars
        print(f"   Correct answer: {q['correct']}\n")
    
    print("✅ Groq is significantly faster than Featherless!")

if __name__ == "__main__":
    test_groq_mmlu()