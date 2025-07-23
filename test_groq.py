#!/usr/bin/env python3
"""Test Groq integration."""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.clients.groq_client import GroqClient

def test_groq():
    """Test Groq client functionality."""
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found in environment")
        print("Please add your Groq API key to .env file")
        return
    
    print("✅ Groq API key found\n")
    
    try:
        # Create client
        print("🔧 Testing model: llama-3.1-8b-instant")
        client = GroqClient(
            model_id="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=100
        )
        print("✅ Client created successfully")
        
        # Test simple completion
        prompt = "What is the capital of France? Answer in one word."
        print(f"\n📝 Prompt: {prompt}")
        
        messages = [{"role": "user", "content": prompt}]
        response = client.complete_sync(messages)
        print(f"🤖 Response: {response.content}")
        
        # Test with system message
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers concisely."},
            {"role": "user", "content": "List three primary colors."}
        ]
        print(f"\n📝 Testing messages format...")
        
        response = client.complete_sync(messages)
        print(f"🤖 Response: {response.content}")
        
        print("\n✅ Groq integration test passed!")
        print("🚀 Groq provides much faster inference than Featherless!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_groq()