#!/usr/bin/env python3
"""Test Featherless AI integration."""

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Import after loading env
from src.clients.featherless_client import FeatherlessClient

def test_featherless():
    """Test basic Featherless functionality."""
    
    # Check if API key is set
    api_key = os.getenv("API_KEY_FEATHERLESS")
    if not api_key:
        print("❌ API_KEY_FEATHERLESS not set in .env file")
        print("Please add your Featherless API key to the .env file")
        return
    
    print("✅ Featherless API key found")
    
    # Test with Llama 3.1 8B
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"\n🔧 Testing model: {model_id}")
    
    try:
        # Create client
        client = FeatherlessClient(
            model_id=model_id,
            temperature=0,
            max_tokens=100
        )
        print("✅ Client created successfully")
        
        # Test simple completion
        prompt = "What is the capital of France? Answer in one word."
        print(f"\n📝 Prompt: {prompt}")
        
        response = client.complete_simple(prompt)
        print(f"🤖 Response: {response}")
        
        # Test with messages format
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers concisely."},
            {"role": "user", "content": "List three primary colors."}
        ]
        print(f"\n📝 Testing messages format...")
        
        response = client.complete_sync(messages)
        print(f"🤖 Response: {response.content}")
        
        print("\n✅ Featherless integration test passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check your API key and internet connection")


if __name__ == "__main__":
    test_featherless()