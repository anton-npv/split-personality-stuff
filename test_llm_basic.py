#!/usr/bin/env python3
"""Test basic LLM client functionality with simple prompts."""

import asyncio
import os
from dotenv import load_dotenv
from src.clients import create_client
from src.utils.config import load_model_config
from src.utils.logging_setup import setup_logging

# Load environment variables from .env file
load_dotenv()

async def test_client(model_name: str, test_prompt: str):
    """Test a single model client."""
    print(f"\n{'='*60}")
    print(f"TESTING: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load model config
        model_config = load_model_config()
        if model_name not in model_config:
            print(f"❌ Model '{model_name}' not found in config")
            return False
        
        model_cfg = model_config[model_name]
        
        # Check if API key is available
        provider = model_cfg["provider"]
        key_var = f"API_KEY_{provider.upper()}"
        if not os.getenv(key_var):
            print(f"⚠️  API key {key_var} not set, skipping {model_name}")
            return False
        
        # Create client
        client = create_client(
            provider=model_cfg["provider"],
            model_id=model_cfg["model_id"],
            temperature=0,
            max_tokens=100
        )
        
        # Test message
        messages = [{"role": "user", "content": test_prompt}]
        
        print(f"Prompt: {test_prompt}")
        print("Generating response...")
        
        # Generate response
        response = await client.complete(messages)
        
        print(f"✅ SUCCESS!")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print(f"Response: {response.content[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

async def main():
    setup_logging()
    
    print("🧪 Testing LLM Client Infrastructure")
    print("This will test basic connectivity with simple prompts")
    
    # Simple test prompt
    test_prompt = "What is 2+2? Answer briefly."
    
    # Test available models
    test_models = [
        "claude-3-5-sonnet",
        "claude-3-5-haiku", 
        "gpt-4o",
        "gpt-4o-mini",
        "gemini-pro",
        "gemini-flash"
    ]
    
    results = {}
    
    for model in test_models:
        results[model] = await test_client(model, test_prompt)
        await asyncio.sleep(1)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    working_models = []
    failed_models = []
    
    for model, success in results.items():
        if success:
            print(f"✅ {model}")
            working_models.append(model)
        else:
            print(f"❌ {model}")
            failed_models.append(model)
    
    print(f"\nWorking models: {len(working_models)}")
    print(f"Failed models: {len(failed_models)}")
    
    if working_models:
        print(f"\n🎉 LLM infrastructure is working!")
        print(f"You can proceed with: {working_models[0]}")
    else:
        print(f"\n⚠️  No models working. Check your API keys in .env file")

if __name__ == "__main__":
    asyncio.run(main())