#!/usr/bin/env python3
"""Test Azure AI integration."""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_azure_client():
    """Test Azure client with a simple prompt."""
    from src.clients.azure_client import AzureClient
    
    # Initialize client
    client = AzureClient(
        model_id="Meta-Llama-3.1-8B-Instruct",  # Change this to match your Azure model
        temperature=0,
        max_tokens=100
    )
    
    # Test prompt
    messages = [
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ]
    
    print("Testing Azure AI client...")
    print(f"Endpoint: {client.endpoint}")
    print(f"Model: {client.model_name}")
    print("\nNote: Make sure your model name matches exactly what's shown in Azure deployment")
    
    try:
        response = client.complete_sync(messages)
        print(f"\nResponse: {response.content}")
        print(f"Model: {response.model}")
        if response.usage:
            print(f"Usage: {response.usage}")
        print("\n✅ Azure client test successful!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure you have set these environment variables:")
        print("- AZURE_ENDPOINT: Your Azure endpoint URL")
        print("- AZURE_API_KEY: Your Azure API key")
        print("\nAlso ensure you have installed the Azure SDK:")
        print("pip install azure-ai-inference")


def test_through_factory():
    """Test Azure client through the factory."""
    from src.clients.factory import create_client
    
    print("\nTesting Azure through factory...")
    
    try:
        # Load model config
        import yaml
        with open("configs/models.yaml") as f:
            models_config = yaml.safe_load(f)
        
        # Get Azure model config
        model_name = "llama-3.1-8b-azure"
        if model_name not in models_config:
            print(f"❌ Model {model_name} not found in configs/models.yaml")
            return
            
        config = models_config[model_name]
        
        # Create client through factory
        client = create_client(**config)
        
        # Test prompt
        messages = [
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ]
        
        response = client.complete_sync(messages)
        print(f"Response: {response.content}")
        print("✅ Factory test successful!")
        
    except Exception as e:
        print(f"❌ Factory test failed: {str(e)}")


if __name__ == "__main__":
    print("Azure AI Integration Test")
    print("=" * 50)
    
    # Check if credentials are set
    if not os.getenv("AZURE_ENDPOINT"):
        print("❌ AZURE_ENDPOINT not set in environment")
        print("\nTo use Azure AI, add these to your .env file:")
        print("AZURE_ENDPOINT=https://your-deployment.services.ai.azure.com/models")
        print("AZURE_API_KEY=your-api-key-here")
        print("\nNOTE: The endpoint should be the full URL from your Azure deployment page")
        exit(1)
        
    if not os.getenv("AZURE_API_KEY"):
        print("❌ AZURE_API_KEY not set in environment")
        exit(1)
    
    # Run tests
    test_azure_client()
    test_through_factory()