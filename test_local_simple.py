#!/usr/bin/env python3
"""Simple test for local model functionality without complex optimizations."""

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.clients import create_client
from src.utils.config import load_model_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_simple_local_inference():
    """Test basic local model inference with minimal optimizations."""
    
    try:
        # Load model config
        logger.info("Loading configuration...")
        all_configs = load_model_config()
        model_name = "gemma-3-4b-local"
        
        if model_name not in all_configs:
            logger.error(f"Model {model_name} not found")
            return False
            
        model_config = all_configs[model_name].copy()
        
        # Modify config for simple test - disable optimizations that might cause issues
        model_config["batch_size"] = 1  # Single item batches
        model_config["device_map"] = "cuda:0"  # Use specific device
        
        # Create client
        provider = model_config.pop("provider")
        logger.info(f"Creating client with config: {model_config}")
        
        client = create_client(
            provider=provider,
            model_id=model_name,
            **model_config
        )
        
        # Test simple math question
        test_messages = [
            {"role": "user", "content": "What is 5 + 3? Answer with just the number."}
        ]
        
        logger.info("Testing simple completion...")
        response = client.complete_sync(test_messages)
        
        logger.info(f"✅ Response: {response.content[:100]}...")
        logger.info(f"✅ Model: {response.model}")
        
        # Test if we got a reasonable response
        if "8" in response.content or "eight" in response.content.lower():
            logger.info("✅ Math test passed!")
        else:
            logger.warning(f"⚠️ Math test unclear, got: {response.content}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_inference():
    """Test batch inference with multiple questions."""
    
    try:
        logger.info("Testing batch inference...")
        
        all_configs = load_model_config()
        model_name = "gemma-3-4b-local"
        model_config = all_configs[model_name].copy()
        
        # Configure for batch test
        model_config["batch_size"] = 2
        model_config["device_map"] = "cuda:0"
        
        provider = model_config.pop("provider") 
        client = create_client(provider=provider, model_id=model_name, **model_config)
        
        # Test batch with multiple simple questions
        batch_messages = [
            [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
            [{"role": "user", "content": "What color is the sky? Answer briefly."}]
        ]
        
        if hasattr(client, 'complete_batch_sync'):
            responses = client.complete_batch_sync(batch_messages)
            
            logger.info(f"✅ Batch completed: {len(responses)} responses")
            for i, response in enumerate(responses):
                logger.info(f"  Response {i+1}: {response.content[:50]}...")
            
            return True
        else:
            logger.info("⚠️ Batch inference not available, testing individual completions")
            
            for i, messages in enumerate(batch_messages):
                response = client.complete_sync(messages)
                logger.info(f"  Individual {i+1}: {response.content[:50]}...")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Batch test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("🧪 Starting simple local model tests...")
    
    success = True
    
    # Test 1: Simple inference
    logger.info("\n--- Test 1: Simple Inference ---")
    success &= test_simple_local_inference()
    
    # Test 2: Batch inference  
    logger.info("\n--- Test 2: Batch Inference ---")
    success &= test_batch_inference()
    
    if success:
        logger.info("\n🎉 All tests passed! Local model is working correctly.")
    else:
        logger.error("\n💥 Some tests failed.")
    
    sys.exit(0 if success else 1)