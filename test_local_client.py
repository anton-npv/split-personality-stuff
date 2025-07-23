#!/usr/bin/env python3
"""Test script for local model client functionality."""

import asyncio
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


async def test_local_client():
    """Test local client with a simple prompt."""
    
    # Test with a small model first
    model_name = "gemma-3-4b-local"
    
    try:
        # Load model config
        logger.info(f"Loading configuration for {model_name}")
        all_configs = load_model_config()
        
        if model_name not in all_configs:
            logger.error(f"Model {model_name} not found in configs")
            return False
            
        model_config = all_configs[model_name]
        logger.info(f"Model config: {model_config}")
        
        # Create client
        logger.info("Creating local client...")
        
        # Extract provider and remove it from config to avoid conflicts
        provider = model_config.pop("provider")
        client = create_client(
            provider=provider,
            model_id=model_name,
            **model_config
        )
        
        # Test simple completion
        test_messages = [
            {"role": "user", "content": "What is 2 + 2? Answer briefly."}
        ]
        
        logger.info("Testing completion...")
        response = await client.complete(test_messages)
        
        logger.info(f"Response: {response.content}")
        logger.info(f"Model: {response.model}")
        logger.info(f"Usage: {response.usage}")
        
        # Test model info
        if hasattr(client, 'get_model_info'):
            info = client.get_model_info()
            logger.info(f"Model info: {info}")
        
        logger.info("✅ Local client test passed!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        logger.error("Install with: pip install -r requirements-local.txt")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    try:
        logger.info("Testing configuration loading...")
        all_configs = load_model_config()
        
        if "gemma-3-4b-local" not in all_configs:
            logger.error("gemma-3-4b-local not found in model configs")
            return False
            
        config = all_configs["gemma-3-4b-local"]
        logger.info(f"Config loaded: {config}")
        
        assert config["provider"] == "local"
        assert "model_path" in config
        
        logger.info("✅ Configuration test passed!")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("Starting local client tests...")
    
    # Test configuration first
    config_ok = test_config_loading()
    if not config_ok:
        return 1
    
    # Test local client (only if dependencies are available)
    client_ok = await test_local_client()
    
    if config_ok and client_ok:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("💥 Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())