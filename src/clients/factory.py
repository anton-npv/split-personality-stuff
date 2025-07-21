"""Factory for creating LLM clients."""

from typing import Dict, Any
from .base import LLMClient
from .anthropic_client import AnthropicClient
from .openai_client import OpenAIClient
from .google_client import GoogleClient


def create_client(provider: str, model_id: str, **config) -> LLMClient:
    """Create an LLM client based on provider."""
    
    provider_map = {
        "anthropic": AnthropicClient,
        "openai": OpenAIClient,
        "google": GoogleClient,
    }
    
    if provider not in provider_map:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(provider_map.keys())}")
    
    client_class = provider_map[provider]
    
    # Remove provider-specific config keys that shouldn't be passed to client
    client_config = {k: v for k, v in config.items() 
                     if k not in ["provider", "description"]}
    
    return client_class(model_id=model_id, **client_config)