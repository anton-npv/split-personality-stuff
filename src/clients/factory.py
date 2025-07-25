"""Factory for creating LLM clients."""

from typing import Dict, Any
from .base import LLMClient
from .anthropic_client import AnthropicClient
from .openai_client import OpenAIClient
from .google_client import GoogleClient
from .featherless_client import FeatherlessClient
from .groq_client import GroqClient
from .azure_client import AzureClient
from .local_client import LocalClient


def create_client(provider: str, model_id: str, **config) -> LLMClient:
    """Create an LLM client based on provider."""
    
    provider_map = {
        "anthropic": AnthropicClient,
        "openai": OpenAIClient,
        "google": GoogleClient,
        "featherless": FeatherlessClient,
        "groq": GroqClient,
        "azure": AzureClient,
        "local": LocalClient,
    }
    
    if provider not in provider_map:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(provider_map.keys())}")
    
    client_class = provider_map[provider]
    
    # Remove provider-specific config keys that shouldn't be passed to client
    client_config = {k: v for k, v in config.items() 
                     if k not in ["provider", "description"]}
    
    # Handle different parameter names for different providers
    if provider == "local":
        # Local client uses model_path instead of model_id
        client_config["model_path"] = client_config.pop("model_path", model_id)
        return client_class(**client_config)
    else:
        return client_class(model_id=model_id, **client_config)