"""Base classes for LLM clients."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Standard response from LLM clients."""
    content: str
    model: str
    usage: Dict[str, int] = {}
    raw_response: Dict[str, Any] = {}


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model_id: str, temperature: float = 0, max_tokens: int = 4096, **kwargs):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
    
    @abstractmethod
    async def complete(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate completion for given messages."""
        pass
    
    def complete_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Synchronous wrapper for completion."""
        import asyncio
        return asyncio.run(self.complete(messages))