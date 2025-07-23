"""Groq client for fast inference."""
import os
import logging
from typing import List, Dict, Optional, Any
from groq import Groq
from .base import LLMClient, LLMResponse


class GroqClient(LLMClient):
    """Client for Groq API models."""
    
    def __init__(self, model_id: str, temperature: float = 0.0, max_tokens: int = 1000, **kwargs):
        """Initialize Groq client.
        
        Args:
            model_id: Model identifier (e.g., 'llama-3.1-8b-instant')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        """
        super().__init__(model_id, temperature, max_tokens, **kwargs)
        
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.client = Groq(api_key=api_key)
        self.logger = logging.getLogger(__name__)
    
    async def complete(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate completion for given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            LLMResponse with completion and metadata
        """
        try:
            # Make synchronous call (Groq SDK is sync)
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                stream=False,  # We don't use streaming for this implementation
                **self.kwargs
            )
            
            # Extract usage information
            usage_dict = {}
            if hasattr(response, 'usage') and response.usage:
                usage_dict = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                self.logger.info(f"Groq API call successful. Usage: {usage_dict}")
            
            # Extract the completion text
            completion = response.choices[0].message.content
            
            return LLMResponse(
                content=completion,
                model=self.model_id,
                usage=usage_dict,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
            )
            
        except Exception as e:
            self.logger.error(f"Groq API error: {str(e)}")
            raise