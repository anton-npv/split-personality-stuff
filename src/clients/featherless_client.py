"""Featherless AI client using OpenAI SDK compatibility."""
import os
import logging
from typing import List, Dict, Optional, Any
from openai import OpenAI
from .base import LLMClient, LLMResponse


class FeatherlessClient(LLMClient):
    """Client for Featherless AI models using OpenAI-compatible API."""
    
    def __init__(self, model_id: str, temperature: float = 0.0, max_tokens: int = 1000, **kwargs):
        """Initialize Featherless client.
        
        Args:
            model_id: Full model name (e.g., 'meta-llama/Meta-Llama-3.1-8B-Instruct')
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for the API
        """
        super().__init__(model_id, temperature, max_tokens)
        
        # Get API key from environment
        api_key = os.getenv("API_KEY_FEATHERLESS")
        if not api_key:
            raise ValueError("API_KEY_FEATHERLESS environment variable not set")
        
        # Initialize OpenAI client with Featherless endpoint
        self.client = OpenAI(
            base_url="https://api.featherless.ai/v1",
            api_key=api_key,
        )
        
        # Store additional parameters
        self.additional_params = kwargs
        self.logger = logging.getLogger(__name__)
    
    async def complete(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate completion for given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            LLMResponse with completion and metadata
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.additional_params
            )
            
            # Extract usage information
            usage_dict = {}
            if hasattr(response, 'usage') and response.usage:
                usage_dict = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                self.logger.info(f"Featherless API call successful. Usage: {usage_dict}")
            
            # Extract the completion text
            completion = response.choices[0].message.content
            
            return LLMResponse(
                content=completion,
                model=self.model_id,
                usage=usage_dict,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
            )
            
        except Exception as e:
            self.logger.error(f"Featherless API error: {str(e)}")
            raise
    
    def complete_simple(self, prompt: str) -> str:
        """Simple completion method for backward compatibility.
        
        Args:
            prompt: The input prompt
            
        Returns:
            The generated completion text
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.complete_sync(messages)
        return response.content