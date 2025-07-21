"""OpenAI client wrapper."""

import os
import logging
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """Client for OpenAI models."""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        
        # Initialize OpenAI client
        import openai
        api_key = os.getenv("API_KEY_OPENAI")
        if not api_key:
            raise ValueError("API_KEY_OPENAI environment variable not set")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def complete(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate completion using OpenAI API."""
        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.kwargs
            )
            
            # Extract content
            content = response.choices[0].message.content or ""
            
            # Extract usage info
            usage = {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            }
            
            logger.info(f"OpenAI API call successful. Usage: {usage}")
            
            return LLMResponse(
                content=content,
                model=self.model_id,
                usage=usage,
                raw_response=response.model_dump()
            )
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise