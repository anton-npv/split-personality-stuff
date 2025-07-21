"""Anthropic client wrapper."""

import os
import logging
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """Client for Anthropic Claude models."""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        
        # Initialize Anthropic client
        import anthropic
        api_key = os.getenv("API_KEY_ANTHROPIC")
        if not api_key:
            raise ValueError("API_KEY_ANTHROPIC environment variable not set")
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def complete(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate completion using Anthropic API."""
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            for msg in messages:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Make API call
            response = self.client.messages.create(
                model=self.model_id,
                messages=anthropic_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.kwargs
            )
            
            # Extract content
            content = response.content[0].text if response.content else ""
            
            # Extract usage info
            usage = {
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
            }
            
            logger.info(f"Anthropic API call successful. Usage: {usage}")
            
            return LLMResponse(
                content=content,
                model=self.model_id,
                usage=usage,
                raw_response=response.model_dump()
            )
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise