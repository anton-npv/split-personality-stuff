"""Google AI client wrapper."""

import os
import logging
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class GoogleClient(LLMClient):
    """Client for Google Gemini models."""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        
        # Initialize Google client
        import google.generativeai as genai
        api_key = os.getenv("API_KEY_GOOGLE")
        if not api_key:
            raise ValueError("API_KEY_GOOGLE environment variable not set")
        
        genai.configure(api_key=api_key)
        
        # Create generation config (use max_output_tokens for Google)
        self.generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        **{k: v for k, v in kwargs.items() if k != "max_tokens"}  # Remove max_tokens if present
        }
        
        self.model = genai.GenerativeModel(
            model_name=self.model_id,
            generation_config=self.generation_config
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def complete(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate completion using Google Gemini API."""
        try:
            # Convert messages to prompt
            # Gemini expects a single prompt, so we'll combine the messages
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
                else:
                    prompt_parts.append(content)
            
            prompt = "\n\n".join(prompt_parts)
            
            # Make API call
            response = self.model.generate_content(prompt)
            
            # Extract content
            content = response.text if response.text else ""
            
            # Extract usage info (Gemini doesn't provide detailed usage)
            usage = {
                "input_tokens": 0,  # Not provided by Gemini API
                "output_tokens": 0,  # Not provided by Gemini API
            }
            
            logger.info(f"Google API call successful")
            
            return LLMResponse(
                content=content,
                model=self.model_id,
                usage=usage,
                raw_response={"text": content, "candidates": response.candidates}
            )
            
        except Exception as e:
            logger.error(f"Google API call failed: {e}")
            raise