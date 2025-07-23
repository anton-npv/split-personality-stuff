"""Azure AI client wrapper for LLM inference."""

import os
import logging
from typing import List, Dict, Any, Optional

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential

from .base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class AzureClient(LLMClient):
    """Client for Azure AI model inference."""
    
    def __init__(self, model_id: str, temperature: float = 0.0, max_tokens: int = 4096, **kwargs):
        """Initialize Azure client.
        
        Args:
            model_id: Model name to use (e.g., "Meta-Llama-3.1-8B-Instruct")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        """
        super().__init__(model_id, temperature, max_tokens, **kwargs)
        
        # Get Azure credentials from environment
        endpoint = os.getenv("AZURE_ENDPOINT")
        api_key = os.getenv("AZURE_API_KEY")
        
        if not endpoint:
            raise ValueError("AZURE_ENDPOINT environment variable not set")
        if not api_key:
            raise ValueError("AZURE_API_KEY environment variable not set")
            
        self.endpoint = endpoint
        self.model_name = model_id
        self.client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
            api_version="2024-05-01-preview"
        )
        
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List:
        """Convert messages to Azure format."""
        azure_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                azure_messages.append(SystemMessage(content=content))
            elif role == "user":
                azure_messages.append(UserMessage(content=content))
            elif role == "assistant":
                azure_messages.append(AssistantMessage(content=content))
            else:
                raise ValueError(f"Unsupported message role: {role}")
                
        return azure_messages
        
    async def complete(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate completion using Azure AI.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            LLMResponse with completion and metadata
        """
        # Convert messages to Azure format
        azure_messages = self._convert_messages(messages)
        
        try:
            # Azure SDK is synchronous, so we make a sync call
            response = self.client.complete(
                messages=azure_messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.kwargs.get("top_p", 1.0),
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
            
            # Extract usage info if available
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }
                logger.info(f"Azure API call successful. Usage: {usage}")
            else:
                logger.info("Azure API call successful")
                
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model_name,
                usage=usage,
                raw_response={"choices": response.choices} if hasattr(response, "choices") else {}
            )
            
        except Exception as e:
            logger.error(f"Azure API error: {str(e)}")
            raise