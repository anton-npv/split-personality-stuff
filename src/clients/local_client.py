"""Local model client for running inference on local GPUs."""

import asyncio
import logging
import torch
from typing import List, Dict, Any, Optional
from ..clients.base import LLMClient, LLMResponse
from ..local.model_handler import load_model_and_tokenizer, generate_completion_batch

logger = logging.getLogger(__name__)


class LocalClient(LLMClient):
    """Client for local model inference."""
    
    def __init__(
        self,
        model_path: str,
        temperature: float = 0,
        max_tokens: int = 2048,
        batch_size: int = 8,
        quantization: Optional[str] = None,
        device_map: str = "auto",
        stop_strings: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(model_path, temperature, max_tokens, **kwargs)
        self.model_path = model_path
        self.batch_size = batch_size
        self.quantization = quantization
        self.device_map = device_map
        self.stop_strings = stop_strings or []
        
        # Lazy load model (loaded on first use)
        self._model = None
        self._tokenizer = None
        self._model_name = None
        self._device = None
        self._loaded = False
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded (lazy loading)."""
        if not self._loaded:
            logger.info(f"Loading model: {self.model_path}")
            self._model, self._tokenizer, self._model_name, self._device = load_model_and_tokenizer(
                self.model_path,
                quantization=self.quantization,
                device_map=self.device_map
            )
            self._loaded = True
            logger.info(f"Model {self._model_name} ready for inference")
    
    async def complete(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate completion for given messages."""
        self._ensure_model_loaded()
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._complete_sync,
            messages
        )
        return result
    
    def _complete_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Synchronous completion generation."""
        self._ensure_model_loaded()  # Ensure model is loaded
        
        prompt_dict = {
            "question_id": 0,
            "messages": messages
        }
        
        results = generate_completion_batch(
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
            prompts=[prompt_dict],
            batch_size=1,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            stop_strings=self.stop_strings
        )
        
        if not results:
            raise RuntimeError("No completion generated")
        
        completion = results[0]["completion"]
        
        return LLMResponse(
            content=completion,
            model=self._model_name or self.model_path,
            usage={"completion_tokens": len(completion.split())},  # Rough estimate
            raw_response={"messages": messages, "completion": completion}
        )
    
    def complete_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Synchronous wrapper for completion."""
        return self._complete_sync(messages)
    
    def complete_batch_sync(
        self,
        messages_batch: List[List[Dict[str, str]]],
        question_ids: Optional[List[int]] = None
    ) -> List[LLMResponse]:
        """Batch completion for better throughput."""
        self._ensure_model_loaded()
        
        if question_ids is None:
            question_ids = list(range(len(messages_batch)))
        
        # Convert to prompt format
        prompts = []
        for i, messages in enumerate(messages_batch):
            prompts.append({
                "question_id": question_ids[i],
                "messages": messages
            })
        
        # Generate completions
        results = generate_completion_batch(
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
            prompts=prompts,
            batch_size=self.batch_size,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            stop_strings=self.stop_strings
        )
        
        # Convert to LLMResponse format
        responses = []
        for result in results:
            completion = result["completion"]
            responses.append(LLMResponse(
                content=completion,
                model=self._model_name or self.model_path,
                usage={"completion_tokens": len(completion.split())},
                raw_response={"completion": completion}
            ))
        
        return responses
    
    @property
    def model_name(self) -> str:
        """Get model name for identification."""
        if self._model_name:
            return self._model_name
        return self.model_path.split("/")[-1]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        self._ensure_model_loaded()
        
        return {
            "model_path": self.model_path,
            "model_name": self._model_name,
            "device": str(self._device),
            "quantization": self.quantization,
            "parameters": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "batch_size": self.batch_size
            }
        }