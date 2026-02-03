"""Thought Signature management for Gemini 3 API.

Implements the capture-and-return loop for thought_signature
to maintain object permanence and reasoning context across
API calls. This is mandatory for dynamic environment processing.

Flow:
1. Extract thought_signature from response candidates[].content.parts
2. Store signature in state's thought_signatures list
3. Inject signature back into conversation history for next call
"""

import time
import uuid
import logging
from typing import Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThoughtSignatureData:
    """Container for thought signature data."""
    
    signature_id: str
    timestamp: float
    content: dict
    model_version: str
    thinking_level: str
    frame_context: dict | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "signature_id": self.signature_id,
            "timestamp": self.timestamp,
            "content": self.content,
            "model_version": self.model_version,
            "thinking_level": self.thinking_level,
            "frame_context": self.frame_context,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ThoughtSignatureData":
        """Reconstruct from dictionary."""
        return cls(
            signature_id=data.get("signature_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            content=data.get("content", {}),
            model_version=data.get("model_version", "unknown"),
            thinking_level=data.get("thinking_level", "unknown"),
            frame_context=data.get("frame_context"),
        )


class ThoughtSignatureManager:
    """Manages thought signature capture, storage, and injection.
    
    Ensures continuous state across Gemini API calls by properly
    handling the thought_signature field in responses.
    """
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self._signature_cache: dict[str, ThoughtSignatureData] = {}
    
    def extract_signature(
        self,
        response: dict,
        model_version: str,
        thinking_level: str,
        frame_context: dict | None = None,
    ) -> dict | None:
        """Extract thought signature from Gemini API response.
        
        Looks for thought_signature in response candidates and
        extracts it for storage and future injection.
        
        Args:
            response: Raw Gemini API response
            model_version: Model identifier (flash/pro)
            thinking_level: Thinking level used (minimal/high)
            frame_context: Optional frame metadata for context
            
        Returns:
            Thought signature dictionary or None if not found
        """
        candidates = response.get("candidates", [])
        if not candidates:
            logger.debug("No candidates in response")
            return None
        
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        
        thought_content = None
        
        for part in parts:
            if "thought" in part:
                thought_content = part.get("thought")
                break
            if "thoughtSignature" in part:
                thought_content = part.get("thoughtSignature")
                break
            if isinstance(part, dict) and part.get("type") == "thought":
                thought_content = part
                break
        
        if thought_content is None:
            model_thinking = response.get("modelThinking", {})
            if model_thinking:
                thought_content = model_thinking
        
        if thought_content is None:
            logger.debug("No thought signature found in response")
            return None
        
        signature = ThoughtSignatureData(
            signature_id=str(uuid.uuid4()),
            timestamp=time.time(),
            content=thought_content if isinstance(thought_content, dict) else {"thought": thought_content},
            model_version=model_version,
            thinking_level=thinking_level,
            frame_context=frame_context,
        )
        
        self._signature_cache[signature.signature_id] = signature
        
        logger.info(f"Extracted thought signature: {signature.signature_id}")
        
        return signature.to_dict()
    
    def build_conversation_with_signatures(
        self,
        signatures: list[dict],
        system_prompt: str | None = None,
    ) -> list[dict]:
        """Build conversation history with injected thought signatures.
        
        Reconstructs the conversation including previous thought
        signatures to maintain reasoning context.
        
        Args:
            signatures: List of previous thought signature dicts
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Conversation history ready for API call
        """
        conversation = []
        
        if system_prompt:
            conversation.append({
                "role": "user",
                "parts": [{"text": system_prompt}],
            })
        
        for sig_dict in signatures:
            sig = ThoughtSignatureData.from_dict(sig_dict)
            
            conversation.append({
                "role": "model",
                "parts": [
                    {
                        "thought": sig.content.get("thought", sig.content),
                    }
                ],
            })
            
            if sig.frame_context:
                conversation.append({
                    "role": "user",
                    "parts": [
                        {
                            "text": f"Frame context: {sig.frame_context}",
                        }
                    ],
                })
        
        return conversation
    
    def inject_into_request(
        self,
        request_content: list[dict],
        signatures: list[dict],
    ) -> list[dict]:
        """Inject thought signatures into API request content.
        
        Ensures signatures are properly formatted and positioned
        in the request for Gemini to maintain context.
        
        Args:
            request_content: Original request content parts
            signatures: Thought signatures to inject
            
        Returns:
            Modified request content with injected signatures
        """
        if not signatures:
            return request_content
        
        injected_content = []
        
        for sig_dict in signatures[-self.max_history:]:
            sig = ThoughtSignatureData.from_dict(sig_dict)
            injected_content.append({
                "thought": sig.content.get("thought", sig.content),
            })
        
        injected_content.extend(request_content)
        
        return injected_content
    
    def prune_old_signatures(
        self,
        signatures: list[dict],
        max_age_seconds: float = 30.0,
    ) -> list[dict]:
        """Remove signatures older than max_age_seconds.
        
        Helps manage memory and ensures only relevant recent
        context is maintained.
        """
        current_time = time.time()
        pruned = []
        
        for sig_dict in signatures:
            timestamp = sig_dict.get("timestamp", 0)
            if current_time - timestamp <= max_age_seconds:
                pruned.append(sig_dict)
        
        removed_count = len(signatures) - len(pruned)
        if removed_count > 0:
            logger.debug(f"Pruned {removed_count} old signatures")
        
        return pruned
    
    def clear_cache(self) -> None:
        """Clear the internal signature cache."""
        self._signature_cache.clear()
        logger.info("Signature cache cleared")
