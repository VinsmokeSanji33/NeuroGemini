"""Gemini 3 API Client with model routing.

Implements dual-model routing:
- Reflex Layer (Flash): thinking_level="minimal", media_resolution="low"
- Cerebral Layer (Pro): thinking_level="high", media_resolution="high"

Handles thought signature capture and injection for state continuity.
"""

import os
import base64
import logging
import time
from typing import Any, Literal

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)


VISION_SYSTEM_PROMPT = """You are a vision system for sensory substitution, helping visually impaired users navigate their environment safely.

Your task is to analyze the camera frame and identify:
1. Obstacles and hazards in the path
2. Moving objects (vehicles, people, animals) with direction and speed estimates
3. Environmental features (stairs, edges, water, construction)
4. Safe navigation paths

Output format (JSON):
{
    "detections": [
        {
            "object_type": "string",
            "confidence": 0.0-1.0,
            "bbox": [x1, y1, x2, y2],  // normalized 0-1
            "risk_level": "low|medium|high",
            "distance_estimate": float_meters_or_null,
            "description": "brief description"
        }
    ],
    "scene_summary": "brief overall scene description",
    "navigation_advice": "immediate navigation guidance",
    "confidence": 0.0-1.0  // overall scene confidence
}

Prioritize safety - when uncertain, classify as higher risk.
Be concise but precise in descriptions for audio feedback."""


class GeminiClient:
    """Gemini 3 API client with dual-model routing."""
    
    MODEL_CONFIGS = {
        "reflex": {
            "model": "gemini-2.0-flash",
            "thinking_level": "minimal",
            "media_resolution": "low",
            "max_output_tokens": 1024,
            "temperature": 0.3,
        },
        "cerebral": {
            "model": "gemini-2.0-pro",
            "thinking_level": "high",
            "media_resolution": "high",
            "max_output_tokens": 2048,
            "temperature": 0.1,
        },
    }
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")
        
        genai.configure(api_key=self.api_key)
        
        self._models: dict[str, Any] = {}
        self._initialize_models()
        
        logger.info("Gemini client initialized")
    
    def _initialize_models(self) -> None:
        """Initialize model instances for both layers."""
        for mode, config in self.MODEL_CONFIGS.items():
            model_name = config["model"]
            
            generation_config = GenerationConfig(
                max_output_tokens=config["max_output_tokens"],
                temperature=config["temperature"],
                response_mime_type="application/json",
            )
            
            self._models[mode] = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction=VISION_SYSTEM_PROMPT,
            )
            
            logger.info(f"Initialized {mode} model: {model_name}")
    
    def analyze_frame(
        self,
        frame_data: bytes,
        frame_metadata: dict,
        conversation_history: list[dict] | None = None,
        mode: Literal["reflex", "cerebral"] = "reflex",
    ) -> dict[str, Any]:
        """Analyze a camera frame using the specified processing mode.
        
        Args:
            frame_data: Raw frame bytes (JPEG or PNG encoded)
            frame_metadata: Frame metadata (dimensions, timestamp, etc.)
            conversation_history: Previous conversation with thought signatures
            mode: Processing mode (reflex=Flash, cerebral=Pro)
            
        Returns:
            Analysis result with detections and thought signature
        """
        config = self.MODEL_CONFIGS[mode]
        model = self._models[mode]
        
        start_time = time.time()
        
        if frame_data[:4] == b'\x89PNG' or frame_data[:2] == b'\xff\xd8':
            image_data = base64.b64encode(frame_data).decode("utf-8")
        else:
            import cv2
            import numpy as np
            
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            height = frame_metadata.get("height", 480)
            width = frame_metadata.get("width", 640)
            channels = frame_metadata.get("channels", 3)
            
            frame_array = frame_array.reshape((height, width, channels))
            _, encoded = cv2.imencode(".jpg", frame_array, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_data = base64.b64encode(encoded.tobytes()).decode("utf-8")
        
        content_parts = []
        
        if conversation_history:
            for msg in conversation_history:
                if "thought" in str(msg):
                    content_parts.append(msg)
        
        content_parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": image_data,
            }
        })
        
        context_text = (
            f"Frame {frame_metadata.get('frame_id', 'unknown')} at "
            f"{frame_metadata.get('timestamp', 'unknown')}. "
            f"Resolution: {frame_metadata.get('width', '?')}x{frame_metadata.get('height', '?')}. "
            f"Mode: {config['media_resolution']} resolution analysis."
        )
        content_parts.append({"text": context_text})
        
        try:
            response = model.generate_content(
                content_parts,
                generation_config=GenerationConfig(
                    max_output_tokens=config["max_output_tokens"],
                    temperature=config["temperature"],
                    response_mime_type="application/json",
                ),
            )
            
            latency = time.time() - start_time
            logger.info(f"{mode} analysis completed in {latency:.3f}s")
            
            return self._parse_response(response, mode, latency)
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._fallback_response(str(e))
    
    def _parse_response(
        self,
        response: Any,
        mode: str,
        latency: float,
    ) -> dict[str, Any]:
        """Parse Gemini response into structured format."""
        import json
        
        result = {
            "detections": [],
            "confidence": 0.0,
            "scene_summary": "",
            "navigation_advice": "",
            "model_used": self.MODEL_CONFIGS[mode]["model"],
            "thinking_level": self.MODEL_CONFIGS[mode]["thinking_level"],
            "latency_ms": latency * 1000,
            "raw_response": None,
            "candidates": [],
        }
        
        try:
            if hasattr(response, "text"):
                text = response.text
                parsed = json.loads(text)
                
                result["detections"] = parsed.get("detections", [])
                result["confidence"] = parsed.get("confidence", 0.0)
                result["scene_summary"] = parsed.get("scene_summary", "")
                result["navigation_advice"] = parsed.get("navigation_advice", "")
            
            if hasattr(response, "candidates"):
                result["candidates"] = [
                    {
                        "content": {
                            "parts": [
                                {"text": c.content.parts[0].text if c.content.parts else ""}
                            ]
                        }
                    }
                    for c in response.candidates
                ]
            
            if hasattr(response, "_result") and response._result:
                raw = response._result
                if hasattr(raw, "model_thinking"):
                    result["raw_response"] = {"modelThinking": raw.model_thinking}
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            result["scene_summary"] = response.text if hasattr(response, "text") else ""
            result["confidence"] = 0.5
        
        return result
    
    def _fallback_response(self, error: str) -> dict[str, Any]:
        """Generate fallback response on API error."""
        return {
            "detections": [],
            "confidence": 0.0,
            "scene_summary": f"Analysis unavailable: {error}",
            "navigation_advice": "Proceed with caution - system error",
            "model_used": "fallback",
            "thinking_level": "none",
            "latency_ms": 0,
            "error": error,
        }
    
    def switch_mode(self, mode: Literal["reflex", "cerebral"]) -> dict[str, Any]:
        """Get configuration for specified mode."""
        return self.MODEL_CONFIGS.get(mode, self.MODEL_CONFIGS["reflex"])
    
    def get_model_info(self) -> dict[str, Any]:
        """Return information about configured models."""
        return {
            "modes": list(self.MODEL_CONFIGS.keys()),
            "models": {k: v["model"] for k, v in self.MODEL_CONFIGS.items()},
            "initialized": list(self._models.keys()),
        }
