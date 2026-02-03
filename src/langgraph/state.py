"""LangGraph state schema for Synthetic Cortex.

Defines the state structure including ThoughtSignature management
for maintaining object permanence and reasoning context.
"""

from typing import Annotated, Literal
from dataclasses import dataclass, field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


@dataclass
class Detection:
    """Single object detection result."""
    
    object_type: str
    confidence: float
    bbox: tuple[float, float, float, float]
    risk_level: Literal["low", "medium", "high"]
    distance_estimate: float | None = None
    velocity_estimate: tuple[float, float] | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "object_type": self.object_type,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "risk_level": self.risk_level,
            "distance_estimate": self.distance_estimate,
            "velocity_estimate": list(self.velocity_estimate) if self.velocity_estimate else None,
        }


@dataclass
class ThoughtSignature:
    """Captures Gemini 3 thought signature for state continuity.
    
    Critical for maintaining object permanence across frames
    and ensuring reasoning context is preserved.
    """
    
    signature_id: str
    timestamp: float
    content: dict
    model_version: str
    thinking_level: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API injection."""
        return {
            "signature_id": self.signature_id,
            "timestamp": self.timestamp,
            "content": self.content,
            "model_version": self.model_version,
            "thinking_level": self.thinking_level,
        }
    
    @classmethod
    def from_response(cls, response: dict, model_version: str, thinking_level: str) -> "ThoughtSignature | None":
        """Extract thought signature from Gemini response."""
        candidates = response.get("candidates", [])
        if not candidates:
            return None
        
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        
        for part in parts:
            if "thought" in part or "thoughtSignature" in part:
                import time
                import uuid
                return cls(
                    signature_id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    content=part,
                    model_version=model_version,
                    thinking_level=thinking_level,
                )
        
        return None


@dataclass
class CAIRMetrics:
    """CAIR (Confidence in AI Results) metrics container."""
    
    value_score: float = 0.0
    risk_score: float = 0.0
    correction_factor: float = 1.0
    
    @property
    def cair_score(self) -> float:
        """Calculate CAIR = Value / (Risk x Correction)."""
        denominator = self.risk_score * self.correction_factor
        if denominator <= 0:
            return self.value_score
        return self.value_score / denominator
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "value_score": self.value_score,
            "risk_score": self.risk_score,
            "correction_factor": self.correction_factor,
            "cair_score": self.cair_score,
        }


class CortexState(TypedDict):
    """Main state schema for LangGraph Cortex.
    
    Maintains all context needed for vision processing,
    model routing, and safety verification.
    """
    
    frame_data: bytes
    frame_metadata: dict
    
    thought_signatures: Annotated[list[dict], add_messages]
    
    detections: list[dict]
    
    confidence: float
    risk_level: Literal["low", "high"]
    requires_clarification: bool
    clarification_reason: str | None
    
    cair_metrics: dict
    
    processing_mode: Literal["reflex", "cerebral"]
    model_used: str
    thinking_level: str
    
    halt_requested: bool
    
    messages: Annotated[list, add_messages]


def create_initial_state(
    frame_data: bytes,
    frame_metadata: dict,
    previous_signatures: list[dict] | None = None,
) -> CortexState:
    """Factory function to create initial state for graph execution."""
    return CortexState(
        frame_data=frame_data,
        frame_metadata=frame_metadata,
        thought_signatures=previous_signatures or [],
        detections=[],
        confidence=0.0,
        risk_level="low",
        requires_clarification=False,
        clarification_reason=None,
        cair_metrics=CAIRMetrics().to_dict(),
        processing_mode="reflex",
        model_used="",
        thinking_level="minimal",
        halt_requested=False,
        messages=[],
    )


HIGH_RISK_OBJECTS = frozenset([
    "vehicle", "car", "truck", "bus", "motorcycle", "bicycle",
    "person_running", "construction", "water", "stairs",
    "edge", "hole", "cliff", "traffic",
])

MEDIUM_RISK_OBJECTS = frozenset([
    "person", "animal", "dog", "cat", "door", "furniture",
    "chair", "table", "pole", "sign",
])


def classify_risk_level(detections: list[Detection]) -> Literal["low", "high"]:
    """Determine overall risk level based on detected objects."""
    for detection in detections:
        if detection.object_type.lower() in HIGH_RISK_OBJECTS:
            return "high"
        if detection.risk_level == "high":
            return "high"
    
    high_confidence_medium_risk = any(
        d.object_type.lower() in MEDIUM_RISK_OBJECTS and d.confidence > 0.8
        for d in detections
    )
    
    if high_confidence_medium_risk and len(detections) > 3:
        return "high"
    
    return "low"
