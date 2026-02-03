"""LangGraph node functions for Synthetic Cortex.

Implements the processing nodes for the vision state machine:
- Risk assessment
- Reflex layer (Flash)
- Cerebral layer (Pro)
- CAIR validation
- Clarification handling
"""

import logging
from typing import Any

from src.cortex_graph.state import (
    CortexState,
    Detection,
    CAIRMetrics,
    ThoughtSignature,
    classify_risk_level,
    HIGH_RISK_OBJECTS,
)
from src.gemini.client import GeminiClient
from src.gemini.thought_signatures import ThoughtSignatureManager
from src.cair.evaluator import CAIREvaluator

logger = logging.getLogger(__name__)

_gemini_client: GeminiClient | None = None
_signature_manager: ThoughtSignatureManager | None = None
_cair_evaluator: CAIREvaluator | None = None


def get_gemini_client() -> GeminiClient:
    """Lazy initialization of Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


def get_signature_manager() -> ThoughtSignatureManager:
    """Lazy initialization of signature manager."""
    global _signature_manager
    if _signature_manager is None:
        _signature_manager = ThoughtSignatureManager()
    return _signature_manager


def get_cair_evaluator() -> CAIREvaluator:
    """Lazy initialization of CAIR evaluator."""
    global _cair_evaluator
    if _cair_evaluator is None:
        _cair_evaluator = CAIREvaluator()
    return _cair_evaluator


def risk_assessment_node(state: CortexState) -> dict[str, Any]:
    """Assess environmental risk to determine processing mode.
    
    Analyzes frame metadata and any preliminary detections
    to route between Reflex (Flash) and Cerebral (Pro) processing.
    """
    logger.info("Executing risk assessment node")
    
    frame_metadata = state.get("frame_metadata", {})
    previous_detections = state.get("detections", [])
    
    detections = [
        Detection(
            object_type=d.get("object_type", "unknown"),
            confidence=d.get("confidence", 0.0),
            bbox=tuple(d.get("bbox", [0, 0, 0, 0])),
            risk_level=d.get("risk_level", "low"),
        )
        for d in previous_detections
    ]
    
    risk_level = classify_risk_level(detections)
    
    resolution_mode = frame_metadata.get("resolution_mode", "low")
    if resolution_mode == "high":
        risk_level = "high"
    
    processing_mode = "cerebral" if risk_level == "high" else "reflex"
    
    logger.info(f"Risk assessment: {risk_level}, mode: {processing_mode}")
    
    return {
        "risk_level": risk_level,
        "processing_mode": processing_mode,
    }


def reflex_layer_node(state: CortexState) -> dict[str, Any]:
    """Reflex Layer: Fast processing with Gemini Flash.
    
    Uses minimal thinking for sub-second response times.
    Suitable for general navigation and low-risk scenarios.
    """
    logger.info("Executing reflex layer (Flash)")
    
    client = get_gemini_client()
    sig_manager = get_signature_manager()
    
    previous_signatures = state.get("thought_signatures", [])
    conversation_history = sig_manager.build_conversation_with_signatures(
        previous_signatures
    )
    
    frame_data = state.get("frame_data", b"")
    frame_metadata = state.get("frame_metadata", {})
    
    response = client.analyze_frame(
        frame_data=frame_data,
        frame_metadata=frame_metadata,
        conversation_history=conversation_history,
        mode="reflex",
    )
    
    new_signature = sig_manager.extract_signature(
        response,
        model_version="gemini-2.0-flash",
        thinking_level="minimal",
    )
    
    detections = response.get("detections", [])
    confidence = response.get("confidence", 0.0)
    
    updated_signatures = previous_signatures.copy()
    if new_signature:
        updated_signatures.append(new_signature)
        logger.info(f"Thought signature captured: {new_signature.get('signature_id', 'unknown')} (Flash, minimal thinking)")
        if len(updated_signatures) > 10:
            updated_signatures = updated_signatures[-10:]
    
    return {
        "detections": detections,
        "confidence": confidence,
        "thought_signatures": updated_signatures,
        "model_used": "gemini-2.0-flash",
        "thinking_level": "minimal",
    }


def cerebral_layer_node(state: CortexState) -> dict[str, Any]:
    """Cerebral Layer: Deep reasoning with Gemini Pro.
    
    Uses high thinking level for complex scenarios.
    Activated for high-risk situations requiring careful analysis.
    """
    logger.info("Executing cerebral layer (Pro)")
    
    client = get_gemini_client()
    sig_manager = get_signature_manager()
    
    previous_signatures = state.get("thought_signatures", [])
    conversation_history = sig_manager.build_conversation_with_signatures(
        previous_signatures
    )
    
    frame_data = state.get("frame_data", b"")
    frame_metadata = state.get("frame_metadata", {})
    
    response = client.analyze_frame(
        frame_data=frame_data,
        frame_metadata=frame_metadata,
        conversation_history=conversation_history,
        mode="cerebral",
    )
    
    new_signature = sig_manager.extract_signature(
        response,
        model_version="gemini-2.0-pro",
        thinking_level="high",
    )
    
    detections = response.get("detections", [])
    confidence = response.get("confidence", 0.0)
    
    updated_signatures = previous_signatures.copy()
    if new_signature:
        updated_signatures.append(new_signature)
        logger.info(f"Thought signature captured: {new_signature.get('signature_id', 'unknown')} (Pro, high thinking)")
        if len(updated_signatures) > 10:
            updated_signatures = updated_signatures[-10:]
    
    return {
        "detections": detections,
        "confidence": confidence,
        "thought_signatures": updated_signatures,
        "model_used": "gemini-2.0-pro",
        "thinking_level": "high",
    }


def cair_check_node(state: CortexState) -> dict[str, Any]:
    """CAIR validation node.
    
    Evaluates confidence against threshold and determines
    if user clarification is needed before autonomous action.
    """
    logger.info("Executing CAIR check")
    
    evaluator = get_cair_evaluator()
    
    confidence = state.get("confidence", 0.0)
    detections = state.get("detections", [])
    risk_level = state.get("risk_level", "low")
    
    cair_metrics = evaluator.calculate_metrics(
        confidence=confidence,
        detections=detections,
        risk_level=risk_level,
    )
    
    threshold = evaluator.get_threshold(risk_level)
    requires_clarification = confidence < threshold
    
    clarification_reason = None
    if requires_clarification:
        low_confidence_detections = [
            d for d in detections
            if d.get("confidence", 0) < threshold
        ]
        if low_confidence_detections:
            objects = [d.get("object_type", "object") for d in low_confidence_detections[:3]]
            clarification_reason = f"Low confidence detection: {', '.join(objects)}"
        else:
            clarification_reason = f"Overall confidence {confidence:.0%} below threshold {threshold:.0%}"
    
    evaluator.log_decision(
        state=state,
        cair_metrics=cair_metrics,
        requires_clarification=requires_clarification,
    )
    
    return {
        "cair_metrics": cair_metrics.to_dict() if hasattr(cair_metrics, 'to_dict') else cair_metrics,
        "requires_clarification": requires_clarification,
        "clarification_reason": clarification_reason,
    }


def ask_clarification_node(state: CortexState) -> dict[str, Any]:
    """Request user clarification via spatial audio.
    
    Triggered when CAIR score falls below threshold.
    Generates audio prompt for user verification.
    """
    logger.info("Executing clarification request")
    
    clarification_reason = state.get("clarification_reason", "Verification needed")
    detections = state.get("detections", [])
    
    primary_detection = None
    if detections:
        primary_detection = max(detections, key=lambda d: d.get("confidence", 0))
    
    clarification_data = {
        "type": "clarification_request",
        "reason": clarification_reason,
        "primary_object": primary_detection.get("object_type") if primary_detection else None,
        "position": primary_detection.get("bbox") if primary_detection else None,
        "confidence": primary_detection.get("confidence") if primary_detection else 0.0,
    }
    
    logger.info(f"Clarification requested: {clarification_reason}")
    
    return {
        "messages": [{"role": "system", "content": f"CLARIFICATION_REQUEST: {clarification_data}"}],
    }


def emit_detections_node(state: CortexState) -> dict[str, Any]:
    """Emit final detections to feedback node.
    
    Packages detection results for spatial audio generation.
    """
    logger.info("Emitting detections")
    
    detections = state.get("detections", [])
    confidence = state.get("confidence", 0.0)
    model_used = state.get("model_used", "")
    
    output = {
        "detections": detections,
        "confidence": confidence,
        "model_used": model_used,
        "frame_metadata": state.get("frame_metadata", {}),
    }
    
    logger.info(f"Emitting {len(detections)} detections with confidence {confidence:.2%}")
    
    return output


def route_by_risk(state: CortexState) -> str:
    """Routing function: determine processing layer based on risk."""
    risk_level = state.get("risk_level", "low")
    return "cerebral_layer" if risk_level == "high" else "reflex_layer"


def route_by_confidence(state: CortexState) -> str:
    """Routing function: determine if clarification needed."""
    requires_clarification = state.get("requires_clarification", False)
    return "ask_clarification" if requires_clarification else "emit_detections"
