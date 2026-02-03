"""CAIR (Confidence in AI Results) Evaluator for Synthetic Cortex.

Implements: CAIR = Value / (Risk x Correction)

Metrics tracked:
- Value: Successful hazard detections
- Risk: False positive rate, latency spikes
- Correction: Time-to-correction when model misidentifies

Integrates with LangSmith for observability and tracing.
"""

import os
import time
import logging
from typing import Any
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class CAIRMetrics:
    """Container for CAIR metric components."""
    
    value_score: float = 0.0
    risk_score: float = 0.001
    correction_factor: float = 1.0
    
    successful_detections: int = 0
    total_frames: int = 0
    false_positives: int = 0
    corrections_made: int = 0
    total_correction_time_ms: float = 0.0
    
    @property
    def cair_score(self) -> float:
        """Calculate CAIR = Value / (Risk x Correction)."""
        denominator = self.risk_score * self.correction_factor
        if denominator <= 0:
            denominator = 0.001
        return self.value_score / denominator
    
    @property
    def average_correction_time_ms(self) -> float:
        """Average time to correction in milliseconds."""
        if self.corrections_made == 0:
            return 0.0
        return self.total_correction_time_ms / self.corrections_made
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cair_score": self.cair_score,
            "value_score": self.value_score,
            "risk_score": self.risk_score,
            "correction_factor": self.correction_factor,
            "successful_detections": self.successful_detections,
            "total_frames": self.total_frames,
            "false_positives": self.false_positives,
            "corrections_made": self.corrections_made,
            "average_correction_time_ms": self.average_correction_time_ms,
        }


@dataclass
class DetectionEvent:
    """Records a detection event for metric calculation."""
    
    timestamp: float
    detections: list[dict]
    confidence: float
    risk_level: str
    model_used: str
    latency_ms: float
    was_corrected: bool = False
    correction_time_ms: float = 0.0


class CAIREvaluator:
    """Evaluates CAIR metrics and manages thresholds.
    
    Tracks detection history to calculate Value, Risk, and Correction
    components of the CAIR metric.
    """
    
    DEFAULT_THRESHOLDS = {
        "low": 0.70,
        "high": 0.85,
    }
    
    def __init__(
        self,
        window_size_seconds: float = 60.0,
        langsmith_project: str | None = None,
    ):
        self.window_size_seconds = window_size_seconds
        self.langsmith_project = langsmith_project or os.getenv(
            "LANGSMITH_PROJECT", "synthetic-cortex"
        )
        
        self._event_history: deque[DetectionEvent] = deque()
        self._current_metrics = CAIRMetrics()
        self._thresholds = self.DEFAULT_THRESHOLDS.copy()
        
        self._last_detection_id: str | None = None
        self._pending_correction: DetectionEvent | None = None
        
        self._langsmith_client = None
        self._init_langsmith()
        
        logger.info(f"CAIR Evaluator initialized (window: {window_size_seconds}s)")
    
    def _init_langsmith(self) -> None:
        """Initialize LangSmith client for observability."""
        try:
            from langsmith import Client
            
            api_key = os.getenv("LANGSMITH_API_KEY")
            if api_key:
                self._langsmith_client = Client()
                logger.info("LangSmith client initialized")
            else:
                logger.warning("LANGSMITH_API_KEY not set, tracing disabled")
        except ImportError:
            logger.warning("LangSmith not available, tracing disabled")
    
    def set_threshold(self, risk_level: str, threshold: float) -> None:
        """Set confidence threshold for a risk level."""
        self._thresholds[risk_level] = threshold
        logger.info(f"Threshold for {risk_level} set to {threshold:.2%}")
    
    def get_threshold(self, risk_level: str) -> float:
        """Get confidence threshold for a risk level."""
        return self._thresholds.get(risk_level, self._thresholds["high"])
    
    def calculate_metrics(
        self,
        confidence: float,
        detections: list[dict],
        risk_level: str,
        latency_ms: float = 0.0,
        model_used: str = "",
    ) -> CAIRMetrics:
        """Calculate CAIR metrics for current frame.
        
        Args:
            confidence: Overall detection confidence
            detections: List of detection results
            risk_level: Current risk assessment
            latency_ms: Processing latency
            model_used: Model identifier
            
        Returns:
            Updated CAIRMetrics
        """
        current_time = time.time()
        
        event = DetectionEvent(
            timestamp=current_time,
            detections=detections,
            confidence=confidence,
            risk_level=risk_level,
            model_used=model_used,
            latency_ms=latency_ms,
        )
        
        self._event_history.append(event)
        self._prune_old_events(current_time)
        
        self._update_metrics(event)
        
        return self._current_metrics
    
    def _prune_old_events(self, current_time: float) -> None:
        """Remove events outside the sliding window."""
        cutoff = current_time - self.window_size_seconds
        
        while self._event_history and self._event_history[0].timestamp < cutoff:
            self._event_history.popleft()
    
    def _update_metrics(self, event: DetectionEvent) -> None:
        """Update metrics based on new event."""
        m = self._current_metrics
        
        m.total_frames += 1
        
        high_confidence_count = sum(
            1 for d in event.detections
            if d.get("confidence", 0) >= self.get_threshold(event.risk_level)
        )
        
        m.successful_detections += high_confidence_count
        
        if m.total_frames > 0:
            m.value_score = m.successful_detections / m.total_frames
        
        latency_threshold_ms = 500 if event.risk_level == "low" else 2000
        is_latency_spike = event.latency_ms > latency_threshold_ms
        
        base_risk = 1.0 - event.confidence
        latency_penalty = 0.1 if is_latency_spike else 0.0
        
        alpha = 0.1
        m.risk_score = (1 - alpha) * m.risk_score + alpha * (base_risk + latency_penalty)
        m.risk_score = max(0.001, min(1.0, m.risk_score))
        
        if m.corrections_made > 0:
            m.correction_factor = 1.0 + (m.average_correction_time_ms / 1000.0) * 0.5
        else:
            m.correction_factor = 1.0
    
    def record_correction(
        self,
        detection_id: str | None = None,
        correction_time_ms: float = 0.0,
    ) -> None:
        """Record a correction event (model misidentification).
        
        Args:
            detection_id: ID of the corrected detection
            correction_time_ms: Time taken to correct in milliseconds
        """
        m = self._current_metrics
        
        m.corrections_made += 1
        m.total_correction_time_ms += correction_time_ms
        
        m.correction_factor = 1.0 + (m.average_correction_time_ms / 1000.0) * 0.5
        
        logger.info(f"Correction recorded: {correction_time_ms:.0f}ms")
        
        self._log_to_langsmith("correction", {
            "detection_id": detection_id,
            "correction_time_ms": correction_time_ms,
            "total_corrections": m.corrections_made,
        })
    
    def record_false_positive(self) -> None:
        """Record a false positive detection."""
        self._current_metrics.false_positives += 1
        
        alpha = 0.05
        self._current_metrics.risk_score = min(
            1.0,
            self._current_metrics.risk_score + alpha,
        )
        
        logger.info(f"False positive recorded (total: {self._current_metrics.false_positives})")
    
    def log_decision(
        self,
        state: dict[str, Any],
        cair_metrics: Any,
        requires_clarification: bool,
    ) -> None:
        """Log decision to LangSmith for observability.
        
        Args:
            state: Current graph state
            cair_metrics: CAIR metrics at decision time
            requires_clarification: Whether clarification was needed
        """
        metrics_dict = (
            cair_metrics.to_dict() if hasattr(cair_metrics, 'to_dict') else cair_metrics
        )
        
        log_data = {
            "confidence": state.get("confidence", 0.0),
            "risk_level": state.get("risk_level", "unknown"),
            "model_used": state.get("model_used", "unknown"),
            "requires_clarification": requires_clarification,
            "detection_count": len(state.get("detections", [])),
            "cair_metrics": metrics_dict,
        }
        
        logger.info(
            f"Decision logged: CAIR={metrics_dict.get('cair_score', 0):.3f}, "
            f"clarification={requires_clarification}"
        )
        
        self._log_to_langsmith("decision", log_data)
    
    def _log_to_langsmith(self, event_type: str, data: dict[str, Any]) -> None:
        """Send trace data to LangSmith."""
        if self._langsmith_client is None:
            return
        
        try:
            self._langsmith_client.create_run(
                name=f"cair_{event_type}",
                run_type="chain",
                inputs=data,
                outputs={"cair_score": self._current_metrics.cair_score},
                project_name=self.langsmith_project,
            )
        except Exception as e:
            logger.debug(f"LangSmith logging failed: {e}")
    
    def get_current_metrics(self) -> CAIRMetrics:
        """Get current CAIR metrics."""
        return self._current_metrics
    
    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of CAIR metrics for reporting."""
        m = self._current_metrics
        return {
            "cair_score": m.cair_score,
            "value": m.value_score,
            "risk": m.risk_score,
            "correction_factor": m.correction_factor,
            "frames_processed": m.total_frames,
            "successful_detections": m.successful_detections,
            "false_positives": m.false_positives,
            "corrections": m.corrections_made,
            "avg_correction_time_ms": m.average_correction_time_ms,
            "window_events": len(self._event_history),
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._event_history.clear()
        self._current_metrics = CAIRMetrics()
        logger.info("CAIR metrics reset")


_evaluator: CAIREvaluator | None = None


def get_cair_evaluator() -> CAIREvaluator:
    """Get or create the global CAIR evaluator."""
    global _evaluator
    if _evaluator is None:
        _evaluator = CAIREvaluator()
    return _evaluator
