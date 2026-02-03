"""LangGraph StateMachine for Synthetic Cortex.

Defines the state machine that manages transitions between
Reflex (minimal thinking) and Cerebral (high thinking) layers
based on environmental risk assessment.
"""

import logging
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.cortex_graph.state import CortexState, create_initial_state
from src.cortex_graph.nodes import (
    risk_assessment_node,
    reflex_layer_node,
    cerebral_layer_node,
    cair_check_node,
    ask_clarification_node,
    emit_detections_node,
    route_by_risk,
    route_by_confidence,
)

logger = logging.getLogger(__name__)


def build_cortex_graph() -> StateGraph:
    """Build the Synthetic Cortex state machine.
    
    Graph Flow:
    
    START -> RiskAssessment
                |
                ├── (low risk) -> ReflexLayer -> CAIRCheck
                |                                    |
                └── (high risk) -> CerebralLayer ----┘
                                                     |
                                    ├── (confidence >= threshold) -> EmitDetections -> END
                                    |
                                    └── (confidence < threshold) -> AskClarification -> EmitDetections -> END
    """
    graph = StateGraph(CortexState)
    
    graph.add_node("risk_assessment", risk_assessment_node)
    graph.add_node("reflex_layer", reflex_layer_node)
    graph.add_node("cerebral_layer", cerebral_layer_node)
    graph.add_node("cair_check", cair_check_node)
    graph.add_node("ask_clarification", ask_clarification_node)
    graph.add_node("emit_detections", emit_detections_node)
    
    graph.set_entry_point("risk_assessment")
    
    graph.add_conditional_edges(
        "risk_assessment",
        route_by_risk,
        {
            "reflex_layer": "reflex_layer",
            "cerebral_layer": "cerebral_layer",
        }
    )
    
    graph.add_edge("reflex_layer", "cair_check")
    graph.add_edge("cerebral_layer", "cair_check")
    
    graph.add_conditional_edges(
        "cair_check",
        route_by_confidence,
        {
            "ask_clarification": "ask_clarification",
            "emit_detections": "emit_detections",
        }
    )
    
    graph.add_edge("ask_clarification", "emit_detections")
    
    graph.add_edge("emit_detections", END)
    
    return graph


class CortexStateMachine:
    """Wrapper class for the Cortex state machine.
    
    Manages graph compilation, checkpointing, and execution.
    """
    
    def __init__(self, enable_checkpointing: bool = True):
        self.graph = build_cortex_graph()
        
        if enable_checkpointing:
            self.checkpointer = MemorySaver()
            self.compiled = self.graph.compile(checkpointer=self.checkpointer)
        else:
            self.checkpointer = None
            self.compiled = self.graph.compile()
        
        self._thought_signature_buffer: list[dict] = []
        self._max_signatures = 10
        
        logger.info("Cortex state machine initialized")
    
    def process_frame(
        self,
        frame_data: bytes,
        frame_metadata: dict,
        thread_id: str = "main",
    ) -> dict[str, Any]:
        """Process a single frame through the state machine.
        
        Args:
            frame_data: Raw frame bytes (Arrow serialized)
            frame_metadata: Frame metadata dictionary
            thread_id: Thread identifier for checkpointing
            
        Returns:
            Processing result containing detections and state
        """
        initial_state = create_initial_state(
            frame_data=frame_data,
            frame_metadata=frame_metadata,
            previous_signatures=self._thought_signature_buffer.copy(),
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        result = None
        for event in self.compiled.stream(initial_state, config):
            result = event
            logger.debug(f"Graph event: {list(event.keys())}")
        
        if result and "emit_detections" in result:
            output = result["emit_detections"]
        elif result:
            output = list(result.values())[-1] if result else {}
        else:
            output = {}
        
        if "thought_signatures" in initial_state:
            self._update_signature_buffer(initial_state.get("thought_signatures", []))
        
        return output
    
    def _update_signature_buffer(self, new_signatures: list[dict]) -> None:
        """Maintain rolling buffer of thought signatures."""
        self._thought_signature_buffer = new_signatures[-self._max_signatures:]
    
    def get_state_snapshot(self, thread_id: str = "main") -> dict[str, Any] | None:
        """Get current state snapshot for debugging."""
        if self.checkpointer is None:
            return None
        
        config = {"configurable": {"thread_id": thread_id}}
        return self.compiled.get_state(config)
    
    def reset(self) -> None:
        """Reset state machine and clear signature buffer."""
        self._thought_signature_buffer.clear()
        logger.info("Cortex state machine reset")
    
    def emergency_halt(self) -> None:
        """Trigger emergency halt - clears all state."""
        self.reset()
        logger.warning("EMERGENCY HALT triggered")


_state_machine: CortexStateMachine | None = None


def get_cortex_state_machine() -> CortexStateMachine:
    """Get or create the global state machine instance."""
    global _state_machine
    if _state_machine is None:
        _state_machine = CortexStateMachine()
    return _state_machine


def process_frame(
    frame_data: bytes,
    frame_metadata: dict,
    thread_id: str = "main",
) -> dict[str, Any]:
    """Convenience function to process a frame."""
    machine = get_cortex_state_machine()
    return machine.process_frame(frame_data, frame_metadata, thread_id)
