"""Cortex Operator Node for dora-rs dataflow.

Processes frames through the LangGraph state machine using
Gemini 3 for vision analysis with CAIR-aware routing.
"""

import os
import json
import logging
from typing import Any

import pyarrow as pa
from dora import Node

from src.cortex_graph.graph import get_cortex_state_machine, CortexStateMachine
from src.mcp.server import get_mcp_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CortexOperator:
    """LangGraph-managed vision processing operator."""

    def __init__(self):
        self.cair_threshold = float(os.getenv("CAIR_THRESHOLD", "0.85"))
        self.default_risk_level = os.getenv("DEFAULT_RISK_LEVEL", "low")
        
        self._state_machine: CortexStateMachine | None = None
        self._mcp_server = None
        self._frame_count = 0
        self._halted = False
        
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the cortex components."""
        try:
            self._state_machine = get_cortex_state_machine()
            self._mcp_server = get_mcp_server()
            logger.info("Cortex operator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize cortex: {e}")
            raise

    def _deserialize_frame(self, arrow_data: bytes) -> tuple[bytes, dict]:
        """Deserialize Arrow-encoded frame data.
        
        Args:
            arrow_data: Arrow IPC serialized frame
            
        Returns:
            Tuple of (frame_bytes, metadata_dict)
        """
        try:
            reader = pa.ipc.open_stream(arrow_data)
            table = reader.read_all()
            
            frame_data = table.column("frame_data")[0].as_py()
            metadata = {
                "frame_id": table.column("frame_id")[0].as_py(),
                "timestamp": table.column("timestamp")[0].as_py(),
                "width": table.column("width")[0].as_py(),
                "height": table.column("height")[0].as_py(),
                "channels": table.column("channels")[0].as_py(),
                "resolution_mode": table.column("resolution_mode")[0].as_py(),
            }
            
            return frame_data, metadata
            
        except Exception as e:
            logger.error(f"Frame deserialization failed: {e}")
            return b"", {}

    def process_frame(
        self,
        frame_data: bytes,
        frame_metadata: dict,
    ) -> dict[str, Any]:
        """Process a single frame through the state machine.
        
        Args:
            frame_data: Raw frame bytes
            frame_metadata: Frame metadata dictionary
            
        Returns:
            Processing result with detections
        """
        if self._halted:
            logger.warning("Cortex is halted, skipping frame")
            return {"detections": [], "halted": True}
        
        if self._state_machine is None:
            logger.error("State machine not initialized")
            return {"detections": [], "error": "State machine unavailable"}
        
        self._frame_count += 1
        thread_id = f"frame_{self._frame_count}"
        
        try:
            result = self._state_machine.process_frame(
                frame_data=frame_data,
                frame_metadata=frame_metadata,
                thread_id=thread_id,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return {"detections": [], "error": str(e)}

    def handle_halt_signal(self) -> None:
        """Handle emergency halt signal."""
        self._halted = True
        if self._state_machine:
            self._state_machine.emergency_halt()
        logger.warning("Cortex halted via signal")

    def resume(self) -> None:
        """Resume processing after halt."""
        self._halted = False
        if self._mcp_server:
            self._mcp_server.clear_halt()
        logger.info("Cortex resumed")

    def get_status(self) -> dict[str, Any]:
        """Get current operator status."""
        return {
            "halted": self._halted,
            "frames_processed": self._frame_count,
            "cair_threshold": self.cair_threshold,
        }


_operator: CortexOperator | None = None


def on_input(
    dora_input: dict[str, Any],
    send_output: Any,
    dora_event: Any,
) -> None:
    """dora-rs input handler for cortex operator."""
    global _operator
    
    if _operator is None:
        _operator = CortexOperator()
    
    input_id = dora_input.get("id", "")
    
    if input_id == "halt_signal":
        _operator.handle_halt_signal()
        return
    
    if input_id == "resume_signal":
        _operator.resume()
        return
    
    if input_id == "frame":
        arrow_data = dora_input.get("value", b"")
        
        if not arrow_data:
            logger.warning("Empty frame received")
            return
        
        frame_data, frame_metadata = _operator._deserialize_frame(arrow_data)
        
        if not frame_data:
            logger.warning("Failed to deserialize frame")
            return
        
        result = _operator.process_frame(frame_data, frame_metadata)
        
        detections_json = json.dumps({
            "detections": result.get("detections", []),
            "confidence": result.get("confidence", 0.0),
            "model_used": result.get("model_used", ""),
            "frame_metadata": frame_metadata,
        }).encode("utf-8")
        
        send_output("detections", detections_json, dora_input.get("metadata", {}))
        
        if result.get("requires_clarification"):
            clarification_json = json.dumps({
                "reason": result.get("clarification_reason", ""),
                "confidence": result.get("confidence", 0.0),
                "detections": result.get("detections", []),
            }).encode("utf-8")
            
            send_output("clarification_request", clarification_json, dora_input.get("metadata", {}))
        
        state_update = json.dumps({
            "frame_id": frame_metadata.get("frame_id"),
            "halted": _operator._halted,
            "cair_metrics": result.get("cair_metrics", {}),
        }).encode("utf-8")
        
        send_output("state_update", state_update, dora_input.get("metadata", {}))


def on_stop() -> None:
    """Cleanup handler for dora-rs."""
    global _operator
    if _operator is not None:
        _operator.handle_halt_signal()
        _operator = None
        logger.info("Cortex operator stopped")
