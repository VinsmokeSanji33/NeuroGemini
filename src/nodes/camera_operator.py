"""Camera Operator Node for dora-rs dataflow.

Captures RTSP stream from mobile phone and serializes frames
using Apache Arrow for zero-copy transfer to downstream nodes.
"""

import os
import time
import logging
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
from dora import Node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraOperator:
    """RTSP Camera capture with Arrow serialization."""

    def __init__(self):
        self.rtsp_url = os.getenv("RTSP_URL", "rtsp://192.168.1.100:5554/live")
        self.target_fps = int(os.getenv("TARGET_FPS", "10"))
        self.frame_width = int(os.getenv("FRAME_WIDTH", "640"))
        self.frame_height = int(os.getenv("FRAME_HEIGHT", "480"))
        
        self.cap = None
        self.frame_interval = 1.0 / self.target_fps
        self.last_frame_time = 0.0
        self.frame_count = 0
        self.resolution_mode = "low"
        
        self._connect()

    def _connect(self) -> bool:
        """Establish connection to RTSP stream."""
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to connect to RTSP stream: {self.rtsp_url}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        logger.info(f"Connected to RTSP stream: {self.rtsp_url}")
        return True

    def set_resolution(self, mode: str) -> None:
        """Switch between low and high resolution modes."""
        if mode == "high":
            self.frame_width = 1280
            self.frame_height = 720
            self.target_fps = 2
        else:
            self.frame_width = 640
            self.frame_height = 480
            self.target_fps = 10
        
        self.resolution_mode = mode
        self.frame_interval = 1.0 / self.target_fps
        
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        logger.info(f"Resolution mode set to: {mode}")

    def capture_frame(self) -> tuple[np.ndarray | None, dict]:
        """Capture a single frame with rate limiting."""
        current_time = time.time()
        
        if current_time - self.last_frame_time < self.frame_interval:
            return None, {}
        
        if self.cap is None or not self.cap.isOpened():
            if not self._connect():
                return None, {}
        
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning("Frame capture failed, attempting reconnect")
            self._connect()
            return None, {}
        
        if frame.shape[:2] != (self.frame_height, self.frame_width):
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        self.last_frame_time = current_time
        self.frame_count += 1
        
        metadata = {
            "frame_id": self.frame_count,
            "timestamp": current_time,
            "width": self.frame_width,
            "height": self.frame_height,
            "resolution_mode": self.resolution_mode,
            "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
        }
        
        return frame, metadata

    def frame_to_arrow(self, frame: np.ndarray, metadata: dict) -> tuple[pa.Buffer, bytes]:
        """Serialize frame to Apache Arrow format for zero-copy transfer."""
        frame_bytes = frame.tobytes()
        
        schema = pa.schema([
            ("frame_data", pa.binary()),
            ("frame_id", pa.int64()),
            ("timestamp", pa.float64()),
            ("width", pa.int32()),
            ("height", pa.int32()),
            ("channels", pa.int32()),
            ("resolution_mode", pa.string()),
        ])
        
        table = pa.table({
            "frame_data": [frame_bytes],
            "frame_id": [metadata["frame_id"]],
            "timestamp": [metadata["timestamp"]],
            "width": [metadata["width"]],
            "height": [metadata["height"]],
            "channels": [metadata["channels"]],
            "resolution_mode": [metadata["resolution_mode"]],
        }, schema=schema)
        
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, schema) as writer:
            writer.write_table(table)
        
        arrow_buffer = sink.getvalue()
        metadata_bytes = pa.serialize(metadata).to_buffer()
        
        return arrow_buffer, metadata_bytes.to_pybytes()

    def close(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Camera operator closed")


_operator = None


def on_input(
    dora_input: dict[str, Any],
    send_output: Any,
    dora_event: Any,
) -> None:
    """dora-rs input handler for camera operator."""
    global _operator
    
    if _operator is None:
        _operator = CameraOperator()
    
    input_id = dora_input.get("id", "")
    
    if input_id == "resolution_change":
        mode = dora_input.get("value", b"low").decode("utf-8")
        _operator.set_resolution(mode)
        return
    
    frame, metadata = _operator.capture_frame()
    
    if frame is None:
        return
    
    arrow_buffer, metadata_bytes = _operator.frame_to_arrow(frame, metadata)
    
    send_output("frame", arrow_buffer.to_pybytes(), dora_input.get("metadata", {}))
    send_output("metadata", metadata_bytes, dora_input.get("metadata", {}))


def on_stop() -> None:
    """Cleanup handler for dora-rs."""
    global _operator
    if _operator is not None:
        _operator.close()
        _operator = None
