"""Main entry point for Synthetic Cortex Vision System.

Runs the system without dora-rs CLI by directly managing the dataflow components.
"""

import os
import sys
import json
import time
import signal
import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cortex_graph.graph import get_cortex_state_machine
from src.audio.spatial import SpatialAudioGenerator, get_audio_generator
from src.gemini.client import GeminiClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/synthetic_cortex.log", mode="a"),
    ]
)
logger = logging.getLogger(__name__)


class SyntheticCortex:
    """Main orchestrator for the Synthetic Cortex Vision System."""

    def __init__(self):
        self.rtsp_url = os.getenv("RTSP_URL", "rtsp://192.168.1.100:5554/live")
        self.target_fps = int(os.getenv("TARGET_FPS", "2"))
        self.frame_width = int(os.getenv("FRAME_WIDTH", "320"))
        self.frame_height = int(os.getenv("FRAME_HEIGHT", "240"))
        self.cair_threshold = float(os.getenv("CAIR_THRESHOLD", "0.85"))
        
        self.frame_skip = int(os.getenv("FRAME_SKIP", "5"))
        self.min_api_interval = float(os.getenv("MIN_API_INTERVAL", "4.0"))
        self.motion_threshold = float(os.getenv("MOTION_THRESHOLD", "0.1"))
        self.enable_motion_detection = os.getenv("ENABLE_MOTION_DETECTION", "true").lower() == "true"

        self.cap = None
        self.state_machine = None
        self.audio_generator = None
        self.gemini_client = None

        self._running = False
        self._frame_count = 0
        self._frames_processed = 0
        self._last_processed_frame = None
        self._last_processed_time = 0.0
        self._lock = threading.Lock()
        self._api_backoff = 0.0
        self._consecutive_errors = 0
        self._api_calls_today = 0
        self._api_calls_reset_time = time.time() + 86400

        logger.info("Synthetic Cortex initializing...")
        logger.info(f"API optimization: FPS={self.target_fps}, Skip={self.frame_skip}, MinInterval={self.min_api_interval}s")

    def _init_camera(self) -> bool:
        """Initialize RTSP camera connection."""
        logger.info(f"Connecting to RTSP stream: {self.rtsp_url}")

        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        if not self.cap.isOpened():
            logger.error(f"Failed to connect to RTSP stream: {self.rtsp_url}")
            return False

        logger.info("Camera connected successfully")
        return True

    def _init_components(self) -> bool:
        """Initialize all system components."""
        try:
            self.state_machine = get_cortex_state_machine()
            logger.info("State machine initialized")

            self.audio_generator = get_audio_generator()
            logger.info("Audio generator initialized")

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not set")
                return False

            self.gemini_client = GeminiClient(api_key=api_key)
            logger.info("Gemini client initialized")

            return True

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False

    def _capture_frame(self) -> tuple[np.ndarray | None, dict]:
        """Capture a single frame from the camera."""
        if self.cap is None or not self.cap.isOpened():
            return None, {}

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Frame capture failed")
            return None, {}

        if frame.shape[:2] != (self.frame_height, self.frame_width):
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

        self._frame_count += 1
        metadata = {
            "frame_id": self._frame_count,
            "timestamp": time.time(),
            "width": self.frame_width,
            "height": self.frame_height,
            "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
            "resolution_mode": "low",
        }

        return frame, metadata

    def _frame_to_bytes(self, frame: np.ndarray) -> bytes:
        """Convert frame to bytes for processing."""
        _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return encoded.tobytes()
    
    def _has_significant_motion(self, frame: np.ndarray) -> bool:
        """Check if frame has significant motion compared to last processed frame."""
        if not self.enable_motion_detection or self._last_processed_frame is None:
            return True
        
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_last = cv2.cvtColor(self._last_processed_frame, cv2.COLOR_BGR2GRAY)
        
        gray_current = cv2.resize(gray_current, (160, 120))
        gray_last = cv2.resize(gray_last, (160, 120))
        
        diff = cv2.absdiff(gray_current, gray_last)
        motion_ratio = np.sum(diff > 30) / (160 * 120)
        
        return motion_ratio > self.motion_threshold
    
    def _should_process_frame(self) -> bool:
        """Determine if frame should be sent to API."""
        current_time = time.time()
        
        if self._api_backoff > 0 and current_time < self._api_backoff:
            return False
        
        if current_time - self._last_processed_time < self.min_api_interval:
            return False
        
        if self._frame_count % self.frame_skip != 0:
            return False
        
        if self._api_calls_today >= 1400:
            if current_time < self._api_calls_reset_time:
                logger.warning(f"Daily API limit reached ({self._api_calls_today}/1500). Waiting for reset.")
                return False
            else:
                self._api_calls_today = 0
                self._api_calls_reset_time = current_time + 86400
        
        return True

    def _process_detections(self, result: dict[str, Any]) -> None:
        """Process detection results and trigger audio feedback."""
        detections = result.get("detections", [])
        confidence = result.get("confidence", 0.0)

        if not detections:
            return

        if self.audio_generator and detections:
            audio_data = self.audio_generator.detections_to_audio(detections)
            logger.debug(f"Generated audio for {len(detections)} detections")

        if confidence < self.cair_threshold:
            logger.warning(f"Low confidence detection: {confidence:.2f}")

    def _main_loop(self) -> None:
        """Main processing loop with aggressive rate limiting and frame skipping."""
        frame_interval = 1.0 / self.target_fps
        last_frame_time = 0.0

        while self._running:
            current_time = time.time()

            if current_time - last_frame_time < frame_interval:
                time.sleep(0.01)
                continue

            frame, metadata = self._capture_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            last_frame_time = current_time

            if not self._should_process_frame():
                continue

            if not self._has_significant_motion(frame):
                continue

            try:
                frame_bytes = self._frame_to_bytes(frame)
                self._api_calls_today += 1

                result = self.state_machine.process_frame(
                    frame_data=frame_bytes,
                    frame_metadata=metadata,
                    thread_id=f"frame_{self._frame_count}",
                )

                if result.get("error") and "429" in str(result.get("error", "")):
                    self._consecutive_errors += 1
                    backoff_time = min(120, 10 * (2 ** self._consecutive_errors))
                    self._api_backoff = time.time() + backoff_time
                    logger.warning(f"API rate limited, backing off for {backoff_time}s")
                    self._api_calls_today -= 1
                    continue

                self._consecutive_errors = 0
                self._last_processed_frame = frame.copy()
                self._last_processed_time = current_time
                self._frames_processed += 1

                self._process_detections(result)

                if self._frames_processed % 10 == 0:
                    logger.info(f"Frames captured: {self._frame_count}, Processed: {self._frames_processed}, API calls today: {self._api_calls_today}")

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    self._consecutive_errors += 1
                    backoff_time = min(120, 10 * (2 ** self._consecutive_errors))
                    self._api_backoff = time.time() + backoff_time
                    logger.warning(f"API quota exceeded, backing off for {backoff_time}s")
                    self._api_calls_today -= 1
                else:
                    logger.error(f"Frame processing error: {e}")

    def start(self) -> None:
        """Start the Synthetic Cortex system."""
        logger.info("Starting Synthetic Cortex Vision System")

        if not self._init_camera():
            logger.error("Camera initialization failed")
            return

        if not self._init_components():
            logger.error("Component initialization failed")
            return

        self._running = True

        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("System started - processing frames")
        self._main_loop()

    def stop(self) -> None:
        """Stop the system gracefully."""
        logger.info("Stopping Synthetic Cortex")
        self._running = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        logger.info("System stopped")


def main():
    """Entry point."""
    Path("logs").mkdir(exist_ok=True)
    cortex = SyntheticCortex()
    cortex.start()


if __name__ == "__main__":
    main()
