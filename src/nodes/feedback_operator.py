"""Feedback Operator Node for dora-rs dataflow.

Receives JSON detection results and generates binaural spatial audio
to communicate hazards and navigation guidance to the user.
"""

import os
import json
import logging
from typing import Any

from dora import Node

from src.audio.spatial import SpatialAudioGenerator, get_audio_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available, audio playback disabled")


class FeedbackOperator:
    """Spatial audio feedback operator."""

    def __init__(self):
        self.audio_device = os.getenv("AUDIO_DEVICE", "default")
        self.sample_rate = int(os.getenv("SAMPLE_RATE", "44100"))
        self.enable_tts = os.getenv("ENABLE_TTS", "false").lower() == "true"
        
        self._audio_generator: SpatialAudioGenerator | None = None
        self._pyaudio: Any = None
        self._stream: Any = None
        self._halt_requested = False
        
        self._initialize()

    def _initialize(self) -> None:
        """Initialize audio components."""
        self._audio_generator = get_audio_generator()
        
        if PYAUDIO_AVAILABLE:
            try:
                self._pyaudio = pyaudio.PyAudio()
                self._stream = self._pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=2,
                    rate=self.sample_rate,
                    output=True,
                )
                logger.info("Audio output stream initialized")
            except Exception as e:
                logger.error(f"Failed to initialize audio: {e}")
                self._pyaudio = None
        else:
            logger.warning("Audio playback unavailable")

    def process_detections(self, detections_data: dict[str, Any]) -> bytes | None:
        """Process detections and generate spatial audio.
        
        Args:
            detections_data: Detection results from cortex
            
        Returns:
            Audio bytes or None if no audio generated
        """
        detections = detections_data.get("detections", [])
        confidence = detections_data.get("confidence", 0.0)
        
        if not detections:
            return None
        
        if self._audio_generator is None:
            return None
        
        audio_segment = self._audio_generator.detections_to_audio(detections)
        
        audio_bytes = self._audio_generator.export_bytes(audio_segment, format="wav")
        
        self._play_audio(audio_bytes)
        
        return audio_bytes

    def process_clarification(self, clarification_data: dict[str, Any]) -> bytes | None:
        """Process clarification request and generate audio prompt.
        
        Args:
            clarification_data: Clarification request from cortex
            
        Returns:
            Audio bytes or None
        """
        reason = clarification_data.get("reason", "Verification needed")
        confidence = clarification_data.get("confidence", 0.0)
        detections = clarification_data.get("detections", [])
        
        if self._audio_generator is None:
            return None
        
        position = 0.0
        if detections:
            primary = max(detections, key=lambda d: d.get("confidence", 0))
            bbox = primary.get("bbox", [0.5, 0.5, 0.5, 0.5])
            position = self._audio_generator.bbox_to_position(bbox)
        
        cue = self._audio_generator.create_cue(
            object_type="question",
            position=position,
            confidence=confidence,
            urgency="medium",
        )
        
        audio_segment = self._audio_generator.render_cue(cue)
        
        audio_bytes = self._audio_generator.export_bytes(audio_segment, format="wav")
        
        self._play_audio(audio_bytes)
        
        logger.info(f"Clarification audio played: {reason}")
        
        return audio_bytes

    def trigger_halt_audio(self, reason: str = "Emergency stop") -> bytes | None:
        """Generate and play emergency halt audio.
        
        Args:
            reason: Reason for halt
            
        Returns:
            Audio bytes or None
        """
        if self._audio_generator is None:
            return None
        
        cue = self._audio_generator.create_cue(
            object_type="halt",
            position=0.0,
            confidence=1.0,
            urgency="critical",
        )
        
        audio_segment = self._audio_generator.render_cue(cue)
        
        audio_bytes = self._audio_generator.export_bytes(audio_segment, format="wav")
        
        self._play_audio(audio_bytes)
        
        logger.warning(f"Halt audio triggered: {reason}")
        
        self._halt_requested = True
        
        return audio_bytes

    def _play_audio(self, audio_bytes: bytes) -> None:
        """Play audio through the output stream.
        
        Args:
            audio_bytes: WAV audio data
        """
        if self._stream is None or not PYAUDIO_AVAILABLE:
            return
        
        try:
            from pydub import AudioSegment
            import io
            
            audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
            
            audio = audio.set_frame_rate(self.sample_rate)
            audio = audio.set_channels(2)
            audio = audio.set_sample_width(2)
            
            raw_data = audio.raw_data
            
            self._stream.write(raw_data)
            
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")

    def request_halt(self) -> None:
        """Set halt request flag."""
        self._halt_requested = True

    def clear_halt_request(self) -> None:
        """Clear halt request flag."""
        self._halt_requested = False

    def is_halt_requested(self) -> bool:
        """Check if halt has been requested."""
        return self._halt_requested

    def close(self) -> None:
        """Release audio resources."""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None
        
        logger.info("Feedback operator closed")


_operator: FeedbackOperator | None = None


def on_input(
    dora_input: dict[str, Any],
    send_output: Any,
    dora_event: Any,
) -> None:
    """dora-rs input handler for feedback operator."""
    global _operator
    
    if _operator is None:
        _operator = FeedbackOperator()
    
    input_id = dora_input.get("id", "")
    
    if input_id == "detections":
        raw_data = dora_input.get("value", b"")
        
        if not raw_data:
            return
        
        try:
            detections_data = json.loads(raw_data.decode("utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse detections JSON: {e}")
            return
        
        audio_bytes = _operator.process_detections(detections_data)
        
        if audio_bytes:
            send_output("audio_cue", audio_bytes, dora_input.get("metadata", {}))
    
    elif input_id == "clarification_request":
        raw_data = dora_input.get("value", b"")
        
        if not raw_data:
            return
        
        try:
            clarification_data = json.loads(raw_data.decode("utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse clarification JSON: {e}")
            return
        
        audio_bytes = _operator.process_clarification(clarification_data)
        
        if audio_bytes:
            send_output("audio_cue", audio_bytes, dora_input.get("metadata", {}))
    
    if _operator.is_halt_requested():
        halt_signal = json.dumps({"halt": True, "source": "feedback"}).encode("utf-8")
        send_output("halt_request", halt_signal, dora_input.get("metadata", {}))
        _operator.clear_halt_request()


def on_stop() -> None:
    """Cleanup handler for dora-rs."""
    global _operator
    if _operator is not None:
        _operator.close()
        _operator = None
