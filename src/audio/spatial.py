"""Spatial Audio Feedback for Synthetic Cortex.

Converts detection bounding boxes to binaural spatial audio cues:
- X-position (0-1) maps to stereo pan (-1 to +1)
- Detection confidence maps to volume
- Object type maps to distinct tone frequencies
"""

import io
import math
import logging
from typing import Any
from dataclasses import dataclass

import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

logger = logging.getLogger(__name__)


TONE_FREQUENCIES = {
    "vehicle": 800,
    "car": 800,
    "truck": 700,
    "bus": 750,
    "motorcycle": 900,
    "bicycle": 850,
    "person": 600,
    "animal": 500,
    "dog": 520,
    "cat": 480,
    "obstacle": 400,
    "furniture": 350,
    "door": 450,
    "stairs": 1000,
    "edge": 1200,
    "water": 300,
    "hazard": 1100,
    "construction": 950,
    "halt": 1500,
    "question": 550,
    "default": 440,
}


URGENCY_CONFIG = {
    "low": {"volume_db": -20, "duration_ms": 150, "pattern": [1]},
    "medium": {"volume_db": -12, "duration_ms": 200, "pattern": [1, 0, 1]},
    "high": {"volume_db": -6, "duration_ms": 250, "pattern": [1, 0, 1, 0, 1]},
    "critical": {"volume_db": 0, "duration_ms": 300, "pattern": [1, 1, 1, 1, 1]},
}


@dataclass
class AudioCue:
    """Represents a spatial audio cue."""
    
    position: float
    frequency: int
    volume_db: float
    duration_ms: int
    pattern: list[int]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "position": self.position,
            "frequency": self.frequency,
            "volume_db": self.volume_db,
            "duration_ms": self.duration_ms,
            "pattern": self.pattern,
        }


class SpatialAudioGenerator:
    """Generates binaural spatial audio from detection data."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._audio_queue: list[AudioSegment] = []
    
    def bbox_to_position(self, bbox: list[float]) -> float:
        """Convert bounding box to stereo position.
        
        Args:
            bbox: [x1, y1, x2, y2] normalized coordinates (0-1)
            
        Returns:
            Position from -1 (left) to +1 (right)
        """
        if len(bbox) < 4:
            return 0.0
        
        center_x = (bbox[0] + bbox[2]) / 2
        
        position = (center_x - 0.5) * 2
        
        return max(-1.0, min(1.0, position))
    
    def confidence_to_volume(
        self,
        confidence: float,
        min_volume: float = -24.0,
        max_volume: float = 0.0,
    ) -> float:
        """Convert confidence score to volume in dB.
        
        Args:
            confidence: Detection confidence (0-1)
            min_volume: Minimum volume in dB
            max_volume: Maximum volume in dB
            
        Returns:
            Volume in dB
        """
        confidence = max(0.0, min(1.0, confidence))
        return min_volume + (max_volume - min_volume) * confidence
    
    def get_frequency(self, object_type: str) -> int:
        """Get tone frequency for object type."""
        return TONE_FREQUENCIES.get(
            object_type.lower(),
            TONE_FREQUENCIES["default"],
        )
    
    def apply_stereo_pan(
        self,
        audio: AudioSegment,
        position: float,
    ) -> AudioSegment:
        """Apply stereo panning to audio segment.
        
        Uses constant-power panning for natural sound.
        
        Args:
            audio: Mono or stereo AudioSegment
            position: Position from -1 (left) to +1 (right)
            
        Returns:
            Stereo panned AudioSegment
        """
        if audio.channels == 1:
            audio = audio.set_channels(2)
        
        angle = (position + 1) * math.pi / 4
        
        left_gain = math.cos(angle)
        right_gain = math.sin(angle)
        
        left_db = 20 * math.log10(max(left_gain, 0.001))
        right_db = 20 * math.log10(max(right_gain, 0.001))
        
        left_channel = audio.split_to_mono()[0].apply_gain(left_db)
        right_channel = audio.split_to_mono()[1].apply_gain(right_db)
        
        return AudioSegment.from_mono_audiosegments(left_channel, right_channel)
    
    def generate_tone(
        self,
        frequency: int,
        duration_ms: int,
        volume_db: float = 0.0,
    ) -> AudioSegment:
        """Generate a pure sine tone.
        
        Args:
            frequency: Tone frequency in Hz
            duration_ms: Duration in milliseconds
            volume_db: Volume adjustment in dB
            
        Returns:
            AudioSegment containing the tone
        """
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
        
        fade_duration = min(20, duration_ms // 4)
        tone = tone.fade_in(fade_duration).fade_out(fade_duration)
        
        if volume_db != 0:
            tone = tone.apply_gain(volume_db)
        
        return tone
    
    def generate_pattern(
        self,
        frequency: int,
        duration_ms: int,
        volume_db: float,
        pattern: list[int],
    ) -> AudioSegment:
        """Generate audio pattern (e.g., beep-pause-beep).
        
        Args:
            frequency: Tone frequency in Hz
            duration_ms: Duration per beat in milliseconds
            volume_db: Volume adjustment in dB
            pattern: List of 1s (sound) and 0s (silence)
            
        Returns:
            AudioSegment containing the pattern
        """
        result = AudioSegment.empty()
        
        for beat in pattern:
            if beat:
                tone = self.generate_tone(frequency, duration_ms, volume_db)
                result += tone
            else:
                silence = AudioSegment.silent(duration=duration_ms // 2)
                result += silence
        
        return result
    
    def create_cue(
        self,
        object_type: str,
        bbox: list[float] | None = None,
        confidence: float = 1.0,
        urgency: str = "medium",
        position: float | None = None,
    ) -> AudioCue:
        """Create an audio cue from detection parameters.
        
        Args:
            object_type: Type of detected object
            bbox: Bounding box [x1, y1, x2, y2] (optional if position provided)
            confidence: Detection confidence (0-1)
            urgency: Urgency level (low/medium/high/critical)
            position: Override position (-1 to +1)
            
        Returns:
            AudioCue object
        """
        if position is None:
            position = self.bbox_to_position(bbox) if bbox else 0.0
        
        frequency = self.get_frequency(object_type)
        
        urgency_config = URGENCY_CONFIG.get(urgency, URGENCY_CONFIG["medium"])
        
        base_volume = urgency_config["volume_db"]
        confidence_adjustment = self.confidence_to_volume(confidence, -12, 0)
        final_volume = base_volume + confidence_adjustment
        
        return AudioCue(
            position=position,
            frequency=frequency,
            volume_db=final_volume,
            duration_ms=urgency_config["duration_ms"],
            pattern=urgency_config["pattern"],
        )
    
    def render_cue(self, cue: AudioCue) -> AudioSegment:
        """Render an AudioCue to an AudioSegment.
        
        Args:
            cue: AudioCue to render
            
        Returns:
            Stereo AudioSegment ready for playback
        """
        audio = self.generate_pattern(
            frequency=cue.frequency,
            duration_ms=cue.duration_ms,
            volume_db=cue.volume_db,
            pattern=cue.pattern,
        )
        
        audio = self.apply_stereo_pan(audio, cue.position)
        
        return audio
    
    def detection_to_audio(
        self,
        detection: dict[str, Any],
        urgency: str | None = None,
    ) -> AudioSegment:
        """Convert a detection result to spatial audio.
        
        Args:
            detection: Detection dictionary with object_type, bbox, confidence, risk_level
            urgency: Override urgency (otherwise derived from risk_level)
            
        Returns:
            Stereo AudioSegment for the detection
        """
        object_type = detection.get("object_type", "obstacle")
        bbox = detection.get("bbox", [0.5, 0.5, 0.5, 0.5])
        confidence = detection.get("confidence", 0.5)
        risk_level = detection.get("risk_level", "low")
        
        if urgency is None:
            urgency_map = {"low": "low", "medium": "medium", "high": "high"}
            urgency = urgency_map.get(risk_level, "medium")
        
        cue = self.create_cue(
            object_type=object_type,
            bbox=bbox,
            confidence=confidence,
            urgency=urgency,
        )
        
        return self.render_cue(cue)
    
    def detections_to_audio(
        self,
        detections: list[dict[str, Any]],
        max_simultaneous: int = 3,
    ) -> AudioSegment:
        """Convert multiple detections to combined spatial audio.
        
        Prioritizes high-risk detections and limits simultaneous sounds.
        
        Args:
            detections: List of detection dictionaries
            max_simultaneous: Maximum number of sounds to play
            
        Returns:
            Combined stereo AudioSegment
        """
        if not detections:
            return AudioSegment.silent(duration=100)
        
        sorted_detections = sorted(
            detections,
            key=lambda d: (
                {"high": 2, "medium": 1, "low": 0}.get(d.get("risk_level", "low"), 0),
                d.get("confidence", 0),
            ),
            reverse=True,
        )
        
        top_detections = sorted_detections[:max_simultaneous]
        
        max_duration = 0
        audio_segments = []
        
        for detection in top_detections:
            audio = self.detection_to_audio(detection)
            audio_segments.append(audio)
            max_duration = max(max_duration, len(audio))
        
        combined = AudioSegment.silent(duration=max_duration)
        
        for audio in audio_segments:
            combined = combined.overlay(audio)
        
        return combined
    
    def export_bytes(self, audio: AudioSegment, format: str = "wav") -> bytes:
        """Export AudioSegment to bytes.
        
        Args:
            audio: AudioSegment to export
            format: Output format (wav, mp3, etc.)
            
        Returns:
            Audio data as bytes
        """
        buffer = io.BytesIO()
        audio.export(buffer, format=format)
        return buffer.getvalue()


_generator: SpatialAudioGenerator | None = None


def get_audio_generator() -> SpatialAudioGenerator:
    """Get or create the global audio generator."""
    global _generator
    if _generator is None:
        _generator = SpatialAudioGenerator()
    return _generator


def generate_audio_for_detections(detections: list[dict]) -> bytes:
    """Convenience function to generate audio from detections."""
    generator = get_audio_generator()
    audio = generator.detections_to_audio(detections)
    return generator.export_bytes(audio)
