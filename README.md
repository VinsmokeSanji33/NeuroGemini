# Synthetic Cortex Vision System

A CAIR-aware vision system for sensory substitution using Gemini 3 Pro/Flash, LangGraph, and dora-rs middleware.

## Overview

Synthetic Cortex is a multi-agent vision system designed for visually impaired users. It processes live RTSP camera feeds from a mobile phone and provides real-time spatial audio feedback about the environment.

Key features:
- Dual-model routing (Flash for reflex, Pro for cerebral processing)
- CAIR (Confidence in AI Results) metric for safety-aware decisions
- Thought signature management for object permanence
- Zero-copy frame transfer using Apache Arrow
- MCP server for agent tool access

## Setup

### Prerequisites

- Docker and Docker Compose
- Gemini API key from ai.google.dev
- Mobile phone with RTSP streaming app (e.g., IP Webcam)

### Environment Configuration

Copy the example environment file and configure:

```bash
cp env.example .env
```

Required variables:
```
GEMINI_API_KEY=your_api_key_here
RTSP_URL=rtsp://your_phone_ip:5554/live
```

Optional variables:
```
TARGET_FPS=10
FRAME_WIDTH=640
FRAME_HEIGHT=480
CAIR_THRESHOLD=0.85
LANGSMITH_API_KEY=your_langsmith_key
```

## Deployment

Start the system:

```bash
docker compose up -d
```

View logs:

```bash
docker compose logs -f synthetic-cortex
```

Stop the system:

```bash
docker compose down
```

## Configuration

### CAIR Thresholds

Edit `configs/cair_thresholds.yaml` to adjust:

- Confidence thresholds for clarification requests
- Risk object classifications
- Model routing parameters
- Audio feedback frequencies

### Model Routing

The system automatically routes between models based on risk:

| Layer | Model | Thinking Level | Resolution | Use Case |
|-------|-------|----------------|------------|----------|
| Reflex | gemini-2.0-flash | minimal | low | Navigation, low-risk |
| Cerebral | gemini-2.0-pro | high | high | Hazards, high-risk |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         dora-rs Dataflow                        │
├─────────────────┬─────────────────────┬─────────────────────────┤
│  Camera Node    │    Cortex Node      │    Feedback Node        │
│  (RTSP+Arrow)   │  (LangGraph+Gemini) │   (Spatial Audio)       │
└────────┬────────┴──────────┬──────────┴────────────┬────────────┘
         │                   │                       │
         │ Arrow Frame       │ JSON Detections       │ Binaural Audio
         └───────────────────┴───────────────────────┘
```

### State Machine Flow

1. RiskAssessment: Evaluate environmental volatility
2. Reflex/Cerebral Layer: Process frame with appropriate model
3. CAIR Check: Validate confidence meets threshold
4. Ask Clarification: Request user input if confidence low
5. Emit Detections: Send results to audio feedback

## MCP Server Tools

The MCP server exposes tools for agent interaction:

| Tool | Description |
|------|-------------|
| toggle_resolution | Switch camera between low/high modes |
| spatial_audio_trigger | Play directional audio cue |
| emergency_halt | Immediate system stop |
| ask_user_clarification | Audio prompt for verification |

## Maintenance

### Logs

Logs are stored in `./logs/` and include:
- Frame processing events
- CAIR metric calculations
- Model routing decisions
- Audio feedback triggers

### LangSmith Tracing

If LANGSMITH_API_KEY is configured, all decisions are traced to LangSmith for observability. View traces at smith.langchain.com.

### Health Checks

The container includes health checks that verify:
- LangGraph state machine initialization
- MCP server availability
- Audio subsystem status

### Troubleshooting

RTSP connection failures:
```bash
# Test RTSP stream
ffprobe rtsp://your_phone_ip:5554/live
```

Audio issues:
```bash
# Check audio devices in container
docker compose exec synthetic-cortex aplay -l
```

Memory issues:
```bash
# Adjust Arrow buffer size in dataflow.yaml
communication:
  buffer_size_mb: 64  # Reduce from 128
```
