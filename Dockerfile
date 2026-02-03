FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libportaudio2 \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt


FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libportaudio2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash cortex

COPY --from=builder /install /usr/local

WORKDIR /app

COPY --chown=cortex:cortex src/ ./src/
COPY --chown=cortex:cortex configs/ ./configs/
COPY --chown=cortex:cortex dataflow.yaml .

RUN mkdir -p /app/logs && chown cortex:cortex /app/logs

USER cortex

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.langgraph.graph import get_cortex_state_machine; get_cortex_state_machine()" || exit 1

CMD ["python", "-m", "dora", "up", "--config", "dataflow.yaml"]
