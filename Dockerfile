FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-group macos --no-dev --no-install-project

COPY transcribe_diarize_app/ ./transcribe_diarize_app/
COPY transcribe_diarize.py ./

RUN uv sync --frozen --no-group macos --no-dev


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    XDG_CACHE_HOME=/home/transcribe/.cache \
    HF_HOME=/home/transcribe/.cache/huggingface \
    MPLCONFIGDIR=/home/transcribe/.cache/matplotlib

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --system transcribe && \
    useradd --system --create-home --gid transcribe transcribe && \
    mkdir -p /home/transcribe/.cache/huggingface \
             /home/transcribe/.cache/matplotlib \
             /home/transcribe/.cache/torch && \
    chown -R transcribe:transcribe /home/transcribe

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --chown=transcribe:transcribe . /app

USER transcribe

ENTRYPOINT ["transcribe"]
