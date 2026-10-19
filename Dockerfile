FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY transcribe_diarize_app/ ./transcribe_diarize_app/

RUN uv sync --frozen --no-dev


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --system meeting && \
    useradd --system --create-home --gid meeting meeting

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --chown=meeting:meeting pyproject.toml uv.lock ./
COPY --chown=meeting:meeting transcribe_diarize_app/ ./transcribe_diarize_app/

USER meeting

ENTRYPOINT ["meeting"]
