FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python -m venv "${VIRTUAL_ENV}"

COPY requirements-docker.txt /tmp/requirements-docker.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements-docker.txt


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    XDG_CACHE_HOME=/home/transcribe/.cache \
    HF_HOME=/home/transcribe/.cache/huggingface \
    MPLCONFIGDIR=/home/transcribe/.cache/matplotlib \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --system transcribe && \
    useradd --system --create-home --gid transcribe transcribe && \
    mkdir -p /home/transcribe/.cache/huggingface /home/transcribe/.cache/matplotlib /home/transcribe/.cache/torch && \
    chown -R transcribe:transcribe /home/transcribe

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=transcribe:transcribe . /app

USER transcribe

ENTRYPOINT ["python", "transcribe_diarize.py"]
