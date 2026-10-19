"""Local meeting transcription with MOSS-Transcribe-Diarize.

Runs fully offline on this machine: one 0.9B model returns timestamped
segments with speaker labels ([S01], [S02]) in a single pass, so no cloud
transcription and no separate diarization model are needed.

Reusing this file in another project takes two steps: copy it, and add these
dependencies (torch plus the MOSS helpers, which live only on GitHub):

    torch>=2.8
    moss-transcribe-diarize @ git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize

Only `transcribe_track` and `select_device` are the public surface; everything
else is an implementation detail of this file.
"""

import logging
from pathlib import Path
from typing import Any

import torch
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
)
from transformers import AutoModelForCausalLM, AutoProcessor

from .logging_utils import logger
from .meeting_audio import read_wav_header
from .settings import Settings

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
# Roughly 130 spoken words per minute plus timestamp markup; scaled by duration
# so long meetings are transcribed in one pass (the model supports 90 minutes).
TOKENS_PER_AUDIO_SECOND = 8
MIN_NEW_TOKENS = 2048

_cache: dict[str, Any] = {}


def select_device(preference: str | None = None) -> str:
    """Picks the fastest available device: explicit preference, else CUDA, Apple GPU, CPU."""
    if preference and preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe_track(audio_wav: Path, settings: Settings) -> list[dict[str, Any]]:
    """Transcribes a 16 kHz mono WAV file. Segments carry start/end/text/speaker_raw."""
    model, processor, device_name = _load(settings)
    device = torch.device(device_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    messages = build_transcription_messages(str(audio_wav))
    _, duration = read_wav_header(audio_wav)
    max_tokens = max(MIN_NEW_TOKENS, int(duration * TOKENS_PER_AUDIO_SECOND))
    logging.getLogger("transformers").setLevel(logging.WARNING)
    result = generate_transcription(
        model,
        processor,
        messages,
        max_new_tokens=max_tokens,
        do_sample=False,
        device=device,
        dtype=dtype,
    )
    segments = []
    for parsed in parse_transcript(result["text"]):
        text = parsed.text.strip()
        if not text:
            continue
        segments.append(
            {
                "start": float(parsed.start),
                "end": float(parsed.end),
                "text": text,
                "speaker_raw": str(parsed.speaker),
            }
        )
    logger.info(
        "Transcribed %.0fs locally with %s on %s (%d segments)",
        duration,
        settings.moss_model_id,
        device_name,
        len(segments),
    )
    return segments


def _load(settings: Settings) -> tuple[Any, Any, str]:
    """Loads the model once per process and reuses it for every track."""
    if _cache.get("model_id") != settings.moss_model_id:
        device_name = select_device(settings.moss_device)
        _cache["model"] = AutoModelForCausalLM.from_pretrained(
            settings.moss_model_id,
            revision=settings.moss_revision,
            trust_remote_code=True,
            dtype="auto",
            attn_implementation="sdpa",
        )
        # Apple GPUs and CPUs run float32; NVIDIA runs bfloat16.
        compute_dtype = torch.bfloat16 if device_name == "cuda" else torch.float32
        _cache["model"] = _cache["model"].to(dtype=compute_dtype).to(device_name).eval()
        _cache["processor"] = AutoProcessor.from_pretrained(
            settings.moss_model_id,
            revision=settings.moss_revision,
            trust_remote_code=True,
        )
        _cache["model_id"] = settings.moss_model_id
        _cache["device"] = device_name
        logger.info("Loaded %s on %s", settings.moss_model_id, device_name)
    return _cache["model"], _cache["processor"], _cache["device"]
