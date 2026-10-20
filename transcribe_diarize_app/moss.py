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
import time
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
# Instruction sent to the transcriber with every track. English is the
# default: the instruction language does not set the transcript language
# (the default Chinese prompt already returned English transcripts), it
# only tells the model which format to produce.
TRANSCRIPTION_PROMPT = (
    "Transcribe the audio to text. Each segment must start with a start "
    "timestamp and speaker label ([S01], [S02], [S03], ...), followed by the "
    "spoken content, and end with an end timestamp marking the speech range."
)
# Roughly 130 spoken words per minute plus timestamp markup; scaled by duration
# so long meetings are transcribed in one pass (the model supports 90 minutes).
TOKENS_PER_AUDIO_SECOND = 8
MIN_NEW_TOKENS = 2048
# A 40-minute track generates close to 20k tokens; report every 512 so a long
# pass shows progress instead of a silent gap in the log.
PROGRESS_TOKEN_STEP = 512

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
    dtype = model.dtype

    messages = build_transcription_messages(str(audio_wav), TRANSCRIPTION_PROMPT)
    _, duration = read_wav_header(audio_wav)
    max_tokens = max(MIN_NEW_TOKENS, int(duration * TOKENS_PER_AUDIO_SECOND))
    logging.getLogger("transformers").setLevel(logging.WARNING)

    started = time.monotonic()
    last_logged = 0

    def report_progress(generated: int) -> None:
        """Keeps a long single pass observable instead of silent for minutes."""
        nonlocal last_logged
        if generated - last_logged < PROGRESS_TOKEN_STEP:
            return
        last_logged = generated
        elapsed = max(time.monotonic() - started, 1e-6)
        logger.info(
            "Transcribing %s: %d of ~%d tokens, %.0fs elapsed, %.1f tokens/s",
            audio_wav.name,
            generated,
            max_tokens,
            elapsed,
            generated / elapsed,
        )

    result = generate_transcription(
        model,
        processor,
        messages,
        max_new_tokens=max_tokens,
        do_sample=False,
        device=device,
        dtype=dtype,
        token_callback=report_progress,
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
        settings.moss_cache_dir.mkdir(parents=True, exist_ok=True)
        _cache["model"] = AutoModelForCausalLM.from_pretrained(
            settings.moss_model_id,
            revision=settings.moss_revision,
            trust_remote_code=True,
            dtype="auto",
            attn_implementation="sdpa",
            cache_dir=str(settings.moss_cache_dir),
        )
        # Half precision halves weights and cache: on a 16 GB Mac the float32
        # copy of a long meeting is what pushes the machine into swap. CPUs have
        # no fast float16 path, so they keep float32.
        compute_dtype = {
            "cuda": torch.bfloat16,
            "mps": torch.float16,
        }.get(device_name, torch.float32)
        _cache["model"] = _cache["model"].to(dtype=compute_dtype).to(device_name).eval()
        _cache["processor"] = AutoProcessor.from_pretrained(
            settings.moss_model_id,
            revision=settings.moss_revision,
            trust_remote_code=True,
            cache_dir=str(settings.moss_cache_dir),
        )
        _cache["model_id"] = settings.moss_model_id
        _cache["device"] = device_name
        logger.info("Loaded %s on %s (%s)", settings.moss_model_id, device_name, compute_dtype)
    return _cache["model"], _cache["processor"], _cache["device"]
