"""Transcription through the `speech` command line tool.

`speech` runs MOSS-Transcribe-Diarize on Apple Silicon through MLX, using
roughly 1.3 GB for ten minutes of audio. It prints a human-readable report
rather than JSON, so this module owns that format.

Install with `brew install speech`. The first run downloads quantized weights.
"""

import re
import shutil
import subprocess
from pathlib import Path

from .logging_utils import logger
from .meeting_audio import read_wav_header

# Two leading spaces, then [start-end][Sxx] text.
SEGMENT_PATTERN = re.compile(r"^\s*\[([\d.]+)s-([\d.]+)s\]\[(S\d+)\]\s*(.*)$")
# Bare [12.34] markers the model emits while repeating itself.
TIMESTAMP_MARKER = re.compile(r"\[\d+\.\d{2}\]")
MIN_MAX_TOKENS = 600
MAX_MAX_TOKENS = 12000


class SpeechCliError(RuntimeError):
    """The speech CLI is missing, failed, or returned something unparseable."""


def parse_segments(output: str) -> list[dict[str, object]]:
    """Extracts timestamped speaker segments from the CLI report."""
    segments: list[dict[str, object]] = []
    for line in output.splitlines():
        match = SEGMENT_PATTERN.match(line)
        if not match:
            continue
        start, end, speaker, text = match.groups()
        text = text.strip()
        if not text:
            continue
        segments.append(
            {
                "start": float(start),
                "end": float(end),
                "text": text,
                "speaker_raw": speaker,
            }
        )
    return segments


def binary(settings) -> str:
    configured = settings.speech_bin
    if configured:
        return str(configured)
    found = shutil.which("speech")
    if not found:
        raise SpeechCliError(
            "The `speech` command is not installed. Install it with `brew install speech`."
        )
    return found


def token_budget(duration_seconds: float) -> int:
    """Caps generated tokens for a window.

    Human speech needs a few tokens per second; twelve leaves a wide margin and
    still stops runaway repetition well before the CLI's own default.
    """
    return max(MIN_MAX_TOKENS, min(MAX_MAX_TOKENS, int(duration_seconds * 12)))


def transcribe(
    audio: Path,
    settings,
    language: str | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, object]]:
    """Transcribes one audio file through the speech CLI."""
    if max_tokens is None:
        duration = read_wav_header(audio)[1]
        max_tokens = token_budget(duration)
    cmd = [
        binary(settings),
        "transcribe",
        str(audio),
        "--engine",
        settings.speech_engine,
        "--backend",
        settings.speech_backend,
        "--model",
        settings.speech_model,
        "--max-tokens",
        str(max_tokens),
    ]
    if language:
        cmd += ["--language", language]
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.speech_timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SpeechCliError(f"The `speech` command could not run: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SpeechCliError(
            f"speech CLI exceeded {settings.speech_timeout_seconds:.0f}s on {audio.name}"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise SpeechCliError(
            f"speech CLI failed on {audio.name} (exit {result.returncode}): "
            f"{detail[-1] if detail else 'no output'}"
        )
    if "Result:" not in result.stdout:
        raise SpeechCliError(
            f"speech CLI printed no transcript report for {audio.name}; "
            "its output format may have changed."
        )
    segments = parse_segments(result.stdout)
    if not segments:
        # Non-speech audio has no valid transcript: the model either emits
        # nothing or repeats timestamp markers until the token cap. Both mean
        # "no words in this window", which is a normal outcome, not a failure.
        markers = len(TIMESTAMP_MARKER.findall(result.stdout))
        reason = "hit the token cap" if "maximumTokens" in result.stdout else "ended early"
        logger.info(
            "%s: no words in this window (%s, %d timestamp markers)", audio.name, reason, markers
        )
        return []
    logger.info(
        "%s: %d segments via speech (%s)", audio.name, len(segments), settings.speech_backend
    )
    return segments
