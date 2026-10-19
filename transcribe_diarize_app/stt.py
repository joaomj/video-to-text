"""Speech-to-text through an OpenAI-compatible transcription endpoint.

The default endpoint is OpenRouter (`POST {base}/audio/transcriptions`), which
accepts the same Bearer key as chat completions. Point `stt_base_url` at a local
server (whisper.cpp with `--inference-path /v1/audio/transcriptions`, or
llama.cpp server) and the same code sends an OpenAI-style multipart upload
instead of OpenRouter's base64 JSON, so cloud and local models share one path.

Long recordings are transcribed in one request first. OpenRouter limits each
request to about 60 seconds of processing time; only when a request times out or
is rejected for size does the client fall back to chunking. Chunking can reset
speaker numbering, so longer single requests keep "Remote N" labels consistent.
"""

import base64
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .logging_utils import logger
from .meeting_audio import read_wav_header
from .settings import Settings

MAX_ATTEMPTS = 3
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
REQUEST_TIMEOUT_SECONDS = 120.0
# A single request is tried first up to this duration; longer audio is chunked only
# when the single request fails. Ten minutes of 16 kHz mono WAV is ~19 MB.
SINGLE_REQUEST_MAX_SECONDS = 600.0
CHUNK_SECONDS = 600.0
MAI_MODEL_PREFIX = "microsoft/mai-transcribe"


class MissingApiKeyError(RuntimeError):
    pass


class RetryableTranscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PartRequest:
    audio_wav: Path
    offset: float
    duration: float
    language: str
    model: str
    base_url: str
    api_key: str
    diarize: bool


def transcribe_track(
    audio_wav: Path,
    language: str,
    model: str,
    settings: Settings,
    diarize: bool,
) -> list[dict[str, Any]]:
    """Transcribes a 16 kHz mono WAV file. Segments carry start/end/text/speaker_raw."""
    api_key = settings.resolved_stt_api_key()
    if api_key is None:
        raise MissingApiKeyError(
            "No OPENROUTER_API_KEY (or STT_API_KEY) is configured, so nothing was transcribed."
        )
    base_url = settings.stt_base_url.rstrip("/")
    secret = api_key.get_secret_value()
    _, duration = read_wav_header(audio_wav)
    if duration <= SINGLE_REQUEST_MAX_SECONDS:
        try:
            return _transcribe_part(
                PartRequest(audio_wav, 0.0, duration, language, model, base_url, secret, diarize)
            )
        except (RetryableTranscriptionError, httpx.TransportError) as exc:
            if duration <= CHUNK_SECONDS:
                raise
            logger.warning("Single transcription request failed (%s); falling back to chunks", exc)
    return _transcribe_chunked(audio_wav, duration, language, model, base_url, secret, diarize)


def _transcribe_chunked(
    audio_wav: Path,
    duration: float,
    language: str,
    model: str,
    base_url: str,
    api_key: str,
    diarize: bool,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    offset = 0.0
    while offset < duration:
        part = _transcribe_part(
            PartRequest(
                audio_wav,
                offset,
                min(CHUNK_SECONDS, duration - offset),
                language,
                model,
                base_url,
                api_key,
                diarize,
            )
        )
        segments.extend(part)
        offset += CHUNK_SECONDS
    return segments


def _transcribe_part(request: PartRequest) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        chunk = Path(tmp) / "chunk.mp3"
        _export_mp3(request.audio_wav, chunk, request.offset, request.duration)
        if "openrouter.ai" in request.base_url:
            data = _post_openrouter(chunk, request)
        else:
            data = _post_multipart(chunk, request)
    cost = data.get("usage", {}).get("cost")
    if cost is not None:
        logger.info(
            "Transcribed %.0fs with %s (cost $%.4f)",
            request.duration,
            request.model,
            cost,
        )
    raw_segments = data.get("segments")
    if not raw_segments:
        text = (data.get("text") or "").strip()
        if not text:
            return []
        raise RuntimeError(
            "The transcription server returned text without timestamps. "
            "Use a server with verbose_json segment support (e.g. whisper.cpp server)."
        )
    segments = []
    for raw in raw_segments:
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        segment: dict[str, Any] = {
            "start": float(raw["start"]) + request.offset,
            "end": float(raw["end"]) + request.offset,
            "text": text,
        }
        if raw.get("speaker") is not None:
            segment["speaker_raw"] = str(raw["speaker"])
        segments.append(segment)
    return segments


def _export_mp3(source: Path, destination: Path, offset: float, duration: float) -> None:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{offset:.2f}",
        "-t",
        f"{duration:.2f}",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        str(destination),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Install it with `brew install ffmpeg`.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to export audio chunk: {exc.stderr.strip()}") from exc


def _post_openrouter(chunk: Path, request: PartRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model,
        "input_audio": {
            "data": base64.b64encode(chunk.read_bytes()).decode(),
            "format": "mp3",
        },
        "language": request.language,
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
    }
    if request.diarize and request.model.startswith(MAI_MODEL_PREFIX):
        payload["provider"] = {"options": {"azure": {"diarization": {"enabled": True}}}}
    return _post_json(f"{request.base_url}/audio/transcriptions", payload, request.api_key)


def _post_multipart(chunk: Path, request: PartRequest) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with chunk.open("rb") as handle:
                response = httpx.post(
                    f"{request.base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {request.api_key}"},
                    files={"file": (chunk.name, handle, "audio/mpeg")},
                    data={
                        "model": request.model,
                        "language": request.language,
                        "response_format": "verbose_json",
                    },
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            if response.status_code in RETRYABLE_STATUS:
                raise RetryableTranscriptionError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )
            response.raise_for_status()
            return response.json()
        except (httpx.TransportError, RetryableTranscriptionError) as exc:
            last_error = exc
            logger.warning("Transcription attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, exc)
            if attempt < MAX_ATTEMPTS:
                time.sleep(2**attempt)
    raise RuntimeError(f"Transcription failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _post_json(url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = httpx.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code in RETRYABLE_STATUS:
                raise RetryableTranscriptionError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )
            response.raise_for_status()
            return response.json()
        except (httpx.TransportError, RetryableTranscriptionError) as exc:
            last_error = exc
            logger.warning("Transcription attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, exc)
            if attempt < MAX_ATTEMPTS:
                time.sleep(2**attempt)
    raise RuntimeError(f"Transcription failed after {MAX_ATTEMPTS} attempts: {last_error}")
