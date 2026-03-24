import importlib
import platform
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol, TypedDict

from .settings import Settings


class WordTimestamp(TypedDict):
    start: float
    end: float
    word: str


class WhisperSegment(TypedDict, total=False):
    start: float
    end: float
    text: str
    words: list[WordTimestamp]


class TranscriptionResult(TypedDict):
    segments: list[WhisperSegment]


@dataclass(frozen=True)
class TranscriptionRequest:
    audio_path: Path
    language: str
    initial_prompt: str


class Transcriber(Protocol):
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Transcribe audio into Whisper-style segments."""


class MlxTranscriber:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        try:
            mlx_whisper = importlib.import_module("mlx_whisper")
        except ImportError as exc:
            raise RuntimeError(
                "mlx-whisper is not installed. Use `--backend faster` "
                "or install native dependencies."
            ) from exc

        result = mlx_whisper.transcribe(
            str(request.audio_path),
            path_or_hf_repo=self._resolve_model_repo(),
            word_timestamps=True,
            language=request.language,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.0,
            logprob_threshold=-0.8,
            no_speech_threshold=0.5,
            hallucination_silence_threshold=1.0,
            initial_prompt=request.initial_prompt,
            verbose=True,
        )
        return {"segments": normalize_segments(result.get("segments", []))}

    def _resolve_model_repo(self) -> str:
        if self._settings.whisper_model == "large-v3":
            return self._settings.mlx_model_repo
        if self._settings.whisper_model.endswith("-mlx") or "/" in self._settings.whisper_model:
            return self._settings.whisper_model
        raise RuntimeError(
            "MLX backend currently supports WHISPER_MODEL=large-v3 or an explicit MLX model repo."
        )


class FasterWhisperTranscriber:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        try:
            whisper_module = importlib.import_module("faster_whisper")
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Add it with `pdm add faster-whisper`."
            ) from exc
        whisper_model_type = whisper_module.WhisperModel

        model = whisper_model_type(
            self._settings.whisper_model,
            device=self._settings.faster_whisper_device,
            compute_type=self._settings.faster_whisper_compute_type,
        )
        segments, _ = model.transcribe(
            str(request.audio_path),
            language=request.language,
            word_timestamps=True,
            initial_prompt=request.initial_prompt,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.0,
            log_prob_threshold=-0.8,
            no_speech_threshold=0.5,
            beam_size=5,
            vad_filter=True,
        )
        normalized_segments: list[WhisperSegment] = []
        for segment in segments:
            item: WhisperSegment = {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": str(segment.text).strip(),
            }
            words = getattr(segment, "words", None)
            if words:
                item["words"] = [
                    {
                        "start": float(word.start),
                        "end": float(word.end),
                        "word": str(word.word),
                    }
                    for word in words
                    if word.start is not None and word.end is not None
                ]
            normalized_segments.append(item)
        return {"segments": normalized_segments}


def create_transcriber(
    settings: Settings,
    requested_backend: str | None = None,
) -> tuple[str, Transcriber]:
    backend = requested_backend or settings.transcription_backend
    if backend == "auto":
        if should_use_mlx() and find_spec("mlx_whisper") is not None:
            return "mlx", MlxTranscriber(settings)
        return "faster", FasterWhisperTranscriber(settings)
    if backend == "mlx":
        return "mlx", MlxTranscriber(settings)
    if backend == "faster":
        return "faster", FasterWhisperTranscriber(settings)
    raise ValueError(f"Unsupported transcription backend: {backend}")


def should_use_mlx() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def normalize_segments(raw_segments: object) -> list[WhisperSegment]:
    if not isinstance(raw_segments, list):
        return []
    normalized: list[WhisperSegment] = []
    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        start = segment.get("start")
        end = segment.get("end")
        text = segment.get("text")
        if start is None or end is None or text is None:
            continue
        item: WhisperSegment = {
            "start": float(start),
            "end": float(end),
            "text": str(text).strip(),
        }
        words = segment.get("words")
        if isinstance(words, list):
            item["words"] = normalize_words(words)
        normalized.append(item)
    return normalized


def normalize_words(raw_words: list[object]) -> list[WordTimestamp]:
    normalized: list[WordTimestamp] = []
    for word in raw_words:
        if not isinstance(word, dict):
            continue
        start = word.get("start")
        end = word.get("end")
        value = word.get("word")
        if start is None or end is None or value is None:
            continue
        normalized.append(
            {
                "start": float(start),
                "end": float(end),
                "word": str(value),
            }
        )
    return normalized
