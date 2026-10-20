"""Audio preparation and signal evidence for recorded meeting tracks."""

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16_000
# A one-second block counts as speech-like activity when it is this far above the
# track's own noise floor (10th percentile of block levels) and above an absolute floor.
# The adaptive threshold is capped: background louder than the cap is content, not
# silence, so a loud room or compressed meeting audio cannot hide speech from detection.
ACTIVITY_ABOVE_FLOOR_DB = 15.0
ACTIVITY_MIN_DB = -55.0
ACTIVITY_CAP_DB = -35.0
# Bump when measurement changes so cached stage results are recomputed.
MEASURE_VERSION = 1
NOISE_FLOOR_PERCENTILE = 10
# Transcription models expect speech near full scale. Recordings from a distant
# microphone often peak 20 dB lower, so the level is lifted to a fixed target
# instead of being handed to the model as captured.
TARGET_PEAK_DB = -1.0
MAX_GAIN_DB = 24.0
# A track this quiet holds no speech worth lifting, only the dither of silence.
QUIET_PEAK_DB = -60.0


@dataclass(frozen=True)
class TrackEvidence:
    duration_seconds: float
    active_seconds: int
    noise_floor_db: float
    peak_db: float


def read_wav_header(path: Path) -> tuple[int, float]:
    """Returns (sample_rate, duration_seconds) and fails on a missing or corrupt file."""
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getframerate(), audio.getnframes() / audio.getframerate()
    except (wave.Error, EOFError) as exc:
        raise RuntimeError(f"{path.name} is not a readable WAV file: {exc}") from exc


def normalize(source: Path, destination: Path) -> None:
    """Writes a 16 kHz mono copy for the models. The source recording is never modified."""
    temporary = destination.with_name(f".{destination.name}.tmp.wav")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        str(temporary),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Install it with `brew install ffmpeg`.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed on {source.name}: {exc.stderr.strip()}") from exc
    temporary.replace(destination)


def apply_gain(path: Path, gain_db: float) -> None:
    """Raises a 16 kHz mono WAV in place with a static ffmpeg gain."""
    temporary = path.with_name(f".{path.name}.tmp.wav")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-af",
        f"volume={gain_db:.1f}dB",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        str(temporary),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Install it with `brew install ffmpeg`.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed to raise the level of {path.name}: {exc.stderr.strip()}"
        ) from exc
    temporary.replace(path)


def lift_level(path: Path, evidence: TrackEvidence) -> TrackEvidence:
    """Brings a quiet track up to the transcription target, then re-measures it."""
    if evidence.peak_db < QUIET_PEAK_DB:
        return evidence
    gain = min(TARGET_PEAK_DB - evidence.peak_db, MAX_GAIN_DB)
    if gain < 1.0:
        return evidence
    apply_gain(path, gain)
    return measure(path)


def measure(path: Path) -> TrackEvidence:
    """Measures per-second levels of a 16-bit mono WAV file."""
    with wave.open(str(path), "rb") as audio:
        rate = audio.getframerate()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    duration = len(samples) / rate
    if len(samples) < rate:
        return TrackEvidence(duration, 0, -120.0, -120.0)
    blocks = samples[: len(samples) // rate * rate].astype(np.float32).reshape(-1, rate) / 32768
    rms = np.sqrt(np.mean(blocks**2, axis=1))
    levels = 20 * np.log10(np.maximum(rms, 1e-6))
    floor = float(np.percentile(levels, NOISE_FLOOR_PERCENTILE))
    threshold = max(min(floor + ACTIVITY_ABOVE_FLOOR_DB, ACTIVITY_CAP_DB), ACTIVITY_MIN_DB)
    peak = float(20 * np.log10(max(float(np.max(np.abs(blocks))), 1e-6)))
    return TrackEvidence(duration, int(np.sum(levels > threshold)), floor, peak)
