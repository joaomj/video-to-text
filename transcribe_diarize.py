import warnings

warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*")

import sys
import argparse
import subprocess
import tempfile
import logging
import uuid
import time
from pathlib import Path
from datetime import timedelta

from typing import Any
from contextlib import contextmanager
from collections.abc import Generator
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm
import mlx_whisper
import torch
from pyannote.audio import Pipeline
from huggingface_hub import login
import httpx
import torchaudio

RUN_ID = uuid.uuid4().hex[:8]
STEP_TIMES: dict[str, float] = {}


class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = RUN_ID  # type: ignore[attr-defined]
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(run_id)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
for handler in logging.root.handlers:
    handler.addFilter(RunIdFilter())
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    hf_access_token: SecretStr
    llm_api_key: SecretStr | None = None


@contextmanager
def timed_step(name: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    logger.info(f"Starting: {name}")
    yield
    elapsed = time.perf_counter() - start
    STEP_TIMES[name] = elapsed
    logger.info(f"Completed: {name} in {elapsed:.2f}s")


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total = int(td.total_seconds())
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = int(seconds // 60), int(seconds % 60)
    return f"{minutes}m {secs}s"


def is_hallucination(text: str) -> bool:
    text_lower = text.lower().strip()
    if len(text_lower) < 3:
        return True
    words = text_lower.split()
    if len(words) >= 4 and words.count(max(set(words), key=words.count)) / len(words) > 0.6:
        return True
    patterns = [
        "thank you. thank you.", "thank you thank you", "openness openness",
        "finding out when", "we discovered when we discovered",
        "yeah. yeah. yeah. yeah", "it was good when",
    ]
    return any(p in text_lower for p in patterns)


def extract_audio(video_path: Path, output_audio_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", str(output_audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")


def analyze_transcript(transcript_path: Path, settings: Settings) -> Path | None:
    if settings.llm_api_key is None:
        logger.warning("LLM_API_KEY not set, skipping analysis")
        return None

    with timed_step("LLM Analysis"):
        prompt = f"""Analyze this job interview transcript:

1. Strengths: What did the candidate do well?
2. Areas for Improvement: Where could they improve?
3. Communication Style: Clarity, confidence, professionalism
4. Technical Answers: Depth and accuracy assessment
5. Actionable Recommendations: Tips for future interviews

Transcript:
{transcript_path.read_text()}"""

        try:
            with tqdm(total=1, desc="Sending to LLM", unit="request") as pbar:
                response = httpx.post(
                    "https://opencode.ai/zen/v1/chat/completions",
                    headers={"Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}"},
                    json={
                        "model": "kimi-k2.5",
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                pbar.update(1)

            analysis = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if not analysis:
                logger.error("Empty response from LLM API")
                return None

            analysis_path = transcript_path.with_stem(transcript_path.stem + "-analysis")
            analysis_path.write_text(f"# Interview Analysis\n\n{analysis}\n")
            logger.info(f"Analysis saved to {analysis_path}")
            return analysis_path
        except Exception as e:
            logger.error(f"Failed to analyze: {e}")
            return None


def process_transcription(
    audio_path: Path, output_path: Path, language: str, settings: Settings
) -> None:
    hf_token = settings.hf_access_token.get_secret_value()
    login(token=hf_token)

    with timed_step("Speaker Diarization"):
        logger.info("Loading Pyannote pipeline...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        if pipeline is None:
            logger.error("Failed to load Pyannote pipeline")
            sys.exit(1)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        try:
            pipeline.to(device)
        except Exception as e:
            logger.warning(f"Could not use {device}: {e}. Using CPU.")
            device = torch.device("cpu")

        with tqdm(total=3, desc="Diarization", unit="step") as pbar:
            waveform, sample_rate = torchaudio.load(audio_path)
            pbar.update(1)
            audio_data: dict[str, Any] = {"waveform": waveform, "sample_rate": sample_rate}
            pbar.update(1)
            diarization = pipeline(audio_data)
            pbar.update(1)

    with timed_step("Speech Transcription"):
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
            word_timestamps=True,
            language=language,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.0,
            logprob_threshold=-0.8,
            no_speech_threshold=0.5,
            hallucination_silence_threshold=1.0,
            initial_prompt="Job interview conversation.",
            verbose=True,  # Shows MLX-Whisper progress
        )

    with timed_step("Speaker Alignment"):
        segments = result.get("segments", [])
        segments = segments if isinstance(segments, list) else []

        transcript_segments: list[dict[str, Any]] = []
        for segment in tqdm(segments, desc="Aligning speakers", unit="segment"):
            start, end = float(segment["start"]), float(segment["end"])
            text = str(segment["text"]).strip()
            if is_hallucination(text):
                text = "[inaudible]"

            speaker, max_duration = "Unknown", 0.0
            for turn, _, s in diarization.itertracks(yield_label=True):
                overlap = min(end, float(turn.end)) - max(start, float(turn.start))
                if overlap > max_duration:
                    max_duration, speaker = overlap, str(s)

            transcript_segments.append({"start": start, "end": end, "speaker": speaker, "text": text})

    with timed_step("Writing Output"):
        with open(output_path, "w") as f:
            f.write(f"# Job Interview Transcript: {audio_path.stem}\n\n")
            last_speaker = None
            for seg in transcript_segments:
                timestamp = format_timestamp(seg["start"])
                if seg["speaker"] != last_speaker:
                    f.write(f"\n**[{timestamp}] {seg['speaker']}:** {seg['text']}")
                    last_speaker = seg["speaker"]
                else:
                    f.write(f" {seg['text']}")
            f.write("\n")
        logger.info(f"Transcript saved to {output_path}")


def print_timing_summary() -> None:
    if not STEP_TIMES:
        return
    total = sum(STEP_TIMES.values())
    logger.info("=" * 50)
    logger.info("TIMING SUMMARY")
    logger.info("=" * 50)
    for step_name, elapsed in STEP_TIMES.items():
        percentage = (elapsed / total) * 100 if total > 0 else 0
        logger.info(f"  {step_name:<25} {format_duration(elapsed):>10} ({percentage:5.1f}%)")
    logger.info("-" * 50)
    logger.info(f"  {'TOTAL':<25} {format_duration(total):>10}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe and diarize video files (MP4/MOV).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  pdm run transcribe video.mp4\n  pdm run transcribe interview.mov --language pt\n  pdm run transcribe video.mp4 --output /path\n  pdm run transcribe video.mp4 --skip-analysis\n  pdm run transcribe --analyze-only transcript.md",
    )
    parser.add_argument("input", help="Path to video file (MP4/MOV) or transcript (.md) if --analyze-only")
    parser.add_argument("--language", default="en", help='Language code (default: en)')
    parser.add_argument("--output", "-o", type=Path, help="Output directory (default: current directory)")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip LLM analysis")
    parser.add_argument("--analyze-only", action="store_true", help="Run only LLM analysis on existing transcript")

    args = parser.parse_args()
    input_path = Path(args.input)
    script_start = time.perf_counter()

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    settings = Settings()

    if args.analyze_only:
        # Run analysis only mode
        if input_path.suffix not in {".md", ".txt"}:
            logger.error(f"Invalid file type for analysis: {input_path.suffix}. Use .md or .txt")
            sys.exit(1)

        logger.info(f"Running analysis only on {input_path.name}")
        analyze_transcript(input_path, settings)

        total_elapsed = time.perf_counter() - script_start
        STEP_TIMES["Script Total"] = total_elapsed
        print_timing_summary()
        logger.info(f"Analysis complete!")
        return

    # Full pipeline mode
    valid_extensions = {".mp4", ".mov", ".MP4", ".MOV"}
    if input_path.suffix not in valid_extensions:
        logger.error(f"Invalid file type: {input_path.suffix}. Supported: {valid_extensions}")
        sys.exit(1)

    output_dir = args.output if args.output else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}-transcript.md"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio_path = Path(temp_dir) / "extracted_audio.wav"

        with timed_step("Audio Extraction"):
            logger.info(f"Extracting audio from {input_path.name}...")
            extract_audio(input_path, temp_audio_path)
            logger.info("Audio extracted successfully")

        process_transcription(temp_audio_path, output_path, args.language, settings)

    if not args.skip_analysis:
        analyze_transcript(output_path, settings)

    total_elapsed = time.perf_counter() - script_start
    STEP_TIMES["Script Total"] = total_elapsed
    print_timing_summary()
    logger.info(f"All done! Check {output_path}")


if __name__ == "__main__":
    main()
