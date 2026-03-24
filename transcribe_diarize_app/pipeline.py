import subprocess
import sys
from pathlib import Path
from typing import Any, TypedDict

import httpx
import torch
import torchaudio
from huggingface_hub import login
from pyannote.audio import Pipeline
from tqdm import tqdm

from .constants import MEETING_PROMPTS
from .logging_utils import format_timestamp, logger, timed_step
from .settings import Settings
from .transcription import TranscriptionRequest, WhisperSegment, create_transcriber


class TranscriptSegment(TypedDict):
    start: float
    end: float
    speaker: str
    text: str


MIN_HALLUCINATION_TEXT_LENGTH = 3
MIN_REPETITION_WORDS = 4
REPETITION_RATIO_THRESHOLD = 0.6
DIARIZATION_PROGRESS_STEPS = 3


def is_hallucination(text: str) -> bool:
    text_lower = text.lower().strip()
    if len(text_lower) < MIN_HALLUCINATION_TEXT_LENGTH:
        return True
    words = text_lower.split()
    repeated_word_ratio = 0.0
    if words:
        repeated_word_ratio = words.count(max(set(words), key=words.count)) / len(words)
    if len(words) >= MIN_REPETITION_WORDS and repeated_word_ratio > REPETITION_RATIO_THRESHOLD:
        return True
    patterns = [
        "thank you. thank you.",
        "thank you thank you",
        "openness openness",
        "finding out when",
        "we discovered when we discovered",
        "yeah. yeah. yeah. yeah",
        "it was good when",
    ]
    return any(pattern in text_lower for pattern in patterns)


def extract_audio(video_path: Path, output_audio_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg failed: {exc.stderr}") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Install it or use the Docker image.") from exc


def analyze_transcript(
    transcript_path: Path,
    settings: Settings,
    meeting_type: str = "generic",
    prompt_file: Path | None = None,
) -> Path | None:
    llm_api_key = settings.resolved_llm_api_key()
    if llm_api_key is None:
        logger.warning(
            "GEMINI_API_KEY, GEMINI_API_KEY_FILE, LLM_API_KEY, or LLM_API_KEY_FILE not set"
        )
        logger.warning("Skipping analysis")
        return None

    with timed_step("LLM Analysis"):
        transcript_text = transcript_path.read_text(encoding="utf-8")
        if prompt_file is not None:
            prompt_template = prompt_file.read_text(encoding="utf-8")
            prompt = prompt_template.replace("{transcript}", transcript_text)
        else:
            template = MEETING_PROMPTS.get(meeting_type, MEETING_PROMPTS["generic"])
            prompt = template["analysis_prompt"].replace("{transcript}", transcript_text)

        try:
            with tqdm(total=1, desc="Sending to LLM", unit="request") as pbar:
                response = httpx.post(
                    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                    headers={"Authorization": f"Bearer {llm_api_key.get_secret_value()}"},
                    json={
                        "model": settings.llm_model,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                pbar.update(1)
        except httpx.HTTPError as exc:
            logger.error("Failed to analyze transcript: %s", exc)
            return None

        analysis = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        if not analysis:
            logger.error("Empty response from LLM API")
            return None

        template = MEETING_PROMPTS.get(meeting_type, MEETING_PROMPTS["generic"])
        analysis_path = transcript_path.with_stem(transcript_path.stem + "-analysis")
        analysis_content = f"# {template['analysis_header']}\n\n{analysis}\n"
        analysis_path.write_text(analysis_content, encoding="utf-8")
        logger.info("Analysis saved to %s", analysis_path)
        return analysis_path


def process_transcription(
    audio_path: Path,
    output_path: Path,
    language: str,
    settings: Settings,
    meeting_type: str = "generic",
    backend_name: str | None = None,
) -> None:
    hf_token = settings.require_hf_access_token().get_secret_value()
    login(token=hf_token)
    template = MEETING_PROMPTS.get(meeting_type, MEETING_PROMPTS["generic"])

    with timed_step("Speaker Diarization"):
        logger.info("Loading Pyannote pipeline...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        if pipeline is None:
            logger.error("Failed to load Pyannote pipeline")
            sys.exit(1)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info("Using diarization device: %s", device)

        try:
            pipeline.to(device)
        except RuntimeError as exc:
            logger.warning("Could not use %s: %s. Using CPU.", device, exc)
            device = torch.device("cpu")
            pipeline.to(device)

        with tqdm(total=DIARIZATION_PROGRESS_STEPS, desc="Diarization", unit="step") as pbar:
            waveform, sample_rate = torchaudio.load(audio_path)
            pbar.update(1)
            audio_data: dict[str, Any] = {"waveform": waveform, "sample_rate": sample_rate}
            pbar.update(1)
            diarization = pipeline(audio_data)
            pbar.update(1)

    selected_backend, transcriber = create_transcriber(settings, backend_name)
    logger.info("Using transcription backend: %s", selected_backend)
    logger.info("Using Whisper model: %s", settings.whisper_model)

    with timed_step("Speech Transcription"):
        result = transcriber.transcribe(
            TranscriptionRequest(
                audio_path=audio_path,
                language=language,
                initial_prompt=template["initial_prompt"],
            )
        )

    with timed_step("Speaker Alignment"):
        transcript_segments = align_speakers(result.get("segments", []), diarization)

    with timed_step("Writing Output"):
        write_transcript(
            output_path,
            audio_path.stem,
            template["transcript_header"],
            transcript_segments,
        )
        logger.info("Transcript saved to %s", output_path)


def align_speakers(segments: list[WhisperSegment], diarization: Any) -> list[TranscriptSegment]:
    transcript_segments: list[TranscriptSegment] = []
    for segment in tqdm(segments, desc="Aligning speakers", unit="segment"):
        start = float(segment["start"])
        end = float(segment["end"])
        text = str(segment["text"]).strip()
        if is_hallucination(text):
            text = "[inaudible]"

        speaker = "Unknown"
        max_duration = 0.0
        for turn, _, diarization_speaker in diarization.itertracks(yield_label=True):
            overlap = min(end, float(turn.end)) - max(start, float(turn.start))
            if overlap > max_duration:
                max_duration = overlap
                speaker = str(diarization_speaker)

        transcript_segments.append(
            {"start": start, "end": end, "speaker": speaker, "text": text}
        )
    return transcript_segments


def write_transcript(
    output_path: Path,
    audio_stem: str,
    transcript_header: str,
    transcript_segments: list[TranscriptSegment],
) -> None:
    parts = [f"# {transcript_header}: {audio_stem}\n"]
    last_speaker: str | None = None
    for segment in transcript_segments:
        timestamp = format_timestamp(segment["start"])
        if segment["speaker"] != last_speaker:
            parts.append(f"\n**[{timestamp}] {segment['speaker']}:** {segment['text']}")
            last_speaker = segment["speaker"]
            continue
        parts.append(f" {segment['text']}")
    parts.append("\n")
    output_path.write_text("".join(parts), encoding="utf-8")
