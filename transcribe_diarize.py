import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import timedelta
from typing import Any
from dotenv import load_dotenv
import mlx_whisper
from pyannote.audio import Pipeline
import torch
from huggingface_hub import login

# Load environment variables (HF_ACCESS_TOKEN)
load_dotenv()


def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    return f"{minutes:02}:{seconds:02}"


def is_hallucination(text: str) -> bool:
    """Detect repetitive hallucination patterns in Whisper output."""
    text_lower = text.lower().strip()

    # Check for very short or empty segments
    if len(text_lower) < 3:
        return True

    # Check for repeated phrases (e.g., "thank you thank you thank you")
    words = text_lower.split()
    if len(words) >= 4:
        # If more than 60% of words are the same, likely hallucination
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        max_repeat = max(word_counts.values())
        if max_repeat / len(words) > 0.6:
            return True

    # Check for known hallucination patterns
    hallucination_patterns = [
        "thank you. thank you.",
        "thank you thank you",
        "openness openness",
        "finding out when",
        "we discovered when we discovered",
        "yeah. yeah. yeah. yeah",
        "it was good when",
    ]
    for pattern in hallucination_patterns:
        if pattern in text_lower:
            return True

    return False


def extract_audio(video_path: str, output_audio_path: str) -> None:
    """Extract audio from video file using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output if exists
        "-i",
        video_path,
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # PCM 16-bit little endian (WAV format)
        "-ar",
        "16000",  # Sample rate 16kHz (optimal for Whisper)
        "-ac",
        "1",  # Mono channel
        output_audio_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed to extract audio: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg: "
            "brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)"
        )


def process_transcription(audio_path: str, output_path: str, language: str) -> None:
    """Run diarization and transcription pipeline."""
    hf_token = os.getenv("HF_ACCESS_TOKEN")

    if not hf_token:
        print("Error: HF_ACCESS_TOKEN not found in .env file.")
        sys.exit(1)

    # Explicitly login to Hugging Face
    login(token=hf_token)

    print("Step 1: Running Diarization (Pyannote)...")
    # Load diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=hf_token
    )

    if pipeline is None:
        print(
            "Error: Failed to load Pyannote pipeline. "
            "Please check your token and repo access."
        )
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        pipeline.to(device)
    except Exception as e:
        print(f"Warning: Could not move pipeline to {device}: {e}. Falling back to CPU.")
        device = torch.device("cpu")

    # Manually load audio to avoid AudioDecoder issues in pyannote
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)
    audio_data = {"waveform": waveform, "sample_rate": sample_rate}

    diarization_output = pipeline(audio_data)
    # Extract the speaker diarization annotation from DiarizeOutput
    diarization = diarization_output.speaker_diarization

    print("Step 2: Running Transcription (MLX-Whisper)...")
    # Hardened transcription parameters to prevent hallucinations
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        word_timestamps=True,
        language=language,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.0,
        logprob_threshold=-0.8,
        no_speech_threshold=0.5,
        hallucination_silence_threshold=1.0,
        initial_prompt="Job interview conversation between interviewer and candidate.",
    )

    print("Step 3: Aligning Speakers and Text...")
    transcript_segments = []

    # Cast segments to list of dicts to satisfy type checker
    segments_raw = result.get("segments", [])
    segments: list[dict[str, Any]] = segments_raw if isinstance(segments_raw, list) else []

    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        text = str(segment["text"]).strip()

        # Skip hallucinated segments (mark as inaudible)
        if is_hallucination(text):
            text = "[inaudible]"

        # Find the dominant speaker in this time range
        speaker = "Unknown"
        max_duration = 0.0

        for turn, _, s in diarization.itertracks(yield_label=True):
            # Intersection of segment [start, end] and turn [turn.start, turn.end]
            overlap_start = max(start, float(turn.start))
            overlap_end = min(end, float(turn.end))
            overlap = overlap_end - overlap_start

            if overlap > max_duration:
                max_duration = overlap
                speaker = str(s)

        transcript_segments.append(
            {"start": start, "end": end, "speaker": speaker, "text": text}
        )

    print("Step 4: Writing Transcript...")
    input_name = Path(audio_path).stem
    with open(output_path, "w") as f:
        f.write(f"# Job Interview Transcript: {input_name}\n\n")
        last_speaker = None
        for seg in transcript_segments:
            timestamp = format_timestamp(seg["start"])
            if seg["speaker"] != last_speaker:
                f.write(f"\n**[{timestamp}] {seg['speaker']}:** {seg['text']}")
                last_speaker = seg["speaker"]
            else:
                f.write(f" {seg['text']}")
        f.write("\n")

    print(f"Done! Transcript saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe and diarize video files (MP4/MOV).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe_diarize.py video.mp4
  python transcribe_diarize.py interview.mov --language pt
""",
    )
    parser.add_argument("input", help="Path to video file (MP4 or MOV)")
    parser.add_argument(
        "--language",
        default="en",
        help="Language for transcription (default: en). "
        'Use language codes like "en", "pt", "es", "fr", etc.',
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    valid_extensions = {".mp4", ".mov", ".MP4", ".MOV"}
    if input_path.suffix not in valid_extensions:
        print(
            f"Error: Invalid file type: {input_path.suffix}. "
            f"Supported formats: {', '.join(sorted(valid_extensions))}"
        )
        sys.exit(1)

    # Generate output path
    output_path = input_path.parent / f"{input_path.stem}-transcript.md"

    # Create temp directory for audio extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio_path = os.path.join(temp_dir, "extracted_audio.wav")

        print(f"Extracting audio from {input_path.name}...")
        extract_audio(str(input_path), temp_audio_path)
        print(f"Audio extracted to temporary file")

        # Process transcription
        process_transcription(temp_audio_path, str(output_path), args.language)

    # Temp directory and audio file are automatically cleaned up here


if __name__ == "__main__":
    main()
