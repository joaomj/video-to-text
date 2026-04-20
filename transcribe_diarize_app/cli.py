import argparse
import os
import tempfile
import time
from pathlib import Path

from .constants import SUPPORTED_LANGUAGES
from .logging_utils import STEP_TIMES, PipelineProgress, logger, print_timing_summary, timed_step
from .pipeline import analyze_transcript, extract_audio, process_transcription
from .settings import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe and diarize video files (MP4/MOV).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run transcribe video.mp4\n"
            "  uv run transcribe interview.mov --type interview --language pt\n"
            "  uv run transcribe video.mp4 --output /path\n"
            "  uv run transcribe video.mp4 --skip-analysis\n"
            "  uv run transcribe --analyze-only transcript.md --type generic\n"
            "  uv run transcribe meeting.mp4 --prompt-file custom_prompt.md\n"
            "  uv run transcribe meeting.mp4 --backend faster"
        ),
    )
    parser.add_argument(
        "input", help="Path to video file (MP4/MOV) or transcript (.md) if --analyze-only"
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=sorted(SUPPORTED_LANGUAGES),
        help=f"Language code (default: en). Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}",
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Output directory (default: same as input file)"
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["interview", "generic"],
        default="generic",
        help="Meeting type (default: generic)",
    )
    parser.add_argument(
        "--prompt-file", "-p", type=Path, help="Custom analysis prompt file (markdown)"
    )
    parser.add_argument("--skip-analysis", action="store_true", help="Skip LLM analysis")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run only LLM analysis on existing transcript",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "mlx", "faster"],
        help="Transcription backend override (default: TRANSCRIPTION_BACKEND or auto)",
    )
    return parser


def resolve_output_dir(input_path: Path, output_arg: Path | None) -> Path:
    if output_arg is not None:
        output_dir = output_arg
    else:
        input_parent = input_path.parent
        output_dir = input_parent if os.access(input_parent, os.W_OK) else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    input_path = Path(args.input)
    script_start = time.perf_counter()

    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    settings = Settings()

    if args.analyze_only:
        run_analysis_only(input_path, settings, args.type, args.prompt_file, script_start)
        return

    valid_extensions = {".mp4", ".mov", ".MP4", ".MOV"}
    if input_path.suffix not in valid_extensions:
        parser.error(
            f"Invalid file type: {input_path.suffix}. Supported: {sorted(valid_extensions)}"
        )

    output_dir = resolve_output_dir(input_path, args.output)
    output_path = output_dir / f"{input_path.stem}-transcript.md"
    total_steps = 4 if args.skip_analysis else 5
    progress = PipelineProgress(total_steps, desc=f"Processing {input_path.name}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio_path = Path(temp_dir) / "extracted_audio.wav"

        with timed_step("Audio Extraction"):
            logger.info("Extracting audio from %s...", input_path.name)
            extract_audio(input_path, temp_audio_path)
            logger.info("Audio extracted successfully")
            progress.step("Extracting audio")

        process_transcription(
            temp_audio_path,
            output_path,
            args.language,
            settings,
            meeting_type=args.type,
            backend_name=args.backend,
            progress=progress,
        )

    if not args.skip_analysis:
        analyze_transcript(
            output_path,
            settings,
            meeting_type=args.type,
            prompt_file=args.prompt_file,
            progress=progress,
        )

    progress.close()
    STEP_TIMES["Script Total"] = time.perf_counter() - script_start
    print_timing_summary()
    logger.info("All done! Check %s", output_path)


def run_analysis_only(
    input_path: Path,
    settings: Settings,
    meeting_type: str,
    prompt_file: Path | None,
    script_start: float,
) -> None:
    if input_path.suffix not in {".md", ".txt"}:
        raise SystemExit(f"Invalid file type for analysis: {input_path.suffix}. Use .md or .txt")

    logger.info("Running analysis only on %s", input_path.name)
    progress = PipelineProgress(1, desc=f"Analyzing {input_path.name}")
    analyze_transcript(
        input_path,
        settings,
        meeting_type=meeting_type,
        prompt_file=prompt_file,
        progress=progress,
    )
    progress.close()
    STEP_TIMES["Script Total"] = time.perf_counter() - script_start
    print_timing_summary()
    logger.info("Analysis complete!")
