# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-02-05

### Added
- **LLM Analysis**: Integrated OpenCode Zen API with Kimi K2.5 model for AI-powered interview feedback
  - Analyzes transcripts for strengths, areas for improvement, communication style, and actionable recommendations
  - Costs approximately $0.005-$0.01 per interview (30 min - 1 hour)
- **Analysis-Only Mode**: New `--analyze-only` flag to run LLM analysis on existing transcript files
  - Supports `.md` and `.txt` transcript formats
  - Useful for re-analyzing transcripts with different prompts or after LLM improvements
- **Progress Bars**: Added tqdm progress bars for all long-running operations:
  - Audio extraction completion indicator
  - 3-step diarization progress (load → tensor → infer)
  - Per-segment speaker alignment progress
  - Single-request LLM analysis progress
- **Timing Summary**: Comprehensive timing breakdown at completion showing:
  - Elapsed time per step (audio extraction, diarization, transcription, alignment, LLM analysis)
  - Percentage of total time spent on each operation
  - Human-readable duration formatting

### Changed
- **LLM Provider**: Switched from Google Gemini 3 Flash to Kimi K2.5 via OpenCode Zen
  - Better handling of long context (interview transcripts)
  - More detailed and structured feedback output
  - Simpler API integration (OpenAI-style format)
- **Dependencies**: Added `httpx>=0.27` and `tqdm>=4.66` for LLM API and progress indicators
- **Documentation**: Significantly expanded README.md and created comprehensive tech-context.md

### Fixed
- **Hallucination Detection**: Fixed bug where `is_hallucination()` was comparing word string instead of count, causing false positives
- **PyTorch Compatibility**: Pinned `torch<2.6` and `torchaudio<2.6` to avoid `weights_only` parameter issues with Pyannote 3.4
- **HuggingFace Hub**: Pinned `huggingface-hub<1.0` to maintain `use_auth_token` API compatibility
- **Diarization API**: Fixed `AttributeError` - `pipeline()` returns `Annotation` directly, not a wrapper object

### Security
- API keys stored as `SecretStr` via Pydantic Settings (auto-masked in logs)
- `.env` file git-ignored by convention

## [0.1.1] - 2026-02-04

### Added
- **Structured Logging**: All logs now include:
  - ISO format timestamps (YYYY-MM-DD HH:MM:SS)
  - Log level (INFO, WARNING, ERROR)
  - 8-character correlation ID per run for traceability
  - `RunIdFilter` automatically injects correlation ID into all log records including library logs
- **PDM Integration**: Migrated from pip to PDM for dependency management
  - Lock file (`pdm.lock`) ensures reproducible builds
  - Built-in script runner (`pdm run transcribe`)
  - PEP 582 local packages (no venv activation needed)
- **Documentation**: Created `docs/tech-context.md` with detailed architecture and design decisions

### Changed
- **README Overhaul**: Completely rewritten with:
  - "Start Here" section for immediate comprehension
  - Architecture diagram showing data flow
  - Performance benchmarks (Apple M2 Pro)
  - Cost breakdown for LLM analysis
  - Common pitfalls and troubleshooting guide
  - Decision log explaining technology choices

### Technical
- Refactored monolithic script into cleaner structure with `timed_step` context manager
- Added `format_timestamp()` and `format_duration()` utilities
- Improved error handling with specific FFmpeg detection

## [0.1.0] - 2026-02-02

### Added
- Initial release of Transcribe-Diarize tool
- **Core Pipeline**:
  - FFmpeg audio extraction from MP4/MOV to 16kHz mono WAV
  - Pyannote 3.1 speaker diarization (identifies SPEAKER_00, SPEAKER_01, etc.)
  - MLX-Whisper transcription using Apple's MLX framework (M-series Macs only)
  - Speaker alignment by time overlap matching
  - Markdown transcript output with timestamps
- **Hallucination Filtering**: Built-in `is_hallucination()` function detects and filters Whisper hallucinations (marks as `[inaudible]`)
- **CLI Interface**: Basic argparse with `--language` support for non-English interviews
- **Configuration**: Environment-based config via `.env` file
  - `HF_ACCESS_TOKEN`: Required for HuggingFace/Pyannote authentication
- **Platform Requirements**:
  - macOS with Apple Silicon (M1/M2/M3) - MLX-Whisper requirement
  - Python 3.11+
  - FFmpeg system dependency
  - HuggingFace account with model permissions

### Technical Details
- ~274 lines of Python
- Dependencies: mlx-whisper, pyannote-audio, torch, torchaudio, huggingface-hub, pydantic-settings, python-dotenv
- First run downloads ~4-5GB of AI models (cached in `~/.cache/`)

[unreleased]: https://github.com/joaomj/transcribe-diarize/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/joaomj/transcribe-diarize/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/joaomj/transcribe-diarize/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/joaomj/transcribe-diarize/releases/tag/v0.1.0
