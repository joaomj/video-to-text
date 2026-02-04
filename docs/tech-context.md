# Transcribe-Diarize System

## Overview

A CLI tool for transcribing and diarizing job interview videos (MP4/MOV format). Extracts audio, identifies speakers, transcribes speech using MLX-Whisper, and optionally analyzes transcripts with LLM for interview feedback.

## Architecture

### Data Flow

```
Input Video (MP4/MOV)
    ↓
FFmpeg: Audio Extraction → WAV 16kHz mono
    ↓
Pyannote: Speaker Diarization → Speaker time ranges
    ↓
MLX-Whisper: Speech-to-Text → Transcript segments with timestamps
    ↓
Alignment: Match speakers to transcript segments by time overlap
    ↓
Markdown Output: {stem}-transcript.md
    ↓
LLM Analysis (optional): Gemini 3 Flash via OpenCode Zen API
    ↓
Analysis Output: {stem}-analysis.md
```

### Components

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| `extract_audio()` | FFmpeg wrapper | Video path | WAV file |
| `process_transcription()` | Core pipeline | Audio path, language | Markdown transcript |
| `analyze_transcript()` | LLM feedback | Transcript path | Analysis markdown |
| `is_hallucination()` | Filter Whisper noise | Text segment | Boolean |

## Configuration

### Environment Variables (`.env`)

```bash
HF_ACCESS_TOKEN=hf_xxx  # Required: HuggingFace for Pyannote diarization
LLM_API_KEY=xxx         # Optional: OpenCode Zen for analysis
```

Loaded via `pydantic-settings` with:
- Type-safe validation
- Auto-secret masking (keys never appear in logs)
- UTF-8 encoding

## CLI Interface

```bash
pdm run transcribe <video> [options]

Options:
  --language, -l    Transcription language (default: en)
  --output, -o      Output directory (default: current working directory)
  --skip-analysis   Skip LLM analysis even if API key present
```

## Logging

All logs include:
- **Timestamp**: ISO format (YYYY-MM-DD HH:MM:SS)
- **Level**: INFO, WARNING, ERROR
- **Correlation ID**: 8-char UUID per run for traceability

Example:
```
2026-02-04 17:23:45 [INFO] [a1b2c3d4] Extracting audio from interview.mp4...
```

## Progress Bars and Timing

The script displays progress bars for long-running operations:

| Step | Progress Indicator |
|------|-------------------|
| Audio Extraction | Spinner with step name |
| Diarization | 3-step progress (load audio → create tensor → run model) |
| Speech Transcription | MLX-Whisper built-in progress |
| Speaker Alignment | Per-segment progress bar |
| LLM Analysis | Single request progress |

**Timing Summary** displayed at completion:
```
==================================================
TIMING SUMMARY
==================================================
  Audio Extraction            2.5s (  3.2%)
  Speaker Diarization         45.2s ( 58.1%)
  Speech Transcription        25.1s ( 32.3%)
  Speaker Alignment            4.8s (  6.2%)
  LLM Analysis                 0.2s (  0.3%)
  Writing Output               0.1s (  0.1%)
--------------------------------------------------
  Script Total                77.9s
==================================================
```

Each step shows:
- **Elapsed time**: Human readable (seconds or minutes)
- **Percentage**: Of total script time
- **Completion notification**: Logged when each step finishes

## Dependencies

| Package | Purpose |
|---------|---------|
| `mlx-whisper` | Apple's MLX-optimized Whisper (M-series Macs) |
| `pyannote-audio` | Speaker diarization |
| `torch`/`torchaudio` | ML framework + audio loading |
| `huggingface-hub` | Model download + auth |
| `pydantic-settings` | Type-safe .env handling |
| `httpx` | HTTP client for Zen API |
| `tqdm` | Progress bars for CLI |

## Decision Log

### Why MLX-Whisper over OpenAI Whisper?

**Decision**: Use MLX-Whisper (Apple Silicon optimized)

**Pros**:
- 3-5x faster on M-series Macs
- No API costs
- Local processing (privacy)

**Cons**:
- Mac-only (M-series)
- Slightly different model weights

**Alternative**: OpenAI Whisper API - better cross-platform, but requires internet + costs.

### Why Pyannote 3.1?

State-of-the-art speaker diarization with:
- Pre-trained on diverse datasets
- MPS/CPU device support
- HuggingFace integration

### Why Gemini 3 Flash for Analysis?

**Decision**: Use `gemini-3-flash` via OpenCode Zen

**Tradeoffs**:
| Model | Cost/M tokens | Speed | Quality |
|-------|--------------|-------|---------|
| gemini-3-flash | $0.50/$3.00 | Fast | Good |
| claude-sonnet-4 | $3/$15 | Medium | Excellent |
| gpt-5-nano | Free | Fastest | Basic |

Flash offers best cost/performance ratio for interview feedback tasks.

### Why PDM over pip/venv?

**Decision**: Use PDM for dependency management

**Pros**:
- Lock file for reproducible builds
- PEP 582 local packages (no venv activation)
- Built-in script runner

**Cons**:
- Learning curve for pip users
- Additional tool dependency

## Security

- Secrets via `SecretStr` - never logged
- `.env` in `.gitignore` (user-managed)
- No hardcoded credentials
- API keys validated but not exposed

## File Structure

```
jobs/
├── transcribe_diarize.py   # Main script (274 lines)
├── pyproject.toml          # PDM + dependencies
├── pdm.lock               # Locked dependency versions
├── .env                   # Secrets (user-managed, git-ignored)
└── docs/
    └── tech-context.md    # This file
```

## Known Limitations

1. **Platform**: MLX-Whisper requires Apple Silicon (M1/M2/M3)
2. **FFmpeg**: Required system dependency
3. **HuggingFace**: Requires access token + model permissions
4. **LLM API**: Optional - analysis skipped without key
5. **File size**: Large videos may require significant RAM

## Future Improvements

- [x] Progress bars for long operations (implemented)
- [x] Timing summary per step (implemented)
- [ ] Batch processing multiple videos
- [ ] Configurable LLM model selection
- [ ] Custom analysis prompts
- [ ] Export formats (JSON, SRT)
