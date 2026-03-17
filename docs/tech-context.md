# Transcribe-Diarize System

## Overview

A CLI tool for transcribing and diarizing video recordings (MP4/MOV format). Extracts audio, identifies speakers, transcribes speech using MLX-Whisper, and optionally analyzes transcripts with LLM. Supports multiple meeting types: interviews, career discussions, ML meetings, and generic professional meetings.

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
LLM Analysis (optional): Gemini 3 Flash Preview via Gemini OpenAI-compatible API
    ↓
Analysis Output: {stem}-analysis.md
```

### Components

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| `extract_audio()` | FFmpeg wrapper | Video path | WAV file |
| `process_transcription()` | Core pipeline | Audio path, language, meeting_type | Markdown transcript |
| `analyze_transcript()` | LLM feedback | Transcript path, meeting_type, prompt_file | Analysis markdown |
| `is_hallucination()` | Filter Whisper noise | Text segment | Boolean |

## Meeting Type Templates

The `MEETING_PROMPTS` constant defines templates for each meeting type:

```python
MEETING_PROMPTS = {
    "interview": {
        "initial_prompt": "Job interview conversation.",
        "analysis_prompt": "...",
        "transcript_header": "Job Interview Transcript",
        "analysis_header": "Interview Analysis",
    },
    "generic": {
        "initial_prompt": "Professional meeting conversation.",
        "analysis_prompt": "...",
        "transcript_header": "Meeting Transcript",
        "analysis_header": "Meeting Analysis",
    },
}
```

| Template Field | Purpose |
|----------------|---------|
| `initial_prompt` | Whisper context hint for better transcription |
| `analysis_prompt` | LLM prompt for structured analysis |
| `transcript_header` | Markdown heading for transcript file |
| `analysis_header` | Markdown heading for analysis file |

**Custom Prompts**: Use `--prompt-file` to override the analysis prompt with a markdown file. The file must contain `{transcript}` placeholder.

## Configuration

### Environment Variables (`.env`)

```bash
HF_ACCESS_TOKEN=hf_xxx  # Required: HuggingFace for Pyannote diarization
GEMINI_API_KEY=xxx      # Optional: Preferred Gemini key for analysis
LLM_API_KEY=xxx         # Optional: Backward-compatible fallback key
```

Loaded via `pydantic-settings` with:
- Type-safe validation
- Auto-secret masking (keys never appear in logs)
- UTF-8 encoding

## CLI Interface

```bash
pdm run transcribe <input> [options]

Arguments:
  input             Path to video file (MP4/MOV) or transcript (.md/.txt) if --analyze-only

Options:
  --language, -l    Transcription language (default: en)
  --output, -o      Output directory (default: current working directory)
  --type, -t        Meeting type: interview or generic (default: generic)
  --prompt-file, -p Custom analysis prompt file (markdown)
  --skip-analysis   Skip LLM analysis even if API key present
  --analyze-only    Run only LLM analysis on existing transcript
```

**Modes:**
1. **Full Pipeline** (default): Video → Audio → Transcription → Optional Analysis
2. **Analysis Only** (`--analyze-only`): Existing transcript → LLM Analysis

**Meeting Types:**
- `interview`: Job interview analysis (strengths, improvements, communication, technical answers)
- `generic`: General meeting analysis (key topics, decisions, action items, open questions)

**Breaking Change**: Default type changed from `interview` to `generic`. Use `--type interview` for interview-specific analysis.

## Logging

All logs include:
- **Timestamp**: ISO format (YYYY-MM-DD HH:MM:SS)
- **Level**: INFO, WARNING, ERROR
- **Correlation ID**: 8-char UUID per run for traceability

Example:
```
2026-02-04 17:23:45 [INFO] [a1b2c3d4] Extracting audio from interview.mp4...
```

**Implementation Note**: Uses `RunIdFilter` on all logging handlers to inject `run_id` into every log record (including library logs). This ensures external library logs also include the correlation ID.

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
| `torch<2.6`/`torchaudio<2.6` | ML framework + audio loading (pinned for compatibility) |
| `huggingface-hub<1.0` | Model download + auth (maintains use_auth_token API) |
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

### Why Gemini 3 Flash Preview for Analysis?

**Decision**: Use `gemini-3-flash-preview` as the default analysis model.

**Reasons for Change**:
- Current active key is Gemini-based
- Fast response for long interview transcripts
- Compatible with OpenAI-style chat completions endpoint used by this script

**Tradeoffs**:
| Option | Pros | Cons |
|-------|------|------|
| Gemini 3 Flash Preview (current) | Fast, low cost, good quality | Output depth may vary on complex prompts |
| Higher-end LLMs | Potentially richer analysis | Higher cost and/or slower latency |

**Fallback Strategy**:
- Prefer `GEMINI_API_KEY`
- Fallback to `LLM_API_KEY` for backward compatibility

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
├── transcribe_diarize.py   # Main script
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

## Bug Fixes & Compatibility

### Recent Fixes (2025-02-05)

1. **PyTorch Compatibility**: Pinned to `torch<2.6` to avoid `weights_only` parameter issues with Pyannote 3.4
2. **HuggingFace Hub**: Pinned to `huggingface-hub<1.0` to maintain `use_auth_token` compatibility
3. **Hallucination Filter**: Fixed bug where `is_hallucination()` was comparing word string instead of count
4. **Diarization API**: Fixed `AttributeError` - `pipeline()` returns `Annotation` directly, not a wrapper object

## Future Improvements

- [x] Progress bars for long operations (v0.2.0)
- [x] Timing summary per step (v0.2.0)
- [x] Analysis-only mode for existing transcripts (v0.2.0)
- [x] LLM analysis integration (v0.2.0)
- [x] Meeting type templates (v0.3.0)
- [x] Custom analysis prompts (v0.3.0)
- [ ] Batch processing multiple videos
- [ ] Configurable LLM model selection
- [ ] Export formats (JSON, SRT)

## Version History

See [CHANGELOG.md](../CHANGELOG.md) for detailed release notes.

| Version | Date | Key Changes |
|---------|------|-------------|
| 0.3.0 | 2026-03-16 | Meeting-agnostic: --type flag, --prompt-file, generic default |
| 0.2.0 | 2026-02-05 | LLM analysis, progress bars, timing summary |
| 0.1.1 | 2026-02-04 | Structured logging, PDM migration, expanded docs |
| 0.1.0 | 2026-02-02 | Initial release - transcription and diarization |
