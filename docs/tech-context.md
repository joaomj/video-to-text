# Transcribe-Diarize System

## Overview

A CLI tool for transcribing and diarizing video recordings (MP4/MOV format). Extracts audio, identifies speakers, transcribes speech using a pluggable Whisper backend, and optionally analyzes transcripts with an LLM. Native Apple Silicon runs use MLX-Whisper, while Docker and other cross-platform environments use faster-whisper.

## Architecture

### Data Flow

```
Input Video (MP4/MOV)
    ↓
FFmpeg: Audio Extraction → WAV 16kHz mono
    ↓
Pyannote: Speaker Diarization → Speaker time ranges
    ↓
Whisper Backend Selection
    ├─ MLX-Whisper on Apple Silicon
    └─ faster-whisper in Docker/cross-platform mode
    ↓
Speech-to-Text → Transcript segments with timestamps
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
HF_ACCESS_TOKEN=hf_xxx           # Required: HuggingFace for Pyannote diarization
HF_ACCESS_TOKEN_FILE=/run/...    # Optional file-based secret alternative
GEMINI_API_KEY=xxx               # Optional: Preferred Gemini key for analysis
GEMINI_API_KEY_FILE=/run/...     # Optional file-based secret alternative
LLM_API_KEY=xxx                  # Optional: Backward-compatible fallback key
LLM_API_KEY_FILE=/run/...        # Optional file-based secret alternative
TRANSCRIPTION_BACKEND=auto       # auto, mlx, or faster
WHISPER_MODEL=large-v3           # large-v3 by default
FASTER_WHISPER_DEVICE=cpu        # cpu for Docker CPU images
FASTER_WHISPER_COMPUTE_TYPE=int8 # smaller CPU footprint for Docker
```

Loaded via `pydantic-settings` with:
- Type-safe validation
- File-based secret loading for Docker secrets or mounted secret files
- UTF-8 encoding

Secret values use Pydantic's `SecretStr` type to prevent accidental logging of API keys.

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
  --backend         Transcription backend override: auto, mlx, or faster
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

**Implementation Note**: Uses `RunIdFilter` on the root logging handler to inject `run_id` into every log record (including library logs). This ensures external library logs also include the correlation ID.

## Progress Bars and Timing

The script displays progress bars for long-running operations:

| Step | Progress Indicator |
|------|-------------------|
| Audio Extraction | Spinner with step name |
| Diarization | 3-step progress (load audio → create tensor → run model) |
| Speech Transcription | MLX-Whisper or faster-whisper model progress |
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
| `faster-whisper` | Cross-platform Whisper backend for Docker and CPU workflows |
| `pyannote-audio` | Speaker diarization |
| `torch<2.6`/`torchaudio<2.6` | ML framework + audio loading (pinned for compatibility) |
| `huggingface-hub<1.0` | Model download + auth (maintains use_auth_token API) |
| `pydantic-settings` | Type-safe .env handling |
| `httpx` | HTTP client for Zen API |
| `tqdm` | Progress bars for CLI |

## Decision Log

### Why MLX-Whisper plus faster-whisper?

**Decision**: Use dual local backends.

**Pros**:
- MLX remains the fastest native option on Apple Silicon
- faster-whisper enables Docker packaging and cross-platform adoption
- Both paths keep audio local and avoid API usage costs

**Cons**:
- Two backend implementations to maintain
- Docker CPU runs are slower than native MLX on M-series Macs

**Alternative**: OpenAI Whisper API would simplify packaging, but adds ongoing cost and sends audio off-machine.

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
- Docker Compose uses a file-based HuggingFace secret path instead of embedding secret values in env vars
- No hardcoded credentials
- API keys validated but not exposed

## Distribution

- `docker-compose.yml` supports local build-and-run development.
- `docker-compose.ghcr.yml` supports pull-and-run usage against a published GHCR image.
- `.github/workflows/publish-image.yml` validates multi-arch builds on pull requests and publishes on `main` pushes and version tags.
- GHCR users persist model downloads in the stable named Docker volume `transcribe-diarize-cache` mounted at `/home/transcribe/.cache`.
- Whisper and Pyannote weights are intentionally downloaded at runtime instead of being baked into the public image.

## File Structure

```
jobs/
├── transcribe_diarize.py        # Compatibility entry point
├── transcribe_diarize_app/      # Package modules and backend selection
├── pyproject.toml               # Native PDM dependencies
├── requirements-docker.txt      # Docker dependency set without MLX
├── Dockerfile                   # CPU-first container image
├── docker-compose.yml           # Local Docker workflow
├── docker-compose.ghcr.yml      # Pulled-image workflow with persistent cache
├── .github/workflows/           # GHCR publish automation
├── .env                         # Secrets (user-managed, git-ignored)
├── tests/                       # Backend selection and settings tests
    └── docs/
        └── tech-context.md          # This file
```

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| "FFmpeg not found" | FFmpeg not installed or not in PATH | `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux) |
| "HF_ACCESS_TOKEN not found" | Missing or invalid HuggingFace token | Check `.env` file or mounted secret file |
| "Access to pyannote/speaker-diarization-3.1: GATED" | Model agreement not accepted | Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and click "Access repository" |
| "GEMINI_API_KEY or LLM_API_KEY not found" | Missing LLM API key | Add key to `.env` or set environment variable |
| "Speakers are labeled wrong" | Pyannote assigns SPEAKER_00/01 arbitrarily | Manual fix in output markdown based on context |
| Long processing times | Video length + hardware limitations | Normal: 30min video = ~5-10min processing |
| "It downloads the models every time" | Missing persistent cache volume | Ensure `transcribe-diarize-cache` volume is mounted |
| "Weights only load failed" (PyTorch) | Compatibility issue with Pyannote 3.4 | Run `pdm sync` to get pinned versions (torch<2.6) |

## Known Limitations

1. **Docker CPU performance**: Cross-platform packaging is slower than native MLX on Apple Silicon
2. **FFmpeg**: Required system dependency for native runs, bundled in Docker
3. **HuggingFace**: Requires access token + model permissions
4. **LLM API**: Optional - analysis skipped without key
5. **File size**: Large videos may require significant RAM and several GB of model cache

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

| Version | Date | Key Changes |
|---------|------|-------------|
| 0.3.0 | 2026-03-16 | Meeting-agnostic: --type flag, --prompt-file, generic default |
| 0.2.0 | 2026-02-05 | LLM analysis, progress bars, timing summary |
| 0.1.1 | 2026-02-04 | Structured logging, PDM migration, expanded docs |
| 0.1.0 | 2026-02-02 | Initial release - transcription and diarization |
