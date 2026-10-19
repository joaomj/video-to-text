# Meeting Recorder

[![Version](https://img.shields.io/badge/version-0.5.0-blue.svg)]

Record a meeting on your Mac (microphone plus system audio), then get a transcript with speaker labels and a separate meeting summary. No external recording app needed.

## What It Does

1. **Records** your microphone and everything your Mac plays, as separate tracks
2. **Warns during the meeting** if a source goes silent, flat, or too quiet
3. **Transcribes** both tracks locally with MOSS-Transcribe-Diarize (timestamps plus speaker labels, no cloud)
4. **Writes `transcript.md`** with timestamps, "Me" for your mic, "Remote N" for other voices
5. **Writes `summary.md`** with main points, decisions, action items, and open questions

## Quick Start (macOS)

```bash
# 1. Install everything (Homebrew tools, Swift recorder, Python venv)
/install.sh

# 2. Log in to Proton Pass once (the OpenRouter key loads from the vault at runtime)
pass-cli login

# 3. Record; press Ctrl-C at the end and processing starts automatically
uv run meeting record
```

**Requirements:** Apple Silicon Mac, macOS 14.4+, Python 3.11+, UV (`brew install uv`), FFmpeg (`brew install ffmpeg`), `pass-cli` (`brew install proton-pass-cli`, logged in once).

## Usage

```bash
# Record with all defaults (English, ~/Meetings)
uv run meeting record

# Portuguese meeting
uv run meeting record --language pt

# Pick the headset mic, name the session
uv run meeting record --mic "External" --name "planning-2026-10-19"

# Record only; process later
uv run meeting record --no-process
uv run meeting process ~/Meetings/2026-10-19_153000

# Resume an interrupted run, or check its state
uv run meeting process ~/Meetings/2026-10-19_153000
uv run meeting status ~/Meetings/2026-10-19_153000

# List microphones
uv run meeting devices
```

Set `MEETING_LANGUAGE=pt` to change the default language, `SUMMARY_MODEL` to change the summary model.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| _(none)_ | `pass://mac-mini/openrouter-macmini-video-to-text/API Key` | OpenRouter key, read from Proton Pass at runtime |
| `OPENROUTER_API_KEY` | — | Override: key as a value (not recommended) |
| `OPENROUTER_PASS_REF` | vault reference above | Override the vault reference; empty disables vault loading |
| `SUMMARY_MODEL` | `deepseek/deepseek-v4.1-flash-1017` | Summary model (high reasoning effort) |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Point at a local OpenAI-compatible server (e.g. llama.cpp) to summarize offline |
| `MOSS_DEVICE` | `auto` | Transcription device: auto picks CUDA, Apple GPU, then CPU; or set `cuda`, `mps`, `cpu` |
| `MOSS_REVISION` | — | Pins the model revision for remote-code safety |
| `MOSS_MODEL_ID` | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | Override the transcription model |

The key is never logged, never written to disk, and never passed as a command-line argument.

## Output

Each session folder (`~/Meetings/<date>/`) holds:

- `mic.wav` / `system.wav` — the original recordings (kept)
- `transcript.md` — full transcript with timestamps and speaker labels
- `summary.md` — main points, decisions, action items with owners/deadlines, open questions
- `recording.log` / `processing.log` / `session.json` — persistent logs and resumable stage state

A run ends as `complete`, `degraded` (warnings or sparse transcript), `incomplete` (transcript saved, summary failed), or `failed`.

## Local Transcription

Transcription runs on this machine with MOSS-Transcribe-Diarize (Apache 2.0):
no audio leaves the Mac and there is no per-minute cost. The ~2 GB weights
download from Hugging Face on the first run and are reused after. GPU is used
whenever available (NVIDIA CUDA, then Apple GPU, then CPU); on this Mac Mini M4
30 seconds of audio takes about 10 seconds. Only the summary uses the cloud
(OpenRouter by default, or any OpenAI-compatible server via `LLM_BASE_URL`):

```bash
export LLM_BASE_URL=http://localhost:8081/v1   # llama.cpp server
export SUMMARY_MODEL=qwen3-local
```

## Costs

Transcription is free (local model). A meeting summary costs a few cents via OpenRouter.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "FFmpeg not found" | `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux) |
| "No OPENROUTER_API_KEY..." | Run `pass-cli login`, or set `OPENROUTER_API_KEY` |
| "Recorder not found" | `swift build -c release --package-path recorder` |
| Speakers labeled wrong | MOSS assigns `[S01]`, `[S02]` per recording; edit manually |

## Technical Documentation

Architecture, session format, and development notes: [docs/tech-context.md](docs/tech-context.md). Target behavior: [docs/meeting-workflow-requirements.md](docs/meeting-workflow-requirements.md).

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Joao Marcos Visotaky Junior
