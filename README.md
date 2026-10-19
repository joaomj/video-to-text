# Meeting Recorder

[![Version](https://img.shields.io/badge/version-0.5.0-blue.svg)]

Record a meeting on your Mac (microphone plus system audio), then get a transcript with speaker labels and a separate meeting summary. No external recording app needed.

## What It Does

1. **Records** your microphone and everything your Mac plays, as separate tracks
2. **Warns during the meeting** if a source goes silent, flat, or too quiet
3. **Transcribes** both tracks via OpenRouter (default: MAI-Transcribe-2, with speaker labels)
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

# Cheaper transcription without speaker separation
uv run meeting record --stt-model openai/whisper-large-v3-turbo

# Record only; process later
uv run meeting record --no-process
uv run meeting process ~/Meetings/2026-10-19_153000

# Resume an interrupted run, or check its state
uv run meeting process ~/Meetings/2026-10-19_153000
uv run meeting status ~/Meetings/2026-10-19_153000

# List microphones
uv run meeting devices
```

Set `MEETING_LANGUAGE=pt` to change the default language, `STT_MODEL` / `SUMMARY_MODEL` to change models.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| _(none)_ | `pass://mac-mini/openrouter-macmini-video-to-text/API Key` | OpenRouter key, read from Proton Pass at runtime |
| `OPENROUTER_API_KEY` | — | Override: key as a value (not recommended) |
| `OPENROUTER_PASS_REF` | vault reference above | Override the vault reference; empty disables vault loading |
| `STT_MODEL` | `microsoft/mai-transcribe-2` | Speech-to-text model (`openai/whisper-large-v3-turbo` for ~10x cheaper, no speaker labels) |
| `SUMMARY_MODEL` | `deepseek/deepseek-v4.1-flash-1017` | Summary model (high reasoning effort) |
| `STT_BASE_URL` / `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Point at a local OpenAI-compatible server (e.g. whisper.cpp, llama.cpp) to run offline |

The key is never logged, never written to disk, and never passed as a command-line argument.

## Output

Each session folder (`~/Meetings/<date>/`) holds:

- `mic.wav` / `system.wav` — the original recordings (kept)
- `transcript.md` — full transcript with timestamps and speaker labels
- `summary.md` — main points, decisions, action items with owners/deadlines, open questions
- `recording.log` / `processing.log` / `session.json` — persistent logs and resumable stage state

A run ends as `complete`, `degraded` (warnings or sparse transcript), `incomplete` (transcript saved, summary failed), or `failed`.

## Local Models

Both API calls are OpenAI-compatible, so switching to a local server is configuration only:

```bash
export STT_BASE_URL=http://localhost:8080/v1   # whisper.cpp server with verbose_json timestamps
export LLM_BASE_URL=http://localhost:8081/v1   # llama.cpp server
export STT_MODEL=whisper-1
export SUMMARY_MODEL=qwen3-local
```

Note: llama.cpp's transcription endpoint returns text without timestamps, which this tool needs — use whisper.cpp for local speech-to-text.

## Costs (OpenRouter, defaults)

About **$0.10 per hour of audio** for transcription plus a few cents per meeting summary.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "FFmpeg not found" | `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux) |
| "No OPENROUTER_API_KEY..." | Run `pass-cli login`, or set `OPENROUTER_API_KEY` |
| "Recorder not found" | `swift build -c release --package-path recorder` |
| Remote voices share one label | Expected with whisper-turbo; use the default MAI model for speaker labels |
| Speakers labeled wrong | MAI assigns voices per request; edit manually |

## Technical Documentation

Architecture, session format, and development notes: [docs/tech-context.md](docs/tech-context.md). Target behavior: [docs/meeting-workflow-requirements.md](docs/meeting-workflow-requirements.md).

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Joao Marcos Visotaky Junior
