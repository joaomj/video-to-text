# Meeting Recorder

[![CI](https://github.com/joaomj/video-to-text/actions/workflows/ci.yml/badge.svg)]

Record a meeting on your Mac (microphone plus system audio), then get a
transcript with speaker labels and a separate meeting summary. No external
recording app needed.

Transcription and summary both run on this Mac: MOSS-Transcribe-Diarize
for speech, Qwen3.5-9B through llama-server for the summary. Nothing
leaves the Mac and there is no per-minute cost.

## Requirements

Apple Silicon Mac, macOS 14.4+, Python 3.11+, UV (`brew install uv`),
FFmpeg (`brew install ffmpeg`), llama.cpp (`brew install llama.cpp`,
provides `llama-server`).

## Setup

```bash
./install.sh
uv run meeting record
```

Press Ctrl-C at the end and processing starts automatically. `record` starts
the local summary server during capture and stops it after processing.
Output lands in `~/Meetings/<date>/`: `transcript.md` plus `summary.md`.

For a quick start on entering a meeting, `install.sh` defines a `meet`
alias (in `~/.zshrc`): `meet` records, `meet --language pt` records in
Portuguese. English is the default language.

## Common usage

```bash
uv run meeting record --language pt --name "planning"
uv run meeting record --mic "External" --no-process
uv run meeting process ~/Meetings/2026-10-19_153000
uv run meeting status ~/Meetings/2026-10-19_153000
uv run meeting devices
```

Set `MEETING_LANGUAGE=pt` to change the default language.

## Configuration

The summary server runs on this Mac. Full variable reference:
[docs/configuration.md](docs/configuration.md).

| Variable | Default | Purpose |
|----------|---------|---------|
| `SUMMARY_MODEL` | `Qwen3.5-9B` | Summary model served by llama-server |
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | Local summary server endpoint |
| `LLM_HF_REPO` | `unsloth/Qwen3.5-9B-GGUF:Q4_K_M` | Model the server loads on start |
| `LLM_CTX_SIZE` | `32768` | Server context size in tokens |
| `LLAMA_SERVER_BIN` | — | Server binary override (default: `llama-server` on PATH) |

## Docs

- [docs/architecture.md](docs/architecture.md) — pipeline, components, speaker labels.
- [docs/operations.md](docs/operations.md) — session files, outcomes, recovery, troubleshooting.
- [docs/configuration.md](docs/configuration.md) — full environment reference.
- [docs/development.md](docs/development.md) — checks, CI, releases.
- [docs/meeting-workflow-requirements.md](docs/meeting-workflow-requirements.md) — product requirements.

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Joao Marcos Visotaky Junior
