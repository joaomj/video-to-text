# Meeting Recorder

[![CI](https://github.com/joaomj/video-to-text/actions/workflows/ci.yml/badge.svg)]

Record a meeting on your Mac (microphone plus system audio), then get a
transcript with speaker labels and a separate meeting summary. No external
recording app needed.

Transcription runs locally with MOSS-Transcribe-Diarize: no audio leaves the
Mac and there is no per-minute cost. Only the summary uses the network
(OpenRouter by default, or any OpenAI-compatible server).

## Requirements

Apple Silicon Mac, macOS 14.4+, Python 3.11+, UV (`brew install uv`),
FFmpeg (`brew install ffmpeg`), `pass-cli` (`brew install proton-pass-cli`,
logged in once).

## Setup

```bash
./install.sh
pass-cli login
uv run meeting record
```

Press Ctrl-C at the end and processing starts automatically. Output lands in
`~/Meetings/<date>/`: `transcript.md` plus `summary.md`.

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

The OpenRouter key loads from Proton Pass at runtime. Full variable reference:
[docs/configuration.md](docs/configuration.md).

| Variable | Default | Purpose |
|----------|---------|---------|
| _(none)_ | `pass://mac-mini/openrouter-macmini-video-to-text/API Key` | Vault reference for the key |
| `OPENROUTER_API_KEY` | — | Override: key as a value (not recommended) |
| `SUMMARY_MODEL` | `deepseek/deepseek-v4.1-flash-1017` | Summary model |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Local OpenAI-compatible server for offline summaries |

## Docs

- [docs/architecture.md](docs/architecture.md) — pipeline, components, speaker labels.
- [docs/operations.md](docs/operations.md) — session files, outcomes, recovery, troubleshooting.
- [docs/configuration.md](docs/configuration.md) — full environment reference.
- [docs/development.md](docs/development.md) — checks, CI, releases.
- [docs/meeting-workflow-requirements.md](docs/meeting-workflow-requirements.md) — product requirements.

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Joao Marcos Visotaky Junior
