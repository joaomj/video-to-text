# Meeting Recorder

[![CI](https://github.com/joaomj/video-to-text/actions/workflows/ci.yml/badge.svg)]

Record a meeting on your Mac (microphone plus system audio), then get a
transcript with speaker labels and a separate meeting summary. No external
recording app needed.

Transcription and summary both run on this Mac: MOSS-Transcribe-Diarize
for speech, Qwen3.5-9B through llama-server for the summary. Nothing
leaves the Mac and there is no per-minute cost.

## Requirements

Apple Silicon Mac, macOS 14.4+. Everything else is installed by the setup script.

## Setup

```bash
./install.sh
```

`install.sh` does everything: installs UV, FFmpeg, and llama.cpp via
Homebrew, builds the Swift recorder, installs the Python environment, and
adds a `meet` alias to `~/.zshrc`. Model weights (transcription ~2 GB,
summary ~6 GB) download automatically on first use and are reused after.
Restart the terminal after setup (or run `source ~/.zshrc`).

## Meeting workflow

Start capturing when the meeting begins:

```bash
meet
```

Ctrl-C ends the capture. Recording loads no model, so a meeting never
competes with transcription or summary weights for memory. Process the
session when the machine is free:

```bash
uv run --project ~/projects/video-to-text meeting process ~/Meetings/2026-10-19_153000
```

Output lands in `~/Meetings/<date>/`: `transcript.md` plus `summary.md`.
The summary stage starts llama-server for that step only, then stops it.
Processing refuses to start when the machine is short of memory; see
[docs/operations.md](docs/operations.md#machine-capacity).

```bash
meet --language pt --name "planning"  # Portuguese session with a name
meet --mic "External" --process       # transcribe as soon as you stop
```

English is the default language. Set `MEETING_LANGUAGE=pt` to change it.

List microphones, or retry and inspect a session (these work from any folder):

```bash
uv run --project ~/projects/video-to-text meeting process ~/Meetings/2026-10-19_153000
uv run --project ~/projects/video-to-text meeting status ~/Meetings/2026-10-19_153000
uv run --project ~/projects/video-to-text meeting devices
```

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

