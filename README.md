# Meeting Recorder

[![CI](https://github.com/joaomj/video-to-text/actions/workflows/ci.yml/badge.svg)](https://github.com/joaomj/video-to-text/actions/workflows/ci.yml)

Record a meeting on your Mac (microphone plus system audio), then get a
transcript with speaker labels and a separate meeting summary. No external
recording app needed.

Transcription and summary both run on this Mac: MOSS-Transcribe-Diarize
through the MLX runtime for speech, Qwen3.5-9B through llama-server for the
summary. Nothing leaves the Mac and there is no per-minute cost. A 40-minute
meeting transcribes in about 6.4 minutes with 1.3 GB of memory per window.

## Requirements

Apple Silicon Mac, macOS 14.4+, and the Xcode Command Line Tools (Swift).
Everything else is installed by the setup script.

## Setup

```bash
./install.sh
```

`install.sh` does everything: installs UV, FFmpeg, llama.cpp, and the
`speech` transcription runtime via Homebrew, builds the Swift recorder,
installs the Python environment, and adds a `meet` alias to `~/.zshrc`.
Model weights (transcription ~1 GB, summary ~6 GB) download automatically on
first use and are reused after. Restart the terminal after setup (or run
`source ~/.zshrc`).

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

Inspect a session, or list microphones (these work from any folder):

```bash
uv run --project ~/projects/video-to-text meeting status ~/Meetings/2026-10-19_153000
uv run --project ~/projects/video-to-text meeting devices
```

English is the default language. Set `MEETING_LANGUAGE=pt` to change it.

## Transcription

Each track is cut into five-minute windows with 30 seconds of overlap and
transcribed window by window, so memory stays flat no matter how long the
meeting runs. Speaker labels are stitched across the overlap, so a label can
shift between windows.

Measured on a Mac mini (M4, 16 GB) with the same 40-minute meeting:

| Transcription approach | Time | Peak memory |
|------------------------|------|-------------|
| Single pass, torch (rejected) | 58.5 min | ~10 GB, swapping |
| Five-minute windows, torch (rejected) | ~9 min | 3.4 GB |
| Whole track, transcribe.cpp (rejected) | ~7 min | 3.6 GB |
| Five-minute windows, MLX (`speech`) | **6.4 min** | **1.3 GB** |

The first and last rows ran end to end on this meeting; the middle two are
extrapolated from measured ten-minute slices. Only the MLX path ships. It
wins on memory, and memory no longer grows with meeting length because each
window is a separate process.

The summary stage is separate: llama-server briefly takes about 8 GB and
starts only for that stage.

Transcription runs through the `speech` CLI (MOSS on MLX). Its weights
(~1 GB) download to `~/Library/Caches/qwen3-speech` on first use. Windows
with no speech produce no output, which is expected and logged; a silent
window never fails the run.

## Configuration

Summary and transcription settings. Full variable reference:
[docs/configuration.md](docs/configuration.md).

| Variable | Default | Purpose |
|----------|---------|---------|
| `SUMMARY_MODEL` | `Qwen3.5-9B` | Summary model served by llama-server |
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | Local summary server endpoint |
| `LLM_HF_REPO` | `unsloth/Qwen3.5-9B-GGUF:Q4_K_M` | Model the server loads on start |
| `LLM_CTX_SIZE` | `32768` | Server context size in tokens |
| `LLAMA_SERVER_BIN` | — | Server binary override (default: `llama-server` on PATH) |
| `SPEECH_MODEL` | `int5` | MOSS weights: `int5` or `int8` |
| `SPEECH_BACKEND` | `mlx` | Runtime backend; `coreml` is slower |

## Docs

- [docs/architecture.md](docs/architecture.md) — pipeline, components, speaker labels.
- [docs/operations.md](docs/operations.md) — session files, outcomes, recovery, troubleshooting.
- [docs/configuration.md](docs/configuration.md) — full environment reference.
- [docs/development.md](docs/development.md) — checks, CI, releases.
- [docs/meeting-workflow-requirements.md](docs/meeting-workflow-requirements.md) — product requirements.

## License

MIT License - see [LICENSE](LICENSE) file for details.

