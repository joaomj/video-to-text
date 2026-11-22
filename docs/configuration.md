# Configuration

All settings are environment variables (or the git-ignored `.env` in the
project root, read from any working directory) via
`pydantic-settings`. See `.env.example` for a template.

## Transcription

The default engine is the `speech` CLI, which runs MOSS through MLX. It is
installed by `install.sh`; on an existing machine install it with
`brew install speech`. The int5 weights (~1 GB) download on first use to
`~/Library/Caches/qwen3-speech/models/aufklarer/`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `SPEECH_BIN` | — | Path to the `speech` binary (default: `speech` on PATH). |
| `SPEECH_ENGINE` | `moss` | Engine passed to the CLI. |
| `SPEECH_BACKEND` | `mlx` | CLI backend; `coreml` is the slower alternative. |
| `SPEECH_MODEL` | `int5` | MOSS weights: `int5` or `int8`. |
| `SPEECH_TIMEOUT_SECONDS` | `3600` | Per-window timeout. |

## Recording

| Variable | Default | Purpose |
|----------|---------|---------|
| `MEETING_LANGUAGE` | `en` | Spoken language (`en` or `pt`). The `--language` flag sets it per run. |
| `MEETREC_BIN` | — | Recorder binary override. Default is the release build under `recorder/.build`. |
