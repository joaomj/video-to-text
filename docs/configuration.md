# Configuration

All settings are environment variables (or the git-ignored `.env` in the
project root, read from any working directory) via
`pydantic-settings`. See `.env.example` for a template.

## Summary

The summary is generated through OpenRouter, so the transcript text is sent
to OpenRouter. The default model is pinned to a dated release
(`deepseek/deepseek-v4.1-flash-1017`) so summaries do not change when an
alias moves. Without `OPENROUTER_API_KEY` the session ends as `incomplete`
and the transcript is kept.

### OpenRouter key

Set one of these in the git-ignored `.env`:

- `OPENROUTER_API_KEY=<key>`, or
- `OPENROUTER_API_KEY_REF=pass://<vault>/<item>/<field>`, a Proton Pass
  reference. The summary stage reads it with `pass-cli item view`, so the key
  is never stored on disk. `pass-cli` must be logged in (`pass-cli info`).

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | — | OpenRouter key for the summary. |
| `OPENROUTER_API_KEY_REF` | — | Proton Pass reference for the key, used when `OPENROUTER_API_KEY` is unset. |
| `SUMMARY_MODEL` | `deepseek/deepseek-v4.1-flash-1017` | OpenRouter model ID. `~deepseek/deepseek-flash-latest` follows the newest Flash release. |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible summary endpoint. |

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
