# Configuration

All settings are environment variables (or a git-ignored `.env` file) via
`pydantic-settings`. See `.env.example` for a template.

## Summary

The summary runs on this Mac through llama-server. The server starts for the
summary stage and stops when that stage ends, so no weights stay resident
during capture or transcription. A server that is already listening is
reused and left running.

The Q4_K_M weights (~6 GB) download from Hugging Face on the first start
and are reused after. The server log is `llama-server.log` in the session
folder, capped at 512 KB.

| Variable | Default | Purpose |
|----------|---------|---------|
| `SUMMARY_MODEL` | `Qwen3.5-9B` | Summary model served by llama-server. |
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | Local summary server endpoint. |
| `LLM_HF_REPO` | `unsloth/Qwen3.5-9B-GGUF:Q4_K_M` | Model the server loads on start. |
| `LLM_CTX_SIZE` | `32768` | Server context size in tokens. |
| `LLAMA_SERVER_BIN` | — | Server binary override (default: `llama-server` on PATH). |
| `SUMMARY_SERVER_AUTOSTART` | `true` | Start and stop the server around the summary stage. Set `false` when you keep your own server running on `LLM_BASE_URL`. |

## Transcription

The default engine is the `speech` CLI, which runs MOSS through MLX. It is
installed by `install.sh`; on an existing machine install it with
`brew install speech`. The int5 weights (~1 GB) download on first use.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ASR_ENGINE` | `speech` | `speech` runs the CLI through MLX; `torch` runs the in-process model. |
| `SPEECH_BIN` | — | Path to the `speech` binary (default: `speech` on PATH). |
| `SPEECH_ENGINE` | `moss` | Engine passed to the CLI. |
| `SPEECH_BACKEND` | `mlx` | CLI backend; `coreml` is the slower alternative. |
| `SPEECH_MODEL` | `int5` | MOSS weights: `int5` or `int8`. |
| `SPEECH_TIMEOUT_SECONDS` | `3600` | Per-window timeout. |

The variables below apply only when `ASR_ENGINE=torch`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `MOSS_MODEL_ID` | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | Transcription model. |
| `MOSS_REVISION` | — | Pins the model revision for remote-code safety. |
| `MOSS_DEVICE` | `auto` | `auto` picks CUDA, then Apple GPU, then CPU. Set `cuda`, `mps`, or `cpu` to fix the device. |
| `MOSS_CACHE_DIR` | `~/.local/share/meeting-weights` | Weight storage outside the cache, so cache cleanup cannot delete the ~2 GB download. |

## Recording

| Variable | Default | Purpose |
|----------|---------|---------|
| `MEETING_LANGUAGE` | `en` | Spoken language (`en` or `pt`). The `--language` flag sets it per run. |
| `MEETREC_BIN` | — | Recorder binary override. Default is the release build under `recorder/.build`. |
