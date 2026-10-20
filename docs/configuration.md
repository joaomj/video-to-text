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
