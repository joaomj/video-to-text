# Configuration

All settings are environment variables (or a git-ignored `.env` file) via
`pydantic-settings`. See `.env.example` for a template.

## Key

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENROUTER_PASS_REF` | `pass://mac-mini/openrouter-macmini-video-to-text/API Key` | Proton Pass reference read at runtime. Empty disables vault loading. |
| `OPENROUTER_API_KEY` | — | Override: key as a value (not recommended). |
| `OPENROUTER_API_KEY_FILE` | — | Override: path to a file that holds the key. |
| `SUMMARY_API_KEY` | — | Summary key override. Falls back to the OpenRouter key. |
| `SUMMARY_API_KEY_FILE` | — | Summary key file override. |

The key is never logged, never written to disk, and never passed as a
command-line argument.

## Summary

| Variable | Default | Purpose |
|----------|---------|---------|
| `SUMMARY_MODEL` | `deepseek/deepseek-v4.1-flash-1017` | Summary model. |
| `SUMMARY_REASONING_EFFORT` | `high` | Reasoning effort for the summary call. |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Chat endpoint. Point at a local OpenAI-compatible server (for example llama.cpp) to summarize offline. |

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

## Offline example

```bash
export LLM_BASE_URL=http://localhost:8081/v1
export SUMMARY_MODEL=qwen3-local
```

Transcription already runs locally. These two settings move the summary
offline as well.
