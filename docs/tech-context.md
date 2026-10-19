# Meeting Recorder System

## Overview

A meeting recorder and processor. A Swift recorder captures microphone and
system audio on macOS with live capture-health monitoring. A Python pipeline
then transcribes both tracks through OpenRouter speech-to-text, merges them
into one transcript, and generates a meeting summary through an
OpenAI-compatible chat model.

## Architecture

### Data Flow

```
meetrec record (Swift, macOS)
    Mic track ──────────────────────► mic.wav (16-bit mono WAV)
    System tap ─────────────────────► system.wav (16-bit mono WAV)
    Health monitor ──► terminal warnings + recording.log
    Ctrl-C ──► finalize + session.json ("recorded")
    ↓
meeting process (Python)
    Validate recording (headers, duration, recorder outcome)
    ↓
    FFmpeg: normalize each track → work/<name>-16k.wav (sources untouched)
    ↓
    OpenRouter STT per track (MP3 chunks, verbose_json timestamps)
      mic track → labeled "Me" (diarization off)
      system track → diarized speaker indices → "Remote N"
    ↓
    Drop mic segments that echo remote speech
    ↓
    Merge by timestamp → transcript.md
    ↓
    Chat completions summary → summary.md
```

### Components

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| `recorder/` (Swift) | Capture + health monitoring | Mic, system tap | `mic.wav`, `system.wav`, `session.json`, `recording.log` |
| `meeting_audio.py` | Normalize to 16 kHz mono, measure signal evidence | Track WAV | `work/*-16k.wav`, activity stats |
| `stt.py` | OpenRouter / OpenAI-compatible transcription | 16 kHz WAV | Segments with timestamps + speaker ids |
| `meeting_pipeline.py` | Stages, merging, completeness, transcript, summary | Session folder | `transcript.md`, `summary.md` |
| `summary.py` | OpenAI-compatible chat summary | Transcript text | Summary markdown |
| `vault.py` | Runtime key loading from Proton Pass | `pass://` reference | Key in memory only |
| `session.py` | Manifest, stage keys, atomic writes, process lock | Session folder | Resumable state |

### Speaker Labels

- The mic track is one known speaker: every segment is "Me".
- The system track is transcribed with diarization enabled
  (MAI-Transcribe-2 via `provider.options.azure.diarization`). Returned speaker
  indices map to "Remote 1", "Remote 2", ... in order of first appearance.
- A mic segment overlapping a system segment with similar words is speaker echo
  leaking into the microphone, and is dropped from "Me".
- Models without diarization (e.g. whisper-turbo) label every remote segment
  "Remote" with no separation.

### Transcription Chunking

One request per track is tried first (up to 10 minutes of audio). OpenRouter
limits each request to ~60 seconds of processing time; only on timeout or size
rejection does the client split into 10-minute MP3 chunks and stitch timestamps
with offsets. Single requests keep speaker numbering consistent; chunking can
reset it.

## Configuration

See `.env.example`. All settings are environment variables (or a git-ignored
`.env` file) via `pydantic-settings`:

- Key: `OPENROUTER_API_KEY`, `*_FILE`, or vault reference `OPENROUTER_PASS_REF`
  (default `pass://mac-mini/openrouter-macmini-video-to-text/API Key`).
- Models: `STT_MODEL`, `SUMMARY_MODEL`, `SUMMARY_REASONING_EFFORT`.
- Endpoints: `STT_BASE_URL`, `LLM_BASE_URL` (OpenRouter by default; any
  OpenAI-compatible server for local models).
- Language: `MEETING_LANGUAGE` (default `en`); `--language` per run.

## Session Format

`~/Meetings/<id>/` (or `--out`/`--name`):

- `mic.wav`, `system.wav` — source recordings, never modified after capture
- `session.json` — manifest: sources, recording outcome, health warnings,
  processing stages with input-fingerprint keys, final state
- `work/` — normalized audio, per-track segments, merged segments (resumable)
- `transcript.md`, `summary.md` — final documents, written atomically
- `recording.log`, `processing.log` — persistent logs

Stage keys hash input file fingerprints plus relevant config, so a rerun skips
completed stages, resumes interrupted ones, and never duplicates final
documents. A file lock prevents two processing runs on one session.

## Outcomes

- `complete` — transcript and summary saved
- `degraded` — saved, but with capture warnings, missing speaker labels, or a
  sparse transcript for the amount of active audio
- `incomplete` — transcript saved, summary failed (retry with `meeting process`)
- `failed` — no usable audio or empty transcript; recordings are kept

## Dependencies

| Package | Purpose |
|---------|---------|
| `httpx` | OpenRouter / OpenAI-compatible HTTP calls |
| `numpy` | Audio level measurement |
| `pydantic` + `pydantic-settings` | Type-safe settings, secret handling |
| `python-dotenv` | `.env` file loading |

System: FFmpeg (audio), Swift 6 toolchain (recorder build), `pass-cli`
(logged in once; runtime key loading).

## Distribution

There is no container image. `install.sh` installs Homebrew tools, builds the
Swift recorder, and syncs the Python environment on macOS.

## File Structure

```
recorder/                    # Swift capture tool (meetrec)
transcribe_diarize_app/      # Python package (meeting command)
├── meeting_cli.py           # record / process / status / devices
├── meeting_pipeline.py      # stages, merging, transcript, summary
├── meeting_audio.py         # normalize + signal evidence
├── stt.py                   # transcription client (cloud + local)
├── summary.py               # chat summary client (cloud + local)
├── vault.py                 # Proton Pass runtime key loading
├── session.py               # manifest, stage keys, locking
├── hallucinations.py        # noise-segment filter
├── constants.py             # languages, prompts, thresholds
├── settings.py              # env-based configuration
└── logging_utils.py         # logging with run id + file logs
tests/test_hallucinations.py
```

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| "FFmpeg not found" | FFmpeg not installed | `brew install ffmpeg` / `apt-get install ffmpeg` |
| "No OPENROUTER_API_KEY..." | Vault unreachable and no key set | `pass-cli login`, or set `OPENROUTER_API_KEY` |
| "Recorder not found" | Swift binary not built | `swift build -c release --package-path recorder` |
| Text without timestamps | Local STT server lacks verbose_json | Use whisper.cpp server for local transcription |
| One "Remote" label for all | Model has no diarization | Use MAI-Transcribe-2 (default) |

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| 0.5.0 | 2026-10-19 | OpenRouter STT + summary; Proton Pass runtime key; removed local transcription path and heavy ML dependencies |
| 0.4.0 | 2026-10 | Meeting recorder (Swift) with capture-health monitoring; session pipeline |
| 0.3.0 | 2026-03 | Meeting-agnostic transcribe CLI: --type, --prompt-file |
| 0.2.0 | 2026-02 | LLM analysis, progress bars, timing summary |
| 0.1.0 | 2026-02 | Initial release - local transcription and diarization |
