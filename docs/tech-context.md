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
    Local MOSS transcription per track (single pass, timestamps plus speakers)
      mic track → labeled "Me"
      system track → speaker labels [S01], [S02] → "Remote N"
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
| `moss.py` | Local transcription (MOSS-Transcribe-Diarize, GPU when available) | 16 kHz WAV | Segments with timestamps + speaker ids |
| `meeting_pipeline.py` | Stages, merging, completeness, transcript, summary | Session folder | `transcript.md`, `summary.md` |
| `summary.py` | OpenAI-compatible chat summary | Transcript text | Summary markdown |
| `vault.py` | Runtime key loading from Proton Pass | `pass://` reference | Key in memory only |
| `session.py` | Manifest, stage keys, atomic writes, process lock | Session folder | Resumable state |

### Speaker Labels

- The mic track is one known speaker: every segment is "Me".
- The system track is transcribed with diarization built in. Returned speaker
  labels ([S01], [S02]) map to "Remote 1", "Remote 2", ... in order of first appearance.
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
- Summary model: `SUMMARY_MODEL`, `SUMMARY_REASONING_EFFORT`.
- Summary endpoint: `LLM_BASE_URL` (OpenRouter by default; any
  OpenAI-compatible server for local summaries).
- Transcription: `MOSS_MODEL_ID`, `MOSS_REVISION`, `MOSS_DEVICE`
  (auto picks CUDA, Apple GPU, then CPU).
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

## Why Local MOSS-Transcribe-Diarize

Transcription runs on this machine instead of a cloud API. One 0.9B model
(Apache 2.0) returns timestamps plus speaker labels in a single pass, so no
separate diarization model is needed. GPU is used whenever available (NVIDIA
CUDA, then Apple GPU, then CPU). Only the summary still uses the cloud.

Options considered (Open ASR Leaderboard word error rates; prices per audio
hour; this Mac Mini M4 has 16 GB RAM):

| Model | WER | Price | Languages | Timestamps | Speakers | Fits 16 GB |
|-------|-----|-------|-----------|------------|----------|------------|
| MOSS-Transcribe-Diarize 0.9B | 5.52 | $0 | 50+ | yes | yes, built in | yes (~2 GB) |
| Qwen3-ASR-1.7B | 4.95 | $0 | 52 | via companion aligner | no | yes (~3 GB) |
| Parakeet TDT 0.6B v3 | 5.64 | $0 | 25 European | word plus segment | no | yes (~2 GB) |
| MAI-Transcribe-2 (cloud) | — | $0.10 | 60 | segment | yes | n/a |
| Whisper large-v3-turbo | 7.03 | $0.01 cloud / $0 local | 99+ | segment | no | yes |
| Muse Voice (cloud) | — | $0.18 | 25 | turn only | yes | n/a |
| Voxtral Mini 3B | 6.4 | $0 | major languages | segment | no | tight (~12 GB) |

Qwen3-ASR wins on accuracy but needs a companion model for timestamps and a
separate diarization step. Parakeet is the fastest (~92x realtime) with the
most permissive license (CC-BY-4.0) but also needs diarization. The cloud
options cost per meeting and send audio off the machine. MOSS is the only
option that transcribes, timestamps, and separates speakers in one local pass.

Verified on this machine (October 2026): 30 seconds of two-speaker audio
transcribes word-perfect with a correct speaker split in 9.5 seconds on the
Apple GPU and 14 seconds on CPU, at zero cost. WER numbers above are
leaderboard aggregates, not this meeting audio.

For reuse in another project: copy `transcribe_diarize_app/moss.py` and add
`torch` plus `moss-transcribe-diarize` (GitHub source, PyPI has no release) to
dependencies. The file documents its own interface.

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
| First run is slow | Model weights (~2 GB) download once | Later runs reuse the cache |

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| 0.5.0 | 2026-10-19 | OpenRouter STT + summary; Proton Pass runtime key; removed local transcription path and heavy ML dependencies |
| 0.4.0 | 2026-10 | Meeting recorder (Swift) with capture-health monitoring; session pipeline |
| 0.3.0 | 2026-03 | Meeting-agnostic transcribe CLI: --type, --prompt-file |
| 0.2.0 | 2026-02 | LLM analysis, progress bars, timing summary |
| 0.1.0 | 2026-02 | Initial release - local transcription and diarization |
