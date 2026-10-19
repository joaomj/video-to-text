# Architecture

A Swift recorder captures microphone and system audio on macOS. A Python
pipeline transcribes both tracks locally with MOSS-Transcribe-Diarize,
merges them into one transcript, and writes a meeting summary through a
local chat model (Qwen3.5-9B via llama-server).

## Data flow

```
meetrec record (Swift, macOS)
    Mic track ──► mic.wav (16-bit mono WAV)
    System tap ─► system.wav (16-bit mono WAV)
    Health monitor ─► terminal warnings + recording.log
    Ctrl-C ─► finalize + session.json ("recorded")
    |
meeting process (Python)
    Validate recording (headers, duration, recorder outcome)
    |
    FFmpeg: normalize each track to work/<name>-16k.wav (sources untouched)
    |
    Local MOSS transcription per track (timestamps plus speaker labels)
      mic track maps to "Me"
      system track labels [S01], [S02] map to "Remote N"
    |
    Drop mic segments that echo remote speech
    |
    Merge by timestamp to transcript.md
    |
    Local chat completions summary to summary.md
    Summary server stops; llama-server.log stays capped in the session
```

## Components

| Component | Purpose |
|-----------|---------|
| `recorder/` (Swift) | Capture plus health monitoring. Writes `mic.wav`, `system.wav`, `session.json`, `recording.log`. |
| `meeting_audio.py` | Normalizes tracks to 16 kHz mono and measures signal evidence. |
| `moss.py` | Local transcription. One 0.9B model returns segments with timestamps plus speaker labels in a single pass. Uses CUDA, then Apple GPU, then CPU. |
| `meeting_pipeline.py` | Stage state, echo filtering, merging, completeness checks, transcript and summary output. |
| `summary.py` | Summary client for the local llama-server (Qwen3.5-9B). |
| `local_llm.py` | Starts llama-server during capture, stops it after processing. |
| `session.py` | Manifest, stage keys, atomic writes, process lock. |
| `hallucinations.py` | Noise-segment filter. Uncertain speech is marked `[inaudible]`, never invented. |

## Speaker labels

- The mic track is one known speaker. Every segment is "Me".
- The system track carries diarization labels ([S01], [S02]). Labels map to
  "Remote 1", "Remote 2", in order of first appearance.
- A mic segment that overlaps a system segment with similar words is echo
  leaking into the microphone. The pipeline drops it from "Me".

## Stage state

Each stage records a key from its inputs plus relevant configuration. A rerun
skips a stage whose key and outputs still match, resumes interrupted work,
and never duplicates final documents. A file lock prevents two processing
runs on one session.

## Why local MOSS

Transcription runs on this machine. One model transcribes, timestamps, and
separates speakers in a single local pass, so no audio leaves the Mac and
there is no per-minute cost. The weights (~2 GB) download from Hugging Face
on the first run and are reused after. The summary weights (~6 GB) download
on the first server start the same way. Nothing leaves the Mac.

For reuse in another project: copy `transcribe_diarize_app/moss.py` and add
`torch` plus `moss-transcribe-diarize` (GitHub source) to dependencies.
