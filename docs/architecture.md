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
    FFmpeg: lift quiet tracks to -1 dBFS peak, at most +24 dB
    |
    Split each track into 5-minute windows with 30 s overlap
    Local MOSS transcription per window through the speech CLI (MLX)
      frames, timestamps and speaker labels in one pass
      windows without speech produce nothing and are logged
      speaker labels stitched across the overlap
      mic track maps to "Me"
      system track labels [S01], [S02] map to "Remote N"
    |
    Drop mic segments that echo remote speech
    |
    Merge by timestamp to transcript.md
    |
    Local chat completions summary to summary.md
      llama-server starts for this stage and stops after it
```

## Components

| Component | Purpose |
|-----------|---------|
| `recorder/` (Swift) | Capture plus health monitoring. Writes `mic.wav`, `system.wav`, `session.json`, `recording.log`. |
| `meeting_audio.py` | Normalizes tracks to 16 kHz mono and measures signal evidence. |
| `speech_cli.py` | Transcription engine. Runs the `speech` CLI (MOSS through MLX) once per window and parses its report into segments. Keeps generation within a token budget derived from window length. |
| `resources.py` | Machine capacity check before transcription and summary. |
| `meeting_pipeline.py` | Stage state, echo filtering, merging, completeness checks, transcript and summary output. |
| `summary.py` | Summary client for the local llama-server (Qwen3.5-9B). |
| `local_llm.py` | Starts llama-server for the summary stage, stops it afterwards. |
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

## Why local MOSS through MLX

Transcription runs on this machine. One model transcribes, timestamps, and
separates speakers in one model pass per window, so no audio leaves the Mac and
there is no per-minute cost.

Measured on the reference Mac mini (M4, 16 GB) with a real 40-minute meeting:

- Window-by-window MLX transcription: 6.4 minutes for the whole meeting, 1.3 GB peak.
- A window with no speech: 7 to 17 seconds, then logged and skipped.

Earlier revisions transcribed a whole track in one pass with torch, which
took 58.5 minutes, peaked near 10 GB, and pushed the machine into swap. That
path is removed; the comparison is in the README.

MLX decodes greedily without a repetition guard, so a window full of noise
makes it repeat timestamp markers until the token budget runs out. That is
why windows yield "no words" rather than an error, and why the token budget
scales with window length.

The weights (~1 GB, int5) download from Hugging Face on the first run and are
reused after. The summary weights (~6 GB) download on the first server start
the same way. Nothing leaves the Mac.
