# Operations

## Session folder

Each run writes to `~/Meetings/<date>/` (or `--out`/`--name`):

- `mic.wav` / `system.wav` — source recordings, never modified after capture.
- `session.json` — manifest: sources, recording outcome, health warnings,
  processing stages with input-fingerprint keys, final state.
- `work/` — normalized audio, per-track segments, merged segments (resumable).
- `transcript.md` — full transcript with timestamps and speaker labels.
- `summary.md` — main points, decisions, action items with owners/deadlines,
  open questions.
- `recording.log` / `processing.log` — persistent logs.
- `llama-server.log` — summary server output, capped at 512 KB.

## Outcomes

A run ends as one of:

- `complete` — transcript and summary saved.
- `degraded` — saved, but with capture warnings, missing speaker labels, or a
  sparse transcript for the amount of active audio.
- `incomplete` — transcript saved, summary failed. Retry with
  `meeting process`.
- `failed` — no usable audio or empty transcript. Recordings are kept.

## Recovery

```bash
meet --name "planning"  # record a new session
uv run --project ~/projects/video-to-text meeting process ~/Meetings/2026-10-19_153000
uv run --project ~/projects/video-to-text meeting status ~/Meetings/2026-10-19_153000
```

Processing resumes from valid completed work. A summary failure leaves the
transcript available and reports `incomplete`.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "FFmpeg not found" | `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux). |
| "Summary request failed" / connection refused | The server is auto-started by `record`/`process`; if it failed, start it manually: `llama-server -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M --port 8080 --chat-template-kwargs '{"enable_thinking": false}'`, then retry `meeting process`. |
| "Recorder not found" | `swift build -c release --package-path recorder`, or set `MEETREC_BIN`. |
| First run is slow | Model weights (~2 GB) download once, then are reused. |
| Speakers labeled wrong | MOSS assigns `[S01]`, `[S02]` per recording; edit manually. |
