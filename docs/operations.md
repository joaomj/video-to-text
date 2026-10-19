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
uv run meeting process ~/Meetings/2026-10-19_153000
uv run meeting status ~/Meetings/2026-10-19_153000
```

Processing resumes from valid completed work. A summary failure leaves the
transcript available and reports `incomplete`.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "FFmpeg not found" | `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux). |
| "No OPENROUTER_API_KEY..." | Run `pass-cli login`, or set `OPENROUTER_API_KEY`. |
| "Recorder not found" | `swift build -c release --package-path recorder`, or set `MEETREC_BIN`. |
| First run is slow | Model weights (~2 GB) download once, then are reused. |
| Speakers labeled wrong | MOSS assigns `[S01]`, `[S02]` per recording; edit manually. |
