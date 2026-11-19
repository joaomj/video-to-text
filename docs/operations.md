# Operations

## Session folder

Each run writes to `~/Meetings/<date>/` (or `--out`/`--name`):

- `mic.wav` / `system.wav` — source recordings, never modified after capture.
- `session.json` — manifest: sources, recording outcome, health warnings,
  processing stages with input-fingerprint keys, final state.
- `work/` — normalized audio, 5-minute windows (`*-chunk-NN.wav`) with one
  checkpoint per window, per-track segments, merged segments (resumable).
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

## Machine capacity

Transcription needs about 1.3 GB per window. Processing is gated on what the
machine can afford. Before transcription the command reads the kernel memory
pressure, the reclaimable page count, and swap usage, then stops with a
message when:

- memory pressure is `warning` or `critical`, or
- less than 2.5 GB is reclaimable.

`--force` runs a stage regardless. CPU load is reported (above 1.5 per CPU at a
terminal) but never blocks a stage.

## Windows without speech

A window with no speech yields no segments. The log records
`no words in this window` with the model's stop reason, the window is
checkpointed empty, and processing continues. A muted microphone track is
therefore free and never fails a run.

## Recovery

```bash
meet --name "planning"  # record a new session
meeting process ~/Meetings/2026-10-19_153000
meeting status ~/Meetings/2026-10-19_153000
```

Processing resumes from valid completed work. A summary failure leaves the
transcript available and reports `incomplete`.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "FFmpeg not found" | `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux). |
| "No OpenRouter key" | Set `OPENROUTER_API_KEY` or `OPENROUTER_API_KEY_REF` in `.env` ([configuration](configuration.md#openrouter-key)), then retry `meeting process <dir>`. |
| "pass-cli could not read OPENROUTER_API_KEY_REF" | Run `pass-cli info`; log in if the session expired, and check the reference, then retry. |
| "The summary hit max_tokens" | The model ran out of output tokens, so no partial summary was saved. Retry `meeting process <dir>`; if it repeats, try another `--summary-model`. |
| "Summary request failed" | Check the network and the OpenRouter key and credit, then retry `meeting process <dir>`. |
| "Not starting transcription" | The machine is short of memory. Close applications, then retry; `--force` overrides the check. |
| "Recorder not found" | `swift build -c release --package-path recorder`, or set `MEETREC_BIN`. |
| First run is slow | Transcription weights (~1 GB) download once, then are reused. |
| Speakers labeled wrong | MOSS assigns `[S01]`, `[S02]` per window and the pipeline stitches labels across overlaps, so one speaker can appear under two labels. Edit the labels by hand in `transcript.md`. |
