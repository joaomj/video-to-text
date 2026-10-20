# Meeting recording and processing requirements

## Status and goal

This document defines the target product behavior. It does not describe completed features.

The user records and processes a meeting through one command-line application on macOS or Linux.
The application captures system audio and microphone audio directly.
The workflow must not require another recording application or a manual recording export.

The primary reliability requirement is to detect capture problems during the meeting.
The user must not discover an unusable recording only after processing produces an almost empty transcript.

## User workflow

1. Shortly before a meeting, the user runs a shell command to start recording.
2. The application checks capture readiness and starts a persistent meeting session.
3. During the meeting, the application records system audio and microphone audio.
4. The application monitors capture health and reports problems while recording continues.
5. At the end of the meeting, the user runs a command or sends a documented stop signal.
6. The application stops recording and finalizes the captured audio.
7. The application records without loading models, and processing starts on request (`meeting process`, or `meet --process`).
8. The application saves the raw transcript and a separate meeting summary.
9. The application reports the outcome and the locations of the outputs and persistent logs.

Exact command names and stop semantics remain design decisions.
Starting post-meeting processing must not require a separate manual command after a normal stop.

## Recording

- Support macOS and Linux.
- Capture both system output and the selected microphone within the same meeting session.
- Capture remote participants through system audio, including when the user wears headphones.
- Check required permissions, capture sources, and output storage before reporting that recording has started.
- Report setup failures with an actionable explanation. Do not report success when a required source is unavailable.
- Identify the active session and the selected capture sources in the shell output and persistent logs.
- Save audio during recording. Do not keep the entire meeting only in memory until recording stops.
- Preserve usable captured audio after a recording or processing failure.
- Finalize audio before post-meeting processing reads it.

## Capture-health monitoring

Monitor capture health throughout recording, not only at startup or after the meeting.
Monitor the microphone and system audio independently, so one working source cannot conceal another failed source.

Detect and report at least these conditions:

- Missing audio frames or a stalled capture source.
- Persistently low volume.
- Persistently low signal power.
- Very low signal variance, including a constant or nearly constant signal.
- Capture errors, source disconnections, and audio-write failures.

Health checks must account for normal pauses and silence from an inactive participant.
A brief silent interval must not, by itself, mean that recording has failed.

When a problem persists, report a visible warning during recording.
Identify the affected source, explain the risk, and suggest a corrective action when one is known.
Do not rely on a log entry that the user sees only after the meeting.
Record the warning, its timing, and any recovery in the persistent logs.

Health thresholds and observation windows must be explicit and documented.
Their values, configuration interface, and warning delivery mechanism remain design decisions.
Signal measurements indicate capture health; they do not guarantee intelligible speech.

## Automatic processing

After recording stops, process the saved audio through explicit stages:

1. Validate the recording and check for missing or unusable audio.
2. Normalize the audio for the downstream models.
3. Perform speaker diarization and speech transcription.
4. Align speaker labels and timestamps with the transcript.
5. Save the raw transcript as a durable document.
6. Generate and save the meeting summary as a separate document.

The order and internal structure of diarization, transcription, and alignment can follow backend requirements.
Audio normalization must not replace the only retained source recording.

Check transcript completeness before reporting overall success.
An empty or suspiciously sparse transcript must produce a clear warning or failure outcome, not an unqualified success message.
Use recording duration and available audio-health evidence when evaluating this condition.
Exact completeness criteria remain a design decision.

## Idempotency and failure recovery

Recording, processing state, outputs, and logs must belong to a persistent meeting session.

- Persist stage state so processing can resume after an error or process restart.
- Preserve completed work when a later stage fails.
- Repeating a stop request must not create duplicate processing jobs for the same recording.
- Repeating processing for the same inputs and configuration must not create duplicate final documents.
- Reuse completed stage results only when their inputs and relevant configuration still match.
- Keep partial outputs distinct from completed outputs.
- Publish completed outputs atomically, so an interrupted write cannot appear complete.
- Bound automatic retries and preserve the final error with stage context.
- Do not convert an unexpected failure into empty data or a successful outcome.
- Permit recovery without recording the meeting again when usable source audio exists.
- Preserve the transcript if summary generation fails. Report that the workflow is incomplete and permit summary retry.

The application must distinguish successful completion, degraded output, and failure.
The exact session format, retry policy, and recovery commands remain design decisions.

## Persistent logs and status

Keep logs on disk for recording and every processing stage.
Logs must remain available after the command exits or the process restarts.

Include:

- Session identity and timestamps.
- Capture sources and capture-health events.
- Stage starts, completions, durations, and failures.
- Retry and recovery events.
- Output locations and the final outcome.

Report errors with enough context to identify the failed stage and the next recovery action.
Do not include credentials or API secrets in logs.
The user must be able to identify whether a session is recording, processing, complete, degraded, or failed.

## Documents

Save two distinct documents for each successfully processed meeting.

### Raw transcript

- Preserve the transcribed discussion rather than replacing it with a summary.
- Include timestamps and speaker labels.
- Mark uncertain or inaudible content instead of inventing missing speech.

### Meeting summary

Include:

- Main points discussed.
- Decisions made.
- Next steps and action items.
- Owners and deadlines when the discussion identifies them.
- Open questions and unresolved issues when present.

Base the summary on the transcript. Do not invent decisions, owners, or deadlines.
Keep the summary separate from the raw transcript, with both linked to the same meeting session.

## Acceptance outcomes

- On each supported operating system, the user can start and stop a meeting recording from the shell.
- The recording includes microphone speech and system audio without an external recording application.
- A persistent problem in either capture source produces a warning before the recording ends.
- Normal pauses do not cause repeated false failure reports.
- A normal stop saves the recording. Processing runs on request and produces separate transcript and summary documents.
- An interrupted processing run can resume from valid completed work without duplicate final documents.
- A summary failure leaves the transcript available and produces an explicit incomplete outcome.
- Missing audio or an almost empty transcript does not produce an unqualified success outcome.
- Persistent logs explain capture problems, processing failures, and recovery actions.

These outcomes define acceptance criteria, not evidence that the current implementation passes them.

## Open design decisions

Resolve these choices during implementation without weakening the requirements above:

- Start, stop, status, and recovery command interfaces.
- Foreground or background recording and warning delivery.
- Supported macOS versions, Linux audio systems, and capture dependencies.
- Audio source selection, storage format, and channel layout.
- Capture-health thresholds, observation windows, and completeness criteria.
- Processing backends for local transcription and local summary generation.
- Recording retention, output locations, and log rotation.

A graphical interface, live transcription, and automatic meeting detection are not required for this workflow.
