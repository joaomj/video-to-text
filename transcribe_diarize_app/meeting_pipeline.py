"""Post-recording stages for a meeting session.

Each stage records a key built from its inputs and relevant configuration. A rerun
skips a stage whose key and outputs still match, so an interrupted run resumes
from completed work and a repeated run does not duplicate final documents.
"""

import json
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from . import moss, resources, speech_cli
from .constants import MEETING_PROMPTS, SUPPORTED_LANGUAGES
from .hallucinations import is_hallucination
from .local_llm import run_server
from .logging_utils import format_timestamp, logger
from .meeting_audio import (
    MEASURE_VERSION,
    OVERLAP_SECONDS,
    lift_level,
    measure,
    normalize,
    read_wav_header,
    split_chunks,
)
from .moss import transcribe_track
from .session import Session, now_iso, stage_key, write_atomic
from .settings import Settings
from .summary import summarize

ME = "Me"
REMOTE = "Remote"
# A mic segment that overlaps a system segment with similar words is the remote
# voice leaking from the speakers into the microphone, and is dropped.
ECHO_SIMILARITY = 0.6
# Normal speech runs at about 130 words per minute. Far fewer words per minute of
# active audio means the transcript is missing speech.
SPARSE_WORDS_PER_ACTIVE_MINUTE = 30
MIN_ACTIVE_SECONDS_FOR_SPARSE_CHECK = 30
MIN_ACTIVE_SECONDS_TOTAL = 5

LOCAL_SERVER_CMD = (
    "llama-server -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M "
    "--port 8080 --chat-template-kwargs '{\"enable_thinking\": false}'"
)
TRANSCRIPT_FILE = "transcript.md"
SUMMARY_FILE = "summary.md"


@dataclass
class Outcome:
    status: str = "complete"
    reasons: list[str] = field(default_factory=list)

    def degrade(self, reason: str) -> None:
        logger.warning("Degraded: %s", reason)
        self.reasons.append(reason)
        if self.status == "complete":
            self.status = "degraded"


@dataclass(frozen=True)
class Options:
    language: str
    meeting_type: str
    summary_model: str | None = None
    force: bool = False


def process_session(session: Session, settings: Settings, options: Options) -> Outcome:
    if options.language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language {options.language!r}")
    runner = StageRunner(session)
    outcome = Outcome()
    processing = session.data.setdefault("processing", {})
    processing.update(
        {"state": "processing", "started_at": now_iso(), "language": options.language}
    )
    session.data["state"] = "processing"
    session.save()
    try:
        tracks = validate_recording(session, outcome)
        evidence = runner.run(
            "normalize",
            [session.path(t["file"]) for t in tracks],
            {"measure_version": MEASURE_VERSION},
            [f"work/{t['name']}-16k.wav" for t in tracks],
            lambda: normalize_tracks(session, tracks),
        )
        segments = transcribe_tracks(session, runner, settings, options, evidence, outcome)
        check_completeness(segments, evidence, outcome)
        transcript_path = session.path(TRANSCRIPT_FILE)
        runner.run(
            "transcript",
            [session.path("work/segments.json")],
            {"type": options.meeting_type},
            [TRANSCRIPT_FILE],
            lambda: write_atomic(transcript_path, render_transcript(session, segments, options)),
        )
        # The summary model needs the memory the transcription model is holding.
        moss.release()
        write_summary(session, runner, settings, transcript_path, options, outcome)
    except Exception as exc:
        processing.update({"state": "failed", "outcome": "failed", "error": str(exc)})
        processing["finished_at"] = now_iso()
        session.data["state"] = "failed"
        session.save()
        raise
    processing.update(
        {"state": outcome.status, "outcome": outcome.status, "reasons": outcome.reasons}
    )
    processing.pop("error", None)
    processing["finished_at"] = now_iso()
    session.data["state"] = outcome.status
    session.save()
    return outcome


class StageRunner:
    def __init__(self, session: Session) -> None:
        self.session = session

    def run(
        self,
        name: str,
        inputs: list[Path],
        config: dict[str, Any],
        outputs: list[str],
        action: Callable[[], Any],
    ) -> Any:
        """Runs `action` unless a matching completed result exists. Returns stored or new result."""
        key = stage_key(inputs, config)
        stage = self.session.stages().get(name, {})
        if self.session.is_done(name, key, outputs):
            logger.info("Stage %s: reusing completed result", name)
            return stage.get("result")
        logger.info("Stage %s: starting", name)
        self.session.update_stage(name, status="running", key=key, started_at=now_iso())
        try:
            result = action()
        except Exception as exc:
            self.session.update_stage(name, status="failed", error=str(exc), finished_at=now_iso())
            raise RuntimeError(f"Stage {name} failed: {exc}") from exc
        self.session.update_stage(name, status="done", result=result, finished_at=now_iso())
        logger.info("Stage %s: done", name)
        return result


def validate_recording(session: Session, outcome: Outcome) -> list[dict[str, Any]]:
    tracks = session.data.get("sources")
    if not tracks:
        raise RuntimeError("The session lists no recorded sources; the recording did not start.")
    usable = []
    for track in tracks:
        path = session.path(track["file"])
        if not path.is_file():
            outcome.degrade(f"{track['name']} track is missing ({path.name}).")
            continue
        _, duration = read_wav_header(path)
        if duration < 1:
            outcome.degrade(f"{track['name']} track has no audio.")
            continue
        usable.append(track)
    if not usable:
        raise RuntimeError("No usable audio was recorded; nothing to process.")
    recording_outcome = session.data.get("recording_outcome")
    warnings = session.data.get("health_warnings", 0)
    if recording_outcome == "failed":
        outcome.degrade("The recorder reported a failure; parts of the meeting may be missing.")
    elif warnings:
        outcome.degrade(
            f"The recording raised {warnings} capture-health warning(s); see recording.log."
        )
    return usable


def normalize_tracks(session: Session, tracks: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = {}
    for track in tracks:
        destination = session.path(f"work/{track['name']}-16k.wav")
        normalize(session.path(track["file"]), destination)
        stats = lift_level(destination, measure(destination))
        evidence[track["name"]] = stats.__dict__
        logger.info(
            "%s: %.0fs, %ds active, noise floor %.0f dBFS, peak %.0f dBFS",
            track["name"],
            stats.duration_seconds,
            stats.active_seconds,
            stats.noise_floor_db,
            stats.peak_db,
        )
    return evidence


# Two chunks describe the same audio in their overlap; labels are compared only
# when their text matches at least this much.
STITCH_MIN_RATIO = 0.5


def stitch_speakers(
    previous: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    start: float,
    end: float,
) -> dict[str, str]:
    """Maps this chunk's speaker labels to the previous chunk's, using the overlap.

    Labels are local to a chunk, so the same voice arrives as [S01] in one window
    and [S02] in the next. The shared audio is transcribed in both, and matching
    text identifies which local label is the same person.
    """

    def window(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [s for s in items if s["end"] > start and s["start"] < end]

    scores: dict[tuple[str, str], list[float]] = {}
    for current in window(segments):
        for earlier in window(previous):
            if min(current["end"], earlier["end"]) <= max(current["start"], earlier["start"]):
                continue
            ratio = SequenceMatcher(None, current["text"], earlier["text"]).ratio()
            scores.setdefault((current["speaker_raw"], earlier["speaker_raw"]), []).append(ratio)

    mapping: dict[str, str] = {}
    taken: set[str] = set()
    ranked = sorted(scores.items(), key=lambda item: -max(item[1]))
    for (local, earlier_label), ratios in ranked:
        if max(ratios) < STITCH_MIN_RATIO or local in mapping or earlier_label in taken:
            continue
        mapping[local] = earlier_label
        taken.add(earlier_label)
    return mapping


def transcribe_file(path: Path, settings: Settings, language: str) -> list[dict[str, Any]]:
    """Transcribes one window with the configured engine."""
    if settings.asr_engine == "torch":
        return transcribe_track(path, settings)
    if settings.asr_engine == "speech":
        return speech_cli.transcribe(path, settings, language=language)
    raise ValueError(f"Unknown ASR_ENGINE {settings.asr_engine!r}; use 'speech' or 'torch'.")


def transcribe_chunked(
    session: Session, name: str, audio: Path, settings: Settings, language: str
) -> list[dict[str, Any]]:
    """Transcribes a track window by window, reusing finished windows on rerun."""
    chunks = split_chunks(audio, session.path("work"), name)
    collected: list[dict[str, Any]] = []
    previous: list[dict[str, Any]] = []
    for chunk in chunks:
        checkpoint = session.path(f"work/{name}-chunk-{chunk.index:02d}.json")
        logger.info("Chunk %d of %d: %s", chunk.index + 1, len(chunks), audio.name)
        if chunk.silent:
            logger.info("Chunk %d has no speech; skipping transcription", chunk.index + 1)
            write_atomic(checkpoint, [])
            continue
        if checkpoint.is_file():
            segments = json.loads(checkpoint.read_text(encoding="utf-8"))
            logger.info("Reusing chunk %d from %s", chunk.index + 1, checkpoint.name)
        else:
            raw = transcribe_file(chunk.path, settings, language)
            segments = [
                {
                    **segment,
                    "start": segment["start"] + chunk.start,
                    "end": segment["end"] + chunk.start,
                }
                for segment in raw
            ]
            write_atomic(checkpoint, json.dumps(segments))
        if chunk.index and previous:
            mapping = stitch_speakers(
                previous, segments, chunk.start, chunk.start + OVERLAP_SECONDS
            )
            if mapping:
                logger.info("Chunk %d speaker mapping: %s", chunk.index + 1, mapping)
            for segment in segments:
                segment["speaker_raw"] = mapping.get(segment["speaker_raw"], segment["speaker_raw"])
        # The last chunk owns everything to the end of the file; earlier chunks
        # stop where the next chunk's overlap starts.
        last = chunk.index == len(chunks) - 1
        own_end = float("inf") if last else chunk.own_end
        collected.extend(segment for segment in segments if segment["start"] < own_end)
        previous = segments
    return collected


def transcribe_tracks(
    session: Session,
    runner: StageRunner,
    settings: Settings,
    options: Options,
    evidence: dict[str, Any],
    outcome: Outcome,
) -> list[dict[str, Any]]:
    def transcribe(name: str) -> list[dict[str, Any]]:
        audio = session.path(f"work/{name}-16k.wav")
        segments = transcribe_chunked(session, name, audio, settings, options.language)
        kept = [s for s in segments if s["text"].strip() and not is_hallucination(s["text"])]
        write_atomic(session.path(f"work/{name}-segments.json"), json.dumps(kept))
        return kept

    model_config = {
        "language": options.language,
        # Part of the cache key: switching engines must not reuse another
        # engine's transcript.
        "stt_engine": settings.asr_engine,
        "speech_model": settings.speech_model if settings.asr_engine == "speech" else None,
        "stt_model": settings.moss_model_id,
        "stt_revision": settings.moss_revision,
        "device": settings.moss_device,
    }
    per_track: dict[str, list[dict[str, Any]]] = {}
    for name in evidence:
        audio = session.path(f"work/{name}-16k.wav")
        if evidence[name]["active_seconds"] == 0:
            logger.info("Skipping transcription of %s: no audible audio", name)
            per_track[name] = []
            continue
        runner.run(
            f"transcribe-{name}",
            [audio],
            model_config,
            [f"work/{name}-segments.json"],
            lambda name=name: len(transcribe(name)),
        )
        per_track[name] = json.loads(session.path(f"work/{name}-segments.json").read_text())

    remote = per_track.get("system", [])
    if remote:
        label_remote(remote, outcome)
    for segment in per_track.get("mic", []):
        segment["speaker"] = ME

    mic = drop_echo(per_track.get("mic", []), remote)
    merged = sorted(mic + remote, key=lambda s: s["start"])
    write_atomic(session.path("work/segments.json"), json.dumps(merged))
    return merged


def label_remote(segments: list[dict[str, Any]], outcome: Outcome) -> None:
    """Maps MOSS speaker labels ([S01], [S02]) to "Remote N" in order of first appearance."""
    names: dict[str, str] = {}
    for segment in segments:
        raw = segment.get("speaker_raw")
        if raw is None:
            continue
        names.setdefault(str(raw), f"{REMOTE} {len(names) + 1}")
        segment["speaker"] = names[str(raw)]
    unlabeled = [s for s in segments if "speaker" not in s]
    if unlabeled:
        outcome.degrade(
            "MOSS returned segments without speaker labels; those voices share one label."
        )
    for segment in unlabeled:
        segment["speaker"] = REMOTE


def drop_echo(mic: list[dict[str, Any]], remote: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    for segment in mic:
        echo = any(
            min(segment["end"], other["end"]) > max(segment["start"], other["start"])
            and SequenceMatcher(None, segment["text"].lower(), other["text"].lower()).ratio()
            >= ECHO_SIMILARITY
            for other in remote
        )
        if echo:
            logger.info("Dropping mic segment at %.1fs as speaker echo", segment["start"])
        else:
            kept.append(segment)
    return kept


def check_completeness(
    segments: list[dict[str, Any]], evidence: dict[str, Any], outcome: Outcome
) -> None:
    total_active = sum(e["active_seconds"] for e in evidence.values())
    if total_active < MIN_ACTIVE_SECONDS_TOTAL:
        raise RuntimeError(
            "No track contains audible audio. Check the microphone and that the meeting app "
            "played through this Mac; the recordings are kept in the session folder."
        )
    words = sum(len(s["text"].split()) for s in segments)
    if words == 0:
        raise RuntimeError(f"The transcript is empty although {total_active}s of audio is active.")
    if total_active >= MIN_ACTIVE_SECONDS_FOR_SPARSE_CHECK:
        rate = words / (total_active / 60)
        if rate < SPARSE_WORDS_PER_ACTIVE_MINUTE:
            outcome.degrade(
                f"The transcript looks sparse: {words} words for {total_active}s of active audio "
                f"({rate:.0f} words/min). Speech may be missing or in another language."
            )


def render_transcript(session: Session, segments: list[dict[str, Any]], options: Options) -> str:
    template = MEETING_PROMPTS.get(options.meeting_type, MEETING_PROMPTS["generic"])
    duration = session.data.get("duration_seconds", 0)
    lines = [
        f"# {template['transcript_header']}: {session.id}",
        "",
        f"- Recorded: {session.data.get('recording_started_at', 'unknown')}",
        f"- Duration: {format_timestamp(float(duration))}",
        f"- Language: {SUPPORTED_LANGUAGES[options.language]}",
        f'- Speakers: "{ME}" is the local microphone; "{REMOTE} N" are system-audio voices.',
        "",
    ]
    last_speaker = None
    for segment in segments:
        if segment["speaker"] != last_speaker:
            stamp = format_timestamp(segment["start"])
            lines.append(f"\n**[{stamp}] {segment['speaker']}:** {segment['text']}")
            last_speaker = segment["speaker"]
        else:
            lines[-1] += f" {segment['text']}"
    return "\n".join(lines).strip() + "\n"


def write_summary(
    session: Session,
    runner: StageRunner,
    settings: Settings,
    transcript: Path,
    options: Options,
    outcome: Outcome,
) -> None:
    summary_path = session.path(SUMMARY_FILE)
    summary_model = options.summary_model or settings.summary_model

    capacity = resources.measure()
    reason = resources.shortfall(capacity, resources.SUMMARY_HEADROOM_GB)
    if reason and not options.force:
        outcome.status = "incomplete"
        outcome.reasons.append(
            f"Summary skipped: {reason}. Close applications, then run "
            f"`meeting process {session.directory}` to generate only the summary. "
            "The transcript is saved."
        )
        logger.warning("Summary skipped: %s (%s)", reason, capacity.describe())
        return
    logger.info("Summary capacity: %s", capacity.describe())

    def generate() -> None:
        body = summarize(transcript.read_text(encoding="utf-8"), settings, summary_model)
        write_atomic(summary_path, f"# Meeting Summary: {session.id}\n\n{body}\n")

    # The summary model is the only stage that needs the LLM server. Loading it
    # here, and unloading it on exit, keeps capture and transcription free of a
    # multi-gigabyte resident model.
    server = (
        run_server(settings, session.directory)
        if settings.summary_server_autostart
        else nullcontext()
    )
    try:
        with server:
            runner.run("summary", [transcript], {"model": summary_model}, [SUMMARY_FILE], generate)
    except RuntimeError as exc:
        reason = str(exc)
        outcome.status = "incomplete"
        outcome.reasons.append(
            f"Summary not generated: {reason} Start the local summary server "
            f"({LOCAL_SERVER_CMD}), then retry with "
            f"`meeting process {session.directory}`. The transcript is saved."
        )
        logger.error("Summary not generated: %s", reason)
