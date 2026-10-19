"""End-to-end proof for `meeting process` with the model and network faked.

Why this test exists: it is the only automated proof that the stages connect —
normalize, transcribe both tracks, merge speaker labels, write the transcript
and summary, report the outcome, and resume completed stages. Unit tests on the
helpers cannot catch wiring or stage-key mistakes. The local MOSS model is
faked at the transcribe_track boundary (loading 2 GB per test run would be
absurd); the summary HTTP call is faked at the httpx boundary. No GPU, money,
or vault needed.
"""

import json
import math
import struct
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from transcribe_diarize_app.meeting_pipeline import Options, process_session
from transcribe_diarize_app.session import Session
from transcribe_diarize_app.settings import Settings

SYSTEM_SEGMENTS = [
    {"start": 0.0, "end": 1.0, "text": "Good morning everyone", "speaker_raw": "S01"},
    {"start": 2.0, "end": 3.0, "text": "Let us ship on Friday", "speaker_raw": "S02"},
]
MIC_SEGMENTS = [{"start": 0.0, "end": 1.0, "text": "I agree with Friday", "speaker_raw": "S01"}]
SUMMARY_BODY = "## Main points\n- The team agreed to ship on Friday."


def write_tone(path: Path) -> None:
    """4s mono WAV: tone, silence, tone, tone, so activity detection fires."""
    rate = 16000
    frames = []
    for second in range(4):
        for i in range(rate):
            sample = math.sin(2 * math.pi * 440 * i / rate) * 0.3 if second != 1 else 0.0
            frames.append(int(sample * 32767))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(struct.pack(f"<{len(frames)}h", *frames))


def make_session(directory: Path) -> Session:
    directory.mkdir(parents=True, exist_ok=True)
    write_tone(directory / "mic.wav")
    write_tone(directory / "system.wav")
    manifest = {
        "session_id": directory.name,
        "state": "recorded",
        "sources": [
            {"name": "mic", "file": "mic.wav"},
            {"name": "system", "file": "system.wav"},
        ],
        "recording_outcome": "complete",
        "health_warnings": 0,
        "duration_seconds": 4,
    }
    (directory / "session.json").write_text(json.dumps(manifest), encoding="utf-8")
    return Session(directory)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def fake_post(url: str, **_: object) -> FakeResponse:
    if url.endswith("/chat/completions"):
        return FakeResponse({"choices": [{"message": {"content": SUMMARY_BODY}}]})
    raise AssertionError(f"Unexpected URL: {url}")


def fake_transcribe(audio_wav: object, _settings: object) -> list[dict[str, object]]:
    name = Path(str(audio_wav)).stem
    if name.startswith("system"):
        return [dict(segment) for segment in SYSTEM_SEGMENTS]
    return [dict(segment) for segment in MIC_SEGMENTS]


def run_process(session: Session) -> tuple[object, int]:
    settings = Settings(openrouter_api_key="test-key")  # type: ignore[arg-type]
    options = Options(language="en", meeting_type="generic")
    transcribe_target = "transcribe_diarize_app.meeting_pipeline.transcribe_track"
    with (
        patch(transcribe_target, side_effect=fake_transcribe),
        patch("httpx.post", side_effect=fake_post) as post,
    ):
        outcome = process_session(session, settings, options)
        return outcome, post.call_count


class MeetingEndToEndTests(unittest.TestCase):
    def test_process_produces_transcript_and_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            session = make_session(Path(tmp) / "meeting")
            outcome, _ = run_process(session)

            self.assertEqual(outcome.status, "complete")
            transcript = (session.directory / "transcript.md").read_text(encoding="utf-8")
            self.assertIn("Remote 1", transcript)
            self.assertIn("Remote 2", transcript)
            self.assertIn("Me", transcript)
            self.assertIn("Good morning everyone", transcript)
            summary = (session.directory / "summary.md").read_text(encoding="utf-8")
            self.assertIn("ship on Friday", summary)
            self.assertEqual(Session(session.directory).data["state"], "complete")

    def test_rerun_reuses_completed_stages(self) -> None:
        with TemporaryDirectory() as tmp:
            session = make_session(Path(tmp) / "meeting")
            _, first_calls = run_process(session)
            self.assertGreater(first_calls, 0)

            session = Session(session.directory)  # reload persisted state
            outcome, second_calls = run_process(session)

            self.assertEqual(outcome.status, "complete")
            self.assertEqual(second_calls, 0)


if __name__ == "__main__":
    unittest.main()
