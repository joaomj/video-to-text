"""`meeting` command: record a meeting, then transcribe and summarize it."""

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .constants import SUPPORTED_LANGUAGES
from .local_llm import run_server
from .logging_utils import add_file_log, logger
from .meeting_pipeline import SUMMARY_FILE, TRANSCRIPT_FILE, Options, Outcome, process_session
from .session import Session, SessionError, processing_lock
from .settings import Settings

DEFAULT_LANGUAGE = "en"
DEFAULT_ROOT = Path.home() / "Meetings"
REPO_RECORDER = (
    Path(__file__).resolve().parent.parent / "recorder" / ".build" / "release" / "meetrec"
)
EXIT_CODES = {"complete": 0, "degraded": 3, "incomplete": 4}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meeting",
        description="Record a meeting (mic and system audio), then transcribe and summarize it.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_processing_options(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--language",
            "-l",
            default=os.environ.get("MEETING_LANGUAGE", DEFAULT_LANGUAGE),
            choices=sorted(SUPPORTED_LANGUAGES),
            help="Spoken language (default: MEETING_LANGUAGE or en)",
        )
        command.add_argument("--type", "-t", choices=["generic", "interview"], default="generic")
        command.add_argument(
            "--summary-model",
            default=os.environ.get("SUMMARY_MODEL"),
            help="Summary model (default: SUMMARY_MODEL or Qwen3.5-9B)",
        )

    record = commands.add_parser("record", help="Record now; process automatically after Ctrl-C")
    record.add_argument(
        "--out", type=Path, default=DEFAULT_ROOT, help="Sessions folder (default: ~/Meetings)"
    )
    record.add_argument("--name", help="Session folder name (default: date and time)")
    record.add_argument("--mic", help="Microphone name or UID (see `meeting devices`)")
    record.add_argument("--max-minutes", type=float, help="Stop automatically after N minutes")
    record.add_argument("--no-process", action="store_true", help="Only record")
    add_processing_options(record)

    process = commands.add_parser("process", help="Process or resume a recorded session")
    process.add_argument("session", type=Path)
    add_processing_options(process)

    status = commands.add_parser("status", help="Show the state of a session")
    status.add_argument("session", type=Path)

    commands.add_parser("devices", help="List microphones")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "record":
            sys.exit(record(args))
        if args.command == "process":
            sys.exit(process(args.session, args))
        if args.command == "status":
            sys.exit(show_status(args.session))
        if args.command == "devices":
            sys.exit(subprocess.call([str(recorder_binary(Settings())), "devices"]))  # noqa: S603
    except SessionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


def recorder_binary(settings: Settings) -> Path:
    candidate = settings.meetrec_bin or REPO_RECORDER
    if not candidate.is_file():
        raise SessionError(
            f"Recorder not found at {candidate}. Build it with "
            "`swift build -c release --package-path recorder`, or set MEETREC_BIN."
        )
    return candidate


def record(args: argparse.Namespace) -> int:
    settings = Settings()
    name = args.name or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    directory = (args.out.expanduser() / name).resolve()
    cmd = [str(recorder_binary(settings)), "record", "--session-dir", str(directory)]
    if args.mic:
        cmd += ["--mic", args.mic]
    if args.max_minutes:
        cmd += ["--max-minutes", str(args.max_minutes)]

    # The summary server starts while capture runs, so the model loads during
    # the meeting. It runs in its own process group to survive Ctrl-C, which
    # the recorder handles by finalizing its files.
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with run_server(settings, directory, wait_ready=False):
            try:
                code = subprocess.call(cmd)  # noqa: S603
            finally:
                signal.signal(signal.SIGINT, previous)
    finally:
        signal.signal(signal.SIGINT, previous)

    if not (directory / "session.json").is_file():
        print("Recording did not start; nothing to process.", file=sys.stderr)
        return code or 1
    if code != 0:
        print(
            f"The recorder exited with code {code}; processing whatever audio was saved.",
            file=sys.stderr,
        )
    if args.no_process:
        print(f"\nProcess later with: meeting process {directory}")
        return code
    return process(directory, args)


def process(directory: Path, args: argparse.Namespace) -> int:
    settings = Settings()
    session = Session(directory.expanduser())
    handler = add_file_log(session.path("processing.log"))
    print(f"\nProcessing session {session.id} (language: {SUPPORTED_LANGUAGES[args.language]})")
    try:
        with run_server(settings, session.directory), processing_lock(session.directory):
            session = Session(session.directory)  # reload under the lock
            outcome = process_session(
                session,
                settings,
                Options(args.language, args.type, args.summary_model),
            )
    except SessionError:
        raise
    except Exception as exc:
        logger.exception("Processing failed")
        print(f"\nProcessing FAILED: {exc}", file=sys.stderr)
        print(f"  Recordings are kept in {session.directory}", file=sys.stderr)
        print(f"  Log: {session.path('processing.log')}", file=sys.stderr)
        print(f"  Retry: meeting process {session.directory}", file=sys.stderr)
        return 1
    finally:
        handler.close()
    report(session, outcome)
    return EXIT_CODES[outcome.status]


def report(session: Session, outcome: Outcome) -> None:
    print(f"\nMeeting processing {outcome.status.upper()}")
    for reason in outcome.reasons:
        print(f"  - {reason}")
    for label, name in (("Transcript", TRANSCRIPT_FILE), ("Summary", SUMMARY_FILE)):
        path = session.path(name)
        if path.is_file():
            print(f"  {label}: {path}")
    print(f"  Logs: {session.path('recording.log')}, {session.path('processing.log')}")


def show_status(directory: Path) -> int:
    session = Session(directory.expanduser())
    data = session.data
    print(f"Session {session.id}: {data.get('state', 'unknown')}")
    print(
        f"  Recording: {data.get('recording_outcome', 'in progress or interrupted')}, "
        f"{data.get('duration_seconds', 0):.0f}s, {data.get('health_warnings', 0)} health warnings"
    )
    processing = data.get("processing", {})
    for name, stage in processing.get("stages", {}).items():
        detail = f" ({stage['error']})" if stage.get("error") else ""
        print(f"  Stage {name}: {stage.get('status')}{detail}")
    for reason in processing.get("reasons", []):
        print(f"  - {reason}")
    if os.environ.get("MEETING_STATUS_JSON"):
        print(json.dumps(data, indent=2))
    return 0
