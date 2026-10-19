"""Meeting session folder: manifest with stage state, atomic writes, and a processing lock."""

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_NAME = "session.json"


class SessionError(RuntimeError):
    """A session folder is missing, invalid, or busy."""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def write_atomic(path: Path, content: str) -> None:
    """Writes to a temporary file and renames it, so an interrupted write never looks complete."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


CONTENT_HASH_LIMIT_BYTES = 20_000_000


def file_fingerprint(path: Path) -> list[Any]:
    """Hashes small files by content, so rewriting identical text keeps later stages valid.
    Large audio files use size and modification time, which change whenever they are rewritten."""
    stat = path.stat()
    if stat.st_size <= CONTENT_HASH_LIMIT_BYTES:
        return [path.name, hashlib.sha256(path.read_bytes()).hexdigest()]
    return [path.name, stat.st_size, stat.st_mtime_ns]


def stage_key(inputs: list[Path], config: dict[str, Any]) -> str:
    """Identifies a stage run by its input files and the configuration that affects its output."""
    fingerprint = {"inputs": [file_fingerprint(path) for path in inputs], "config": config}
    encoded = json.dumps(fingerprint, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


class Session:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.manifest_path = self.directory / MANIFEST_NAME
        if not self.manifest_path.is_file():
            raise SessionError(f"{self.directory} is not a meeting session (no {MANIFEST_NAME}).")
        try:
            self.data: dict[str, Any] = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SessionError(f"{self.manifest_path} is not valid JSON: {exc}") from exc
        self.work = self.directory / "work"
        self.work.mkdir(exist_ok=True)

    @property
    def id(self) -> str:
        return str(self.data.get("session_id", self.directory.name))

    def path(self, relative: str) -> Path:
        return self.directory / relative

    def save(self) -> None:
        write_atomic(self.manifest_path, json.dumps(self.data, indent=2, sort_keys=True) + "\n")

    def stages(self) -> dict[str, dict[str, Any]]:
        processing = self.data.setdefault("processing", {})
        return processing.setdefault("stages", {})

    def is_done(self, name: str, key: str, outputs: list[str]) -> bool:
        stage = self.stages().get(name, {})
        return (
            stage.get("status") == "done"
            and stage.get("key") == key
            and all(self.path(output).exists() for output in outputs)
        )

    def update_stage(self, name: str, **fields: Any) -> None:
        self.stages().setdefault(name, {}).update(fields)
        self.save()


@contextmanager
def processing_lock(directory: Path) -> Iterator[None]:
    """Prevents two processing runs on the same session, such as a repeated stop."""
    lock_path = directory / ".processing.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise SessionError(f"Another process is already processing {directory}.") from exc
    try:
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
