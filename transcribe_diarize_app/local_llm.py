"""Lifecycle for the local summary server (llama-server).

The summary stage starts the server, so a multi-gigabyte model is never
resident while a meeting is being recorded or transcribed. The server stops
when the summary finishes. A server that is already listening is reused and
left running. Server output goes to `llama-server.log` in the session folder.
"""

import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import IO
from urllib.parse import urlparse

import httpx

from .logging_utils import logger
from .settings import Settings

SERVER_LOG_NAME = "llama-server.log"
SERVER_LOG_CAP_BYTES = 512 * 1024
STARTUP_TIMEOUT_SECONDS = 300.0
HEALTH_POLL_SECONDS = 2.0
STOP_TIMEOUT_SECONDS = 15.0
# Qwen3.5 reasons in a thinking trace by default; for summaries that trace
# runs thousands of tokens and never reaches the answer in time.
THINKING_KWARGS = '{"enable_thinking": false}'


def server_origin(base_url: str) -> str:
    """Returns the scheme://host:port of the summary endpoint."""
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8080
    scheme = parsed.scheme or "http"
    return f"{scheme}://{host}:{port}"


def cap_file(path: Path, limit: int = SERVER_LOG_CAP_BYTES) -> None:
    """Truncates a log file to its last `limit` bytes, keeping whole lines."""
    try:
        if path.stat().st_size <= limit:
            return
    except OSError:
        return
    try:
        data = path.read_bytes()[-limit:]
        data = data.split(b"\n", 1)[-1]
        path.write_bytes(data)
    except OSError as exc:
        logger.warning("Could not cap %s: %s", path, exc)


def is_serving(base_url: str) -> bool:
    """Returns True when a summary server answers its health endpoint."""
    try:
        response = httpx.get(f"{server_origin(base_url)}/health", timeout=5.0)
        return response.status_code == httpx.codes.OK
    except httpx.HTTPError:
        return False


def open_log(session_dir: Path, log_path: Path) -> tuple[IO[bytes] | None, str | None]:
    """Opens the server log for appending.

    The recorder refuses to start in an existing folder, so this never
    creates the session folder: until the recorder creates it, output goes
    to a staged temp file. Returns the open file and the staged temp path
    (None when logging straight to the session).
    """
    try:
        if session_dir.is_dir():
            cap_file(log_path)
            return log_path.open("ab"), None
        fd, name = tempfile.mkstemp(prefix="llama-server-", suffix=".log")
        os.close(fd)
        return Path(name).open("ab"), name
    except OSError as exc:
        logger.warning("Cannot write %s; starting summary server without a log: %s", log_path, exc)
        return None, None


def close_log(
    log_file: IO[bytes] | None,
    log_path: Path,
    staged_path: str | None,
    session_dir: Path,
) -> None:
    """Closes the server log, moving staged output into the session."""
    if log_file is not None:
        log_file.close()
    if staged_path is None:
        cap_file(log_path)
        return
    try:
        if session_dir.is_dir():
            data = Path(staged_path).read_bytes()
            if data:
                with log_path.open("ab") as target:
                    target.write(data)
                cap_file(log_path)
    except OSError as exc:
        logger.warning("Could not keep the summary server log: %s", exc)
    finally:
        with suppress(OSError):
            Path(staged_path).unlink()


def wait_until_ready(process: subprocess.Popen[bytes], base_url: str, log_path: Path) -> None:
    """Blocks until the server answers, exits, or the startup timeout passes."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            logger.warning(
                "Summary server exited during startup; see %s. The transcript is still processed.",
                log_path,
            )
            return
        if is_serving(base_url):
            logger.info("Summary server ready at %s", base_url)
            return
        time.sleep(HEALTH_POLL_SECONDS)
    logger.warning(
        "Summary server not ready after %.0fs; continuing without it.", STARTUP_TIMEOUT_SECONDS
    )


def stop_server(process: subprocess.Popen[bytes]) -> None:
    """Terminates the server, killing it when it does not exit in time."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()


@contextmanager
def run_server(settings: Settings, session_dir: Path, wait_ready: bool = True) -> Iterator[None]:
    """Starts llama-server for the session, stopping it on exit.

    Reuses a server that is already listening. With `wait_ready=False` the
    server loads in the background and capture starts at once. Never fails:
    when the binary is missing or the server does not start in time, the
    session continues and the summary stage reports how to retry.
    """
    base_url = settings.llm_base_url.rstrip("/")
    if is_serving(base_url):
        logger.info("Summary server already running at %s; reusing it", base_url)
        yield
        return
    binary = str(settings.llama_server_bin or shutil.which("llama-server") or "llama-server")
    log_path = session_dir / SERVER_LOG_NAME
    log_file, staged_path = open_log(session_dir, log_path)
    try:
        process = subprocess.Popen(  # noqa: S603
            [
                binary,
                "-hf",
                settings.llm_hf_repo,
                "--port",
                str(urlparse(base_url).port or 8080),
                "-c",
                str(settings.llm_ctx_size),
                "--chat-template-kwargs",
                THINKING_KWARGS,
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        logger.warning("Summary server not started (%s); the transcript is still processed.", exc)
        close_log(log_file, log_path, staged_path, session_dir)
        yield
        return
    logger.info("Summary server starting (%s); log: %s", settings.llm_hf_repo, log_path)
    if wait_ready:
        wait_until_ready(process, base_url, log_path)
    try:
        yield
    finally:
        stop_server(process)
        close_log(log_file, log_path, staged_path, session_dir)
        logger.info("Summary server stopped")
