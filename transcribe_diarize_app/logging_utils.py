import logging
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta

RUN_ID = uuid.uuid4().hex[:8]
STEP_TIMES: dict[str, float] = {}
SECONDS_PER_MINUTE = 60


class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = RUN_ID  # type: ignore[attr-defined]
        return True


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(run_id)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in logging.root.handlers:
        handler.addFilter(RunIdFilter())
    return logging.getLogger("transcribe_diarize")


logger = configure_logging()


@contextmanager
def timed_step(name: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    logger.info("Starting: %s", name)
    yield
    elapsed = time.perf_counter() - start
    STEP_TIMES[name] = elapsed
    logger.info("Completed: %s in %.2fs", name, elapsed)


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total = int(td.total_seconds())
    hours, remainder = divmod(total, 3600)
    minutes, remainder_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02}:{minutes:02}:{remainder_seconds:02}"
    return f"{minutes:02}:{remainder_seconds:02}"


def format_duration(seconds: float) -> str:
    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    minutes = int(seconds // SECONDS_PER_MINUTE)
    remaining_seconds = int(seconds % SECONDS_PER_MINUTE)
    return f"{minutes}m {remaining_seconds}s"


def print_timing_summary() -> None:
    if not STEP_TIMES:
        return
    total = sum(STEP_TIMES.values())
    logger.info("%s", "=" * 50)
    logger.info("TIMING SUMMARY")
    logger.info("%s", "=" * 50)
    for step_name, elapsed in STEP_TIMES.items():
        percentage = (elapsed / total) * 100 if total > 0 else 0
        logger.info("  %-25s %10s (%5.1f%%)", step_name, format_duration(elapsed), percentage)
    logger.info("%s", "-" * 50)
    logger.info("  %-25s %10s", "TOTAL", format_duration(total))
    logger.info("%s", "=" * 50)
