import logging
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta

from tqdm import tqdm

RUN_ID = uuid.uuid4().hex[:8]
STEP_TIMES: dict[str, float] = {}
SECONDS_PER_MINUTE = 60


class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = RUN_ID  # type: ignore[attr-defined]
        return True


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


class PipelineProgress:
    def __init__(self, total: int, desc: str = "Processing") -> None:
        self._bar = tqdm(
            total=total,
            desc=desc,
            unit="step",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

    def step(self, label: str) -> None:
        self._bar.set_postfix_str(label)
        self._bar.update(1)

    def close(self) -> None:
        self._bar.close()


def configure_logging() -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    handler = TqdmLoggingHandler()
    handler.addFilter(RunIdFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(run_id)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(handler)
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
