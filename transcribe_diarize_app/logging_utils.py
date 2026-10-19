import logging
import uuid
from pathlib import Path

RUN_ID = uuid.uuid4().hex[:8]


class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = RUN_ID  # type: ignore[attr-defined]
        return True


def configure_logging() -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.addFilter(RunIdFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(run_id)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(handler)
    return logging.getLogger("meeting")


logger = configure_logging()


def add_file_log(path: Path) -> logging.Handler:
    """Also writes every log record to `path`, so logs outlive the terminal session."""
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.addFilter(RunIdFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(run_id)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)
    return handler


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, remainder_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02}:{minutes:02}:{remainder_seconds:02}"
    return f"{minutes:02}:{remainder_seconds:02}"
