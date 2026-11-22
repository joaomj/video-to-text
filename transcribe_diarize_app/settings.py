from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchored to the project root so `meeting` finds the key from any directory.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    meetrec_bin: Path | None = None
    # Transcription runs through the `speech` CLI (MOSS on MLX), installed by
    # `brew install speech`.
    speech_bin: Path | None = None
    speech_engine: str = "moss"
    speech_backend: str = "mlx"
    speech_model: str = "int5"
    speech_timeout_seconds: float = 3600.0
