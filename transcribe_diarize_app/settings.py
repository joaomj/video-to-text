from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    meetrec_bin: Path | None = None
    # Summary through OpenRouter (OpenAI-compatible protocol). The model is
    # pinned to a dated release so summaries do not change under an alias.
    openrouter_api_key: str | None = None
    # Proton Pass reference (pass://vault/item/field) read with pass-cli when
    # openrouter_api_key is unset. Keep it in the git-ignored .env.
    openrouter_api_key_ref: str | None = None
    summary_model: str = "deepseek/deepseek-v4.1-flash-1017"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    # Transcription runs through the `speech` CLI (MOSS on MLX), installed by
    # `brew install speech`.
    speech_bin: Path | None = None
    speech_engine: str = "moss"
    speech_backend: str = "mlx"
    speech_model: str = "int5"
    speech_timeout_seconds: float = 3600.0
