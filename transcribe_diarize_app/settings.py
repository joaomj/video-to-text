from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    meetrec_bin: Path | None = None
    llama_server_bin: Path | None = None
    # Local summary server (llama.cpp, OpenAI-compatible protocol).
    # `meeting record` starts it during capture and stops it after processing.
    summary_model: str = "Qwen3.5-9B"
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_hf_repo: str = "unsloth/Qwen3.5-9B-GGUF:Q4_K_M"
    llm_ctx_size: int = 32768
    # Local transcription. Device is auto (CUDA, Apple GPU, then CPU),
    # or cuda, mps, or cpu. Revision pins the model for remote-code safety.
    moss_model_id: str = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    moss_revision: str | None = None
    moss_device: str = "auto"
    # Persistent weight storage outside ~/.cache, so cache cleanup cannot delete
    # the ~2 GB download. Overridable with MOSS_CACHE_DIR.
    moss_cache_dir: Path = Path.home() / ".local" / "share" / "meeting-weights"
