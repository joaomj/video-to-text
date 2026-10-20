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
    # Started only for the summary stage, then stopped.
    summary_model: str = "Qwen3.5-9B"
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_hf_repo: str = "unsloth/Qwen3.5-9B-GGUF:Q4_K_M"
    llm_ctx_size: int = 32768
    # Start and stop llama-server around the summary stage. Turn this off when
    # you keep your own server running on llm_base_url.
    summary_server_autostart: bool = True
    # Local transcription. Device is auto (CUDA, Apple GPU, then CPU),
    # or cuda, mps, or cpu. Revision pins the model for remote-code safety.
    moss_model_id: str = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    moss_revision: str | None = None
    moss_device: str = "auto"
    # Transcription engine: "speech" drives the MLX runtime installed by
    # `brew install speech`; "torch" keeps the in-process MOSS model.
    asr_engine: str = "speech"
    speech_bin: Path | None = None
    speech_engine: str = "moss"
    speech_backend: str = "mlx"
    speech_model: str = "int5"
    speech_timeout_seconds: float = 3600.0
    # Fallback-engine weights (~1.7 GB), stored outside ~/.cache so cache
    # cleanup cannot delete them. Overridable with MOSS_CACHE_DIR.
    moss_cache_dir: Path = Path.home() / ".local" / "share" / "meeting-weights"
