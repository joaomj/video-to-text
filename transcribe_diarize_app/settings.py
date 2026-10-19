from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .vault import DEFAULT_OPENROUTER_REF, read_pass_ref


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    meetrec_bin: Path | None = None
    # OpenRouter (cloud) by default. Point the base URLs at a local OpenAI-compatible
    # server (e.g. llama.cpp / whisper.cpp) to run local models instead.
    openrouter_api_key: SecretStr | None = None
    openrouter_api_key_file: Path | None = None
    # Vault reference used when no key or key file is configured. Empty disables it.
    openrouter_pass_ref: str | None = DEFAULT_OPENROUTER_REF
    summary_api_key: SecretStr | None = None
    summary_api_key_file: Path | None = None
    summary_model: str = "deepseek/deepseek-v4.1-flash-1017"
    summary_reasoning_effort: str = "high"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    # Local transcription. Device is auto (CUDA, Apple GPU, then CPU),
    # or cuda, mps, or cpu. Revision pins the model for remote-code safety.
    moss_model_id: str = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    moss_revision: str | None = None
    moss_device: str = "auto"

    def resolved_openrouter_api_key(self) -> SecretStr | None:
        if self.openrouter_api_key is not None:
            return self.openrouter_api_key
        try:
            file_secret = self._secret_from_file(self.openrouter_api_key_file)
        except RuntimeError:
            file_secret = None
        if file_secret is not None:
            return file_secret
        reference = self.openrouter_pass_ref or DEFAULT_OPENROUTER_REF
        if not reference:
            return None
        try:
            return SecretStr(read_pass_ref(reference))
        except RuntimeError as exc:
            raise RuntimeError(f"OPENROUTER_API_KEY is not set and {exc}") from exc

    def resolved_summary_api_key(self) -> SecretStr | None:
        return (
            self.summary_api_key
            or self._secret_from_file(self.summary_api_key_file)
            or self.resolved_openrouter_api_key()
        )

    def _secret_from_file(self, path: Path | None) -> SecretStr | None:
        if path is None:
            return None
        try:
            content = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise RuntimeError(f"Secret file not found: {path}") from exc
        except OSError as exc:
            raise RuntimeError(f"Could not read secret file: {path}") from exc
        if not content:
            raise RuntimeError(f"Secret file is empty: {path}")
        return SecretStr(content)
