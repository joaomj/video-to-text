from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hf_access_token: SecretStr | None = None
    hf_access_token_file: Path | None = None
    gemini_api_key: SecretStr | None = None
    gemini_api_key_file: Path | None = None
    llm_api_key: SecretStr | None = None
    llm_api_key_file: Path | None = None
    llm_model: str = "gemini-3-flash-preview"
    transcription_backend: str = "auto"
    whisper_model: str = "large-v3"
    mlx_model_repo: str = "mlx-community/whisper-large-v3-mlx"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute_type: str = "int8"

    def resolved_hf_access_token(self) -> SecretStr | None:
        return self.hf_access_token or self._secret_from_file(self.hf_access_token_file)

    def require_hf_access_token(self) -> SecretStr:
        token = self.resolved_hf_access_token()
        if token is None:
            raise RuntimeError("HF_ACCESS_TOKEN or HF_ACCESS_TOKEN_FILE is required")
        return token

    def resolved_llm_api_key(self) -> SecretStr | None:
        return (
            self.gemini_api_key
            or self._secret_from_file(self.gemini_api_key_file)
            or self.llm_api_key
            or self._secret_from_file(self.llm_api_key_file)
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
