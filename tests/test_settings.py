import tempfile
import unittest
from pathlib import Path

from pydantic import SecretStr

from transcribe_diarize_app.settings import Settings


class SettingsTests(unittest.TestCase):
    def test_prefers_inline_llm_key_over_file(self) -> None:
        settings = Settings(
            _env_file=None,
            gemini_api_key=SecretStr("inline-key"),
            gemini_api_key_file=Path("unused"),
        )

        self.assertEqual(settings.resolved_llm_api_key().get_secret_value(), "inline-key")

    def test_reads_hf_token_from_secret_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            secret_path = Path(temp_dir) / "hf_token.txt"
            secret_path.write_text("hf_test_token\n", encoding="utf-8")
            settings = Settings(_env_file=None, hf_access_token_file=secret_path)

            self.assertEqual(settings.require_hf_access_token().get_secret_value(), "hf_test_token")


if __name__ == "__main__":
    unittest.main()
