import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transcribe_diarize_app.cli import build_parser, resolve_output_dir
from transcribe_diarize_app.constants import (
    HALLUCINATION_PATTERNS,
    MIN_HALLUCINATION_TEXT_LENGTH,
    MIN_REPETITION_WORDS,
    REPETITION_RATIO_THRESHOLD,
    SUPPORTED_LANGUAGES,
)
from transcribe_diarize_app.logging_utils import PipelineProgress
from transcribe_diarize_app.pipeline import is_hallucination


class ResolveOutputDirTests(unittest.TestCase):
    def test_default_uses_input_parent_when_writable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "video.mp4"
            input_path.write_text("", encoding="utf-8")
            result = resolve_output_dir(input_path, None)
            self.assertEqual(result, Path(tmp))

    def test_explicit_output_overrides_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path("/fake/video.mp4")
            output_arg = Path(tmp)
            result = resolve_output_dir(input_path, output_arg)
            self.assertEqual(result, output_arg)

    @patch("transcribe_diarize_app.cli.os.access", return_value=False)
    def test_falls_back_to_cwd_when_input_dir_not_writable(self, _mock_access: object) -> None:
        input_path = Path("/readonly/video.mp4")
        result = resolve_output_dir(input_path, None)
        self.assertEqual(result, Path.cwd())

    def test_creates_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "subdir" / "nested"
            input_path = Path(tmp) / "video.mp4"
            input_path.write_text("", encoding="utf-8")
            result = resolve_output_dir(input_path, output_path)
            self.assertTrue(result.exists())


class LanguageValidationTests(unittest.TestCase):
    def test_supported_languages_contain_en_and_pt(self) -> None:
        self.assertIn("en", SUPPORTED_LANGUAGES)
        self.assertIn("pt", SUPPORTED_LANGUAGES)

    def test_only_en_and_pt_are_supported(self) -> None:
        self.assertEqual(set(SUPPORTED_LANGUAGES.keys()), {"en", "pt"})

    def test_parser_rejects_invalid_language(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["video.mp4", "--language", "fr"])

    def test_parser_accepts_valid_languages(self) -> None:
        parser = build_parser()
        for lang in SUPPORTED_LANGUAGES:
            args = parser.parse_args(["video.mp4", "--language", lang])
            self.assertEqual(args.language, lang)


class IsHallucinationTests(unittest.TestCase):
    def test_short_text_is_hallucination(self) -> None:
        self.assertTrue(is_hallucination("ab"))

    def test_normal_text_is_not_hallucination(self) -> None:
        self.assertFalse(is_hallucination("Hello, how are you today?"))

    def test_repetitive_text_is_hallucination(self) -> None:
        self.assertTrue(is_hallucination("yeah yeah yeah yeah yeah yeah"))

    def test_known_pattern_is_hallucination(self) -> None:
        for pattern in HALLUCINATION_PATTERNS:
            self.assertTrue(is_hallucination(pattern), f"Pattern not detected: {pattern}")

    def test_three_char_text_is_not_hallucination(self) -> None:
        self.assertFalse(is_hallucination("abc"))


class PipelineProgressTests(unittest.TestCase):
    def test_step_updates_bar(self) -> None:
        progress = PipelineProgress(total=3, desc="Test")
        progress.step("Step 1")
        progress.step("Step 2")
        progress.step("Step 3")
        progress.close()

    def test_close_without_steps(self) -> None:
        progress = PipelineProgress(total=1, desc="Test")
        progress.close()


class ConstantsConsistencyTests(unittest.TestCase):
    def test_hallucination_thresholds_are_positive(self) -> None:
        self.assertGreater(MIN_HALLUCINATION_TEXT_LENGTH, 0)
        self.assertGreater(MIN_REPETITION_WORDS, 0)
        self.assertGreater(REPETITION_RATIO_THRESHOLD, 0)
        self.assertLess(REPETITION_RATIO_THRESHOLD, 1)

    def test_patterns_are_non_empty(self) -> None:
        for pattern in HALLUCINATION_PATTERNS:
            self.assertTrue(len(pattern) > 0)


if __name__ == "__main__":
    unittest.main()
