import unittest

from transcribe_diarize_app.constants import (
    HALLUCINATION_PATTERNS,
    MIN_HALLUCINATION_TEXT_LENGTH,
    MIN_REPETITION_WORDS,
    REPETITION_RATIO_THRESHOLD,
    SUPPORTED_LANGUAGES,
)
from transcribe_diarize_app.hallucinations import is_hallucination


class LanguageValidationTests(unittest.TestCase):
    def test_supported_languages_contain_en_and_pt(self) -> None:
        self.assertIn("en", SUPPORTED_LANGUAGES)
        self.assertIn("pt", SUPPORTED_LANGUAGES)

    def test_only_en_and_pt_are_supported(self) -> None:
        self.assertEqual(set(SUPPORTED_LANGUAGES.keys()), {"en", "pt"})


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
