import unittest
from unittest.mock import patch

from transcribe_diarize_app.settings import Settings
from transcribe_diarize_app.transcription import (
    MlxTranscriber,
    create_transcriber,
    normalize_segments,
)


class TranscriptionTests(unittest.TestCase):
    def test_normalize_segments_skips_invalid_items(self) -> None:
        segments = normalize_segments(
            [
                {
                    "start": 0,
                    "end": 1.5,
                    "text": "Hello",
                    "words": [{"start": 0, "end": 0.5, "word": "Hello"}],
                },
                {"start": 2},
                "invalid",
            ]
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "Hello")
        self.assertEqual(segments[0]["words"][0]["word"], "Hello")

    @patch("transcribe_diarize_app.transcription.find_spec", return_value=object())
    @patch("transcribe_diarize_app.transcription.should_use_mlx", return_value=True)
    def test_auto_backend_prefers_mlx_on_apple_silicon(self, *_args: object) -> None:
        backend_name, backend = create_transcriber(Settings(_env_file=None))

        self.assertEqual(backend_name, "mlx")
        self.assertIsInstance(backend, MlxTranscriber)

    @patch("transcribe_diarize_app.transcription.find_spec", return_value=None)
    @patch("transcribe_diarize_app.transcription.should_use_mlx", return_value=True)
    def test_auto_backend_falls_back_to_faster_when_mlx_missing(self, *_args: object) -> None:
        backend_name, _backend = create_transcriber(Settings(_env_file=None))

        self.assertEqual(backend_name, "faster")


if __name__ == "__main__":
    unittest.main()
