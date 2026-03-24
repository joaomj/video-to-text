# ruff: noqa: E402, I001
import warnings

warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*")

from transcribe_diarize_app.cli import main


if __name__ == "__main__":
    main()
