# Transcription System

A complete transcription and speaker diarization system for video files (MP4/MOV). Extracts audio, identifies speakers, and generates structured markdown transcripts.

## Features

- **One-command operation**: Simply pass a video file path
- **Automatic audio extraction**: Converts video to optimized WAV format using ffmpeg
- **Speaker diarization**: Identifies and labels different speakers using Pyannote
- **Accurate transcription**: Uses MLX-Whisper with hallucination detection
- **Smart output**: Saves transcript as `{filename}-transcript.md` in the same folder
- **Multi-language support**: Configurable language for transcription

## Requirements

### System Requirements

- **macOS** with Apple Silicon (M1/M2/M3) or Linux with GPU support
- **Python 3.11+**
- **ffmpeg** installed on your system
- **~15GB free disk space** (for model downloads)
- **8GB+ RAM** recommended

### Disk Space Requirements

The system automatically downloads AI models on first run. Ensure you have sufficient space:

| Model | Size | Purpose |
|-------|------|---------|
| Whisper Large v3 (MLX) | ~3GB | Speech-to-text transcription |
| Pyannote Diarization 3.1 | ~400MB | Speaker identification |
| Pyannote Segmentation 3.0 | ~300MB | Audio segmentation |
| **Total (first run)** | **~4-5GB** | **Minimum required** |

Models are cached in `~/.cache/` and reused across runs.

### Hugging Face Access

This project requires access to gated Hugging Face models:

1. Create a Hugging Face account: https://huggingface.co/join
2. Request access to:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Generate an access token at: https://huggingface.co/settings/tokens

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd transcription-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install ffmpeg (if not already installed):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg
```

5. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your HF_ACCESS_TOKEN
```

## Usage

### Basic Usage

```bash
python transcribe_diarize.py /path/to/video.mp4
```

Output: `/path/to/video-transcript.md`

### Specify Language

```bash
python transcribe_diarize.py /path/to/video.mp4 --language pt
```

### Examples

```bash
# English interview
python transcribe_diarize.py interviews/candidate-john.mp4

# Portuguese meeting
python transcribe_diarize.py meetings/team-sync.mov --language pt

# Spanish presentation
python transcribe_diarize.py presentations/quarterly-review.mp4 --language es
```

## Output Format

Transcripts are saved as markdown files with the following structure:

```markdown
# Job Interview Transcript: candidate-john

**[00:15] SPEAKER_00:** Welcome to the interview. Can you tell me about your experience?

**[00:22] SPEAKER_01:** Thank you for having me. I've worked in software development for 5 years...

**[01:30] SPEAKER_00:** That's impressive. What technologies are you most comfortable with?
```

## Supported Formats

- **Input**: MP4, MOV (case-insensitive)
- **Output**: Markdown (.md)

## Performance

Typical processing times (Apple M2 Pro, 16GB RAM):

| Video Length | Processing Time |
|--------------|-----------------|
| 5 minutes | ~30-45 seconds |
| 30 minutes | ~3-5 minutes |
| 1 hour | ~8-12 minutes |

Processing time depends on:
- Video length
- Number of speakers
- Audio quality
- Hardware (Apple Silicon significantly faster)

## Troubleshooting

### "ffmpeg not found"
Install ffmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux)

### "HF_ACCESS_TOKEN not found"
Create a `.env` file with: `HF_ACCESS_TOKEN=your_token_here`

### "Access to pyannote/speaker-diarization-3.1: GATED"
Visit the model page on Hugging Face and accept the user agreement

### Out of memory errors
- Close other applications
- Use shorter video segments
- Upgrade RAM if consistently hitting limits

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Joao Marcos Visotaky Junior

## Acknowledgments

- [MLX-Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) - Apple's optimized Whisper implementation
- [Pyannote](https://github.com/pyannote/pyannote-audio) - Speaker diarization pipeline
- [Hugging Face](https://huggingface.co/) - Model hosting and access
