# Transcribe-Diarize

[![Version](https://img.shields.io/badge/version-0.4.0-blue.svg)]

Convert video recordings into readable transcripts with speaker labels. Works with interviews, team meetings, and professional conversations.

## What It Does

This tool takes your video file and:

1. **Extracts the audio** from MP4/MOV files
2. **Identifies who is speaking** using AI speaker detection
3. **Transcribes speech to text** with timestamps
4. **Produces a markdown transcript** with speaker labels
5. **Optionally analyzes** the content with AI for insights

## Quick Start

### Option 1: Native Apple Silicon (Fastest)

```bash
# 1. Install dependencies
uv sync --group macos

# 2. Create .env file with your HuggingFace token
echo "HF_ACCESS_TOKEN=hf_your_token_here" > .env

# 3. Run it
uv run transcribe meeting.mp4
```

**Requirements:** Apple Silicon Mac, Python 3.11+, UV (`brew install uv`), FFmpeg (`brew install ffmpeg`)

### Option 2: Docker (Cross-Platform)

```bash
# 1. Prepare folders and HuggingFace token
mkdir -p input output secrets
printf 'hf_your_token_here' > secrets/hf_access_token.txt

# 2. Build and run
docker compose build
docker compose run --rm transcribe /data/input/meeting.mp4 --output /data/output
```

**Requirements:** Docker Desktop or Docker Engine

### Option 3: Prebuilt GHCR Image (No Build)

```bash
# 1. Prepare folders and token
mkdir -p input output secrets
printf 'hf_your_token_here' > secrets/hf_access_token.txt

# 2. Pull and run
docker pull ghcr.io/joaomj/video-to-text:latest
TRANSCRIBE_DIARIZE_IMAGE=ghcr.io/joaomj/video-to-text:latest \
  docker compose -f docker-compose.ghcr.yml run --rm transcribe \
  /data/input/meeting.mp4 --output /data/output
```

## Usage

### Basic Transcription

```bash
# Transcribe with default settings (generic meeting type)
uv run transcribe meeting.mp4

# Job interview analysis
uv run transcribe interview.mp4 --type interview

# Specify language (default: English)
uv run transcribe meeting.mp4 --language pt

# Custom output directory
uv run transcribe meeting.mp4 --output ~/Documents/transcripts/

# Skip AI analysis (transcript only)
uv run transcribe meeting.mp4 --skip-analysis
```

### Analysis Only Mode

Already have a transcript? Run AI analysis on it:

```bash
uv run transcribe meeting-transcript.md --analyze-only
```

### Custom Analysis Prompts

Create a markdown file with `{transcript}` placeholder:

```markdown
# My Analysis

Summarize key points from this meeting:
{transcript}
```

Then use it:

```bash
uv run transcribe meeting.mp4 --prompt-file custom-prompt.md
```

## Meeting Types

| Type | Use Case | Analysis Focus |
|------|----------|----------------|
| `generic` (default) | Team meetings, discussions | Topics, decisions, action items |
| `interview` | Job interviews | Strengths, improvements, communication |

## Output

Two files are generated:

- `{filename}-transcript.md` - Full transcript with speaker labels and timestamps
- `{filename}-analysis.md` - AI analysis (if API key provided)

### Example Transcript

```markdown
# Meeting Transcript: team-sync

**[00:15] SPEAKER_00:** Let's start with the project update.

**[00:22] SPEAKER_01:** We completed the core module yesterday.
```

## Requirements

### API Keys

**HuggingFace Token** (required for speaker detection):
1. Go to https://huggingface.co/settings/tokens
2. Create a token with "read" access
3. Accept the user agreement at https://huggingface.co/pyannote/speaker-diarization-3.1

**Gemini API Key** (optional, for AI analysis):
- Get a key at https://aistudio.google.com/apikey

### Disk Space

Models are downloaded on first run (~3-5GB depending on backend) and cached for reuse.

| Model | Size | Purpose |
|-------|------|---------|
| Whisper Large v3 | ~3GB | Speech-to-text transcription |
| Pyannote Diarization | ~400MB | Speaker identification |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "FFmpeg not found" | `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux) |
| "HF_ACCESS_TOKEN not found" | Check your `.env` file has the token |
| "Access to pyannote... GATED" | Visit the HuggingFace model page and click "Access repository" |
| Speakers labeled wrong | Pyannote assigns SPEAKER_00, SPEAKER_01 arbitrarily. Edit manually. |
| Slow processing | First run downloads models. Later runs reuse cache. |

## Technical Documentation

For architecture details, API reference, and development information, see [docs/tech-context.md](docs/tech-context.md).

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Joao Marcos Visotaky Junior
