# Transcribe-Diarize

[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](CHANGELOG.md)

## Start Here

This tool converts your video recordings into readable transcripts with speaker labels, then optionally uses AI to analyze the content. Works with interviews, career discussions, team meetings, and any professional conversation.

### What It Does

Imagine you record a meeting on your phone. This tool:

1. **Pulls out the audio** from your video file (MP4/MOV)
2. **Identifies who is speaking** when (different speakers)
3. **Transcribes every word** with timestamps and shows progress bars
4. **Writes it all** to a clean markdown file in your current directory
5. **Shows timing** for each step so you know what's taking time
6. **Optionally analyzes** the meeting content with AI for insights

### Meeting Types

| Type | Use Case | Analysis Focus |
|------|----------|----------------|
| `generic` (default) | Team meetings, discussions | Key topics, decisions, action items, open questions |
| `interview` | Job interviews | Strengths, improvements, communication, technical answers |

### Custom Prompts

Use `--prompt-file` to provide your own analysis template (markdown file with `{transcript}` placeholder).

### Quick Start

Native on Apple Silicon:

```bash
# 1. Install dependencies
pdm install

# 2. Add your API keys to .env file
echo "HF_ACCESS_TOKEN=your_huggingface_token" > .env
echo "GEMINI_API_KEY=your_gemini_key" >> .env

# 3. Run it (generic meeting type by default)
pdm run transcribe meeting.mp4

# For job interviews
pdm run transcribe interview.mp4 --type interview
```

Cross-platform with Docker:

```bash
# 1. Create the HuggingFace secret file
mkdir -p secrets input output
printf 'hf_your_token_here' > secrets/hf_access_token.txt

# 2. Build the image
docker compose build

# 3. Put your video in ./input and run the container
docker compose run --rm transcribe /data/input/meeting.mp4 --output /data/output
```

Prebuilt GHCR image users can use `docker-compose.ghcr.yml`, which defaults to the persistent named volume `transcribe-diarize-cache` for model reuse across runs.

Run from GHCR without building locally:

```bash
# 1. Pull the published image
docker pull ghcr.io/<owner>/transcribe-diarize:latest

# 2. Prepare local folders and required Hugging Face secret
mkdir -p input output secrets
printf 'hf_your_token_here' > secrets/hf_access_token.txt

# 3. Create the persistent model cache volume once
docker volume create transcribe-diarize-cache

# 4. Run the container against a local file in ./input
docker run --rm \
  -v "$PWD/input:/data/input:ro" \
  -v "$PWD/output:/data/output" \
  -v transcribe-diarize-cache:/home/transcribe/.cache \
  -v "$PWD/secrets/hf_access_token.txt:/run/secrets/hf_access_token:ro" \
  -e HF_ACCESS_TOKEN_FILE=/run/secrets/hf_access_token \
  -e TRANSCRIPTION_BACKEND=faster \
  ghcr.io/<owner>/transcribe-diarize:latest \
  /data/input/meeting.mp4 --output /data/output
```

Optional analysis with an LLM key:

```bash
docker run --rm \
  -v "$PWD/input:/data/input:ro" \
  -v "$PWD/output:/data/output" \
  -v transcribe-diarize-cache:/home/transcribe/.cache \
  -v "$PWD/secrets/hf_access_token.txt:/run/secrets/hf_access_token:ro" \
  -e HF_ACCESS_TOKEN_FILE=/run/secrets/hf_access_token \
  -e GEMINI_API_KEY=your_gemini_key \
  -e TRANSCRIPTION_BACKEND=faster \
  ghcr.io/<owner>/transcribe-diarize:latest \
  /data/input/meeting.mp4 --output /data/output
```

You'll get:
- `meeting-transcript.md` - Full transcript with speaker labels
- `meeting-analysis.md` - AI analysis (if API key provided)

**Already have a transcript?** Run analysis only:
```bash
pdm run transcribe meeting-transcript.md --analyze-only
pdm run transcribe interview-transcript.md --analyze-only --type interview
```

## Why This Exists

Professional recordings are valuable but underutilized. Most people:
- Forget exactly what was said
- Can't objectively assess the conversation
- Miss subtle patterns and action items

This tool gives you an objective record and AI-powered analysis so you can review, learn, and follow up.

## How It Works (Architecture)

```
Your Video
    ↓
FFmpeg extracts clean audio (16kHz mono WAV)
    ↓
Pyannote AI figures out "Speaker A" vs "Speaker B" vs "Speaker C"
    ↓
Selected transcription backend converts speech to text
    ↓
System matches text segments to speakers by timing overlap
    ↓
Pretty markdown file with timestamps and speaker labels
    ↓
Gemini 3 Flash Preview reads transcript and gives structured feedback
```

### The Components

| Step | What | Why This Choice |
|------|------|-----------------|
| Audio Extraction | FFmpeg | Industry standard, handles any video format |
| Speaker ID | Pyannote 3.1 | Best open-source diarization model |
| Transcription | MLX-Whisper or faster-whisper | MLX stays fastest on Apple Silicon, faster-whisper unlocks Docker |
| Analysis | Gemini 3 Flash Preview | Fast, low-cost, good quality |

## Requirements

### System Requirements

- **Option A: Apple Silicon Mac** (M1/M2/M3) for the fastest native MLX backend
- **Option B: Docker** on macOS, Linux, or Windows for the cross-platform faster-whisper backend
- **Python 3.11+**
- **PDM** for native installs: `pip install pdm`
- **FFmpeg** for native installs: `brew install ffmpeg`
- **Docker Desktop** or Docker Engine for containerized use
- **HuggingFace account** - Free, needed for Pyannote speaker model
- **~15GB free disk space** (for model downloads on first run)
- **8GB+ RAM** recommended

### Disk Space Requirements

The system automatically downloads AI models on first run:

| Model | Size | Purpose |
|-------|------|---------|
| Whisper Large v3 (MLX) | ~3GB | Native Apple Silicon transcription |
| Whisper Large v3 (faster-whisper) | ~1.5GB | Cross-platform Docker transcription |
| Pyannote Diarization 3.1 | ~400MB | Speaker identification |
| Pyannote Segmentation 3.0 | ~300MB | Audio segmentation |
| **Total (first run)** | **~3-5GB** | **Depends on backend** |

Models are cached in `~/.cache/` and reused across runs.

For GHCR users, the recommended cache location is the named Docker volume `transcribe-diarize-cache`, which keeps model downloads between container runs by default.

## Setup Guide

### 1. Get Your API Keys

**HuggingFace Token** (required for speaker detection):
1. Go to https://huggingface.co/settings/tokens
2. Create a token with "read" access
3. Accept the user agreement at https://huggingface.co/pyannote/speaker-diarization-3.1

**Gemini API Key** (optional, for AI analysis):
1. Go to https://aistudio.google.com/apikey
2. Create an API key
3. Copy your API key

### 2. Configure Environment

```bash
# Create .env file
cat > .env << 'EOF'
HF_ACCESS_TOKEN=hf_your_token_here
GEMINI_API_KEY=your_gemini_key_here
EOF
```

**Security note**: Never commit this file. It's already git-ignored by convention.

### 3. Install and Run

#### Native Apple Silicon

```bash
# Install all dependencies (this creates a .venv managed by PDM)
pdm install

# Transcribe a video
pdm run transcribe ~/Movies/my-interview.mp4

# For other languages
pdm run transcribe interview.mp4 --language pt  # Portuguese

# Save to specific folder
pdm run transcribe interview.mp4 --output ~/Documents/interviews/

# Skip AI analysis (just get transcript)
pdm run transcribe interview.mp4 --skip-analysis

# Analyze an existing transcript file
pdm run transcribe interview-transcript.md --analyze-only
```

#### Docker

```bash
# Build once
docker compose build

# Run on a file placed in ./input
docker compose run --rm transcribe /data/input/interview.mp4 --output /data/output

# Force the Docker backend explicitly
docker compose run --rm transcribe /data/input/interview.mp4 --output /data/output --backend faster

# Skip analysis when you do not mount an LLM secret
docker compose run --rm transcribe /data/input/interview.mp4 --output /data/output --skip-analysis
```

#### GHCR Prebuilt Image

```bash
# Pull the published image
docker pull ghcr.io/<owner>/transcribe-diarize:latest

# Run with the provided GHCR compose file after setting the image reference
TRANSCRIBE_DIARIZE_IMAGE=ghcr.io/<owner>/transcribe-diarize:latest \
docker compose -f docker-compose.ghcr.yml run --rm transcribe \
  /data/input/interview.mp4 --output /data/output
```

The GHCR compose path uses the stable named volume `transcribe-diarize-cache`, so Whisper and Pyannote downloads are reused automatically across runs.

## Usage Examples

### Basic Usage

```bash
pdm run transcribe /path/to/video.mp4
```

Output: `{filename}-transcript.md` in current working directory (or `--output` directory)

### Specify Language

```bash
pdm run transcribe /path/to/video.mp4 --language pt
```

### All Options

```bash
# Generic meeting with custom output directory
pdm run transcribe meeting.mp4 --output ~/Documents/meetings/

# Job interview
pdm run transcribe interview.mp4 --type interview

# Career discussion with custom analysis
pdm run transcribe career-talk.mp4 --type generic --prompt-file my-prompt.md

# Meeting in another language
pdm run transcribe reunion.mp4 --language es

# Skip AI analysis (just get transcript)
pdm run transcribe meeting.mp4 --skip-analysis
```

### Custom Prompt File

Create a markdown file with your own analysis template. Use `{transcript}` as placeholder:

```markdown
# My Custom Analysis

Summarize this meeting:

1. Main Discussion Points
2. Decisions Made
3. Next Steps

Transcript:
{transcript}
```

Then run:
```bash
pdm run transcribe meeting.mp4 --prompt-file custom-prompt.md
```

## Understanding the Output

### Transcript Format

```markdown
# Meeting Transcript: team-sync

**[00:15] SPEAKER_00:** Let's start with the project update. What's the current status?

**[00:22] SPEAKER_01:** We completed the core module yesterday. Testing starts next week.

**[01:30] SPEAKER_00:** Great. Any blockers we should discuss?
```

**Notes**:
- `[00:15]` = minutes:seconds timestamp
- `SPEAKER_00`, `SPEAKER_01` = automatically detected speakers
- `[inaudible]` = segments the AI couldn't understand (filtered for hallucinations)

### Analysis Format (Generic Meeting)

```markdown
# Meeting Analysis

1. **Key Topics**: Main subjects discussed
   - Project status update
   - Testing timeline
   
2. **Decisions Made**: Any conclusions or agreements
   - Testing starts next week
   
3. **Action Items**: Tasks or follow-ups assigned
   - Prepare test cases
   
4. **Open Questions**: Unresolved items needing attention
   - Resource allocation for testing
```

### Analysis Format (Job Interview)

```markdown
# Interview Analysis

1. **Strengths**: What did the candidate do well?
   - Clear technical explanations
   - Good use of examples
   
2. **Areas for Improvement**: Where could they improve?
   - Could be more concise
   
3. **Communication Style**: How was their clarity, confidence, professionalism?
4. **Technical Answers**: If applicable, assess depth and accuracy
5. **Actionable Recommendations**: Specific tips for future interviews
```

## Key Decisions Explained

### Why Two Whisper Backends?

**The Tradeoff**:
- **MLX-Whisper**: Best native performance on Apple Silicon, but Mac-only.
- **faster-whisper**: Works in Docker across macOS, Linux, and Windows, but is slower on CPU than MLX.

**We kept both because**:
- Meeting recordings stay local in both paths
- Mac users keep the fastest experience
- Teams can now share one Docker workflow without requiring Apple Silicon

**Default behavior**: `auto` uses MLX on Apple Silicon and faster-whisper everywhere else.

### Why Generic as Default Meeting Type?

**Breaking Change in v0.3.0**: The default changed from `interview` to `generic`.

**Reasoning**:
- Most recordings are not interviews
- Generic template works for any professional conversation
- Interview-specific analysis requires explicit `--type interview`

**Migration**: If you want interview analysis, add `--type interview` to your command.

### Why PDM Instead of Regular pip?

PDM is a modern Python package manager. Compared to `pip` + `requirements.txt`:

**Advantages**:
- Lock file (`pdm.lock`) ensures exact same versions everywhere
- No "activate virtualenv" needed - `pdm run` handles it
- Better dependency resolution

**Learning curve**: Commands are slightly different (`pdm install` vs `pip install -r requirements.txt`), but worth it for reproducibility.

### Why Two Separate Output Files?

We generate `{video}-transcript.md` and `{video}-analysis.md` separately because:

1. **You might not want AI analysis** (costs money, or you just want the transcript)
2. **Transcript is factual**, analysis is opinionated - keep them separate
3. **Easy to share** just the transcript without your self-critique

## Common Pitfalls & Lessons

### 1. "It says FFmpeg not found"

**Problem**: FFmpeg isn't installed or not in your PATH.

**Fix**: 
```bash
brew install ffmpeg  # macOS
# or
sudo apt-get install ffmpeg  # Linux
```

### 2. "HuggingFace authentication failed"

**Problem**: Either your token is wrong, or you didn't accept the model agreement.

**Fix**:
1. Check token at https://huggingface.co/settings/tokens
2. Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and click "Access repository"

### 3. "LLM analysis didn't run"

**Problem**: You didn't set `GEMINI_API_KEY` (or fallback `LLM_API_KEY`) in `.env`, or the key is invalid.

**Check**: confirm `GEMINI_API_KEY`, `LLM_API_KEY`, `GEMINI_API_KEY_FILE`, or `LLM_API_KEY_FILE` is set.

If missing, analysis is skipped gracefully and you still get the transcript.

### 4. "Speakers are labeled wrong"

**Problem**: Pyannote assigns `SPEAKER_00`, `SPEAKER_01` arbitrarily. It doesn't know which is interviewer vs candidate.

**Fix**: Manual fix in the markdown, or rename them based on context (who asks questions vs who answers).

### 5. Long videos take forever

**Problem**: Processing scales with video length.

**Reality check**:
- 30 min video = ~5-10 minutes processing
- 1 hour video = ~15-20 minutes
- Uses significant RAM (8GB+ recommended)

**GHCR note**: The first run is always the slowest because the container downloads Whisper and Pyannote models into `transcribe-diarize-cache`. Later runs reuse that cache volume.

### 6. "It downloads the models every time"

**Problem**: The container is being run without the persistent named volume.

**Fix**:
```bash
docker volume create transcribe-diarize-cache

docker run --rm \
  -v transcribe-diarize-cache:/home/transcribe/.cache \
  ...
```

If the volume mount is missing, the container starts with a fresh empty cache each run.

If output writes fail on Linux, make sure your local `input`, `output`, and `secrets` folders exist and are writable by your user before starting the container.

### 7. Transcript has weird repetitions

**Problem**: Whisper sometimes hallucinates (repeats "thank you" endlessly, etc.).

**Our fix**: Built-in `is_hallucination()` filter detects common patterns and marks them `[inaudible]`.

## Log Output Format

Every run produces structured logs with timestamps and run IDs:

```
2026-02-04 17:23:45 [INFO] [a1b2c3d4] Extracting audio from interview.mp4...
2026-02-04 17:23:52 [INFO] [a1b2c3d4] Running diarization (Pyannote)...
2026-02-04 17:23:55 [INFO] [a1b2c3d4] Using device: mps
2026-02-04 17:25:30 [INFO] [a1b2c3d4] Analysis saved to interview-analysis.md
```

- **Timestamp**: When it happened
- **Level**: INFO (good), WARNING (fyi), ERROR (bad)
- **Run ID**: Unique 8-char code identifying this specific execution (for debugging)

## Progress Bars and Real-Time Feedback

The script shows visual progress for each major step:

```
Extracting audio from interview.mp4...
Audio extracted successfully
Diarization: 100%|████████████████| 3/3 [00:45<00:00, 15.03s/step]
Aligning speakers: 100%|████████| 156/156 [00:04<00:00, 35.2segment/s]
Sending to LLM: 100%|████████████| 1/1 [00:02<00:00,  2.12s/request]
```

**Steps with progress bars:**
1. **Audio Extraction** - Simple completion message
2. **Diarization** - 3-step progress (load → tensor → infer)
3. **Transcription** - MLX-Whisper's built-in progress
4. **Speaker Alignment** - Per-segment progress
5. **LLM Analysis** - Single API call progress

**Timing Summary** appears at the end:
```
==================================================
TIMING SUMMARY
==================================================
  Audio Extraction            2.5s (  3.2%)
  Speaker Diarization         45.2s ( 58.1%)
  Speech Transcription        25.1s ( 32.3%)
  Speaker Alignment            4.8s (  6.2%)
  LLM Analysis                 0.2s (  0.3%)
--------------------------------------------------
  Script Total                77.9s
==================================================
```

This helps you understand:
- Which steps take longest (usually diarization + transcription)
- Whether the process is still running or stuck
- How much time was spent on each operation

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

## Cost Breakdown

If using Gemini LLM analysis:

| Video Length | Input Tokens | Typical Cost |
|--------------|--------------|--------------|
| 30 min interview | ~10,000 | See current Gemini pricing |
| 1 hour interview | ~20,000 | See current Gemini pricing |

This script uses `gemini-3-flash-preview` by default. Check current pricing at Google AI Studio pricing docs.

## Troubleshooting

### "ffmpeg not found"
Install ffmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux)

### "HF_ACCESS_TOKEN not found"
Create a `.env` file with: `HF_ACCESS_TOKEN=your_token_here`

### "Access to pyannote/speaker-diarization-3.1: GATED"
Visit the model page on Hugging Face and accept the user agreement

### "GEMINI_API_KEY or LLM_API_KEY not found"
The transcript will still be generated, but AI analysis will be skipped with a warning.

### Out of memory errors
- Close other applications
- Use shorter video segments
- Upgrade RAM if consistently hitting limits

### "Weights only load failed" (PyTorch error)
This was a compatibility issue between Pyannote 3.4 and PyTorch 2.6+. 
**Fix**: Dependencies are now pinned to compatible versions (torch<2.6, huggingface-hub<1.0).
Run `pdm sync` to ensure you have the correct versions.

## Supported Formats

- **Input**: MP4, MOV (case-insensitive)
- **Output**: Markdown (.md) - Transcript and optional Analysis

## Architecture Details

### File Structure

```
jobs/
├── transcribe_diarize.py   # Main script
├── pyproject.toml          # Dependencies and PDM config
├── pdm.lock               # Exact versions locked
├── .env                   # Your secrets (never commit this!)
├── docs/
│   └── tech-context.md    # Detailed technical docs
├── interview-transcript.md      # Generated
└── interview-analysis.md        # Generated (optional)
```

### Key Code Patterns

**Settings Management**:
```python
class Settings(BaseSettings):
    hf_access_token: SecretStr  # Auto-masked in logs
    llm_api_key: SecretStr | None = None  # Optional
```

**Logging with Correlation ID**:
```python
logger.info("Processing video...")  # Auto-includes [a1b2c3d4]
```

**Hallucination Filtering**:
```python
if is_hallucination(text):
    text = "[inaudible]"  # Clean up Whisper noise
```

## Contributing

This is a personal tool, but improvements welcome:

1. Support non-Mac platforms (would need alternative Whisper)
2. Add progress bars for long operations
3. Export formats (SRT subtitles, JSON)
4. Speaker renaming (manually label interviewer vs candidate)
5. Interview question extraction

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Joao Marcos Visotaky Junior

## Acknowledgments

- [MLX-Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) - Apple's optimized Whisper implementation
- [Pyannote](https://github.com/pyannote/pyannote-audio) - Speaker diarization pipeline
- [Hugging Face](https://huggingface.co/) - Model hosting and access
- [Google Gemini API](https://ai.google.dev/) - LLM analysis provider
