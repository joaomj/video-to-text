# Transcribe-Diarize

## Start Here

This tool converts your job interview recordings into readable transcripts with speaker labels, then optionally uses AI to give you feedback on how you did.

### What It Does

Imagine you record a job interview on your phone. This tool:

1. **Pulls out the audio** from your video file (MP4/MOV)
2. **Identifies who is speaking** when (interviewer vs. you)
3. **Transcribes every word** with timestamps and shows progress bars
4. **Writes it all** to a clean markdown file in your current directory
5. **Shows timing** for each step so you know what's taking time
6. **Optionally analyzes** the interview with AI for feedback

### Quick Start

```bash
# 1. Install dependencies
pdm install

# 2. Add your API keys to .env file
echo "HF_ACCESS_TOKEN=your_huggingface_token" > .env
echo "LLM_API_KEY=your_opencode_zen_key" >> .env

# 3. Run it
pdm run transcribe interview.mp4
```

You'll get:
- `interview-transcript.md` - Full transcript with speaker labels
- `interview-analysis.md` - AI feedback on your performance (if you provided LLM_API_KEY)

## Why This Exists

Job interviews are high-stakes conversations where you want to improve. Most people:
- Forget exactly what they said
- Can't objectively assess their performance
- Miss subtle communication patterns

This tool gives you an objective record and AI-powered feedback so you can iterate and improve.

## How It Works (Architecture)

```
Your Video
    ↓
FFmpeg extracts clean audio (16kHz mono WAV)
    ↓
Pyannote AI figures out "Speaker A" vs "Speaker B" vs "Speaker C"
    ↓
MLX-Whisper (Apple Silicon optimized) converts speech to text
    ↓
System matches text segments to speakers by timing overlap
    ↓
Pretty markdown file with timestamps and speaker labels
    ↓
Gemini AI reads transcript and gives structured feedback
```

### The Components

| Step | What | Why This Choice |
|------|------|-----------------|
| Audio Extraction | FFmpeg | Industry standard, handles any video format |
| Speaker ID | Pyannote 3.1 | Best open-source diarization model |
| Transcription | MLX-Whisper | 3-5x faster than OpenAI on M-series Macs |
| Analysis | Gemini 3 Flash | Fast, cheap ($0.50/1M tokens), high quality |

## Requirements

### System Requirements

- **macOS with Apple Silicon** (M1/M2/M3) - MLX-Whisper requires this
- **Python 3.11+**
- **PDM** - Python dependency manager: `pip install pdm`
- **FFmpeg** - Audio/video tool: `brew install ffmpeg`
- **HuggingFace account** - Free, needed for Pyannote speaker model
- **~15GB free disk space** (for model downloads on first run)
- **8GB+ RAM** recommended

### Disk Space Requirements

The system automatically downloads AI models on first run:

| Model | Size | Purpose |
|-------|------|---------|
| Whisper Large v3 (MLX) | ~3GB | Speech-to-text transcription |
| Pyannote Diarization 3.1 | ~400MB | Speaker identification |
| Pyannote Segmentation 3.0 | ~300MB | Audio segmentation |
| **Total (first run)** | **~4-5GB** | **Minimum required** |

Models are cached in `~/.cache/` and reused across runs.

## Setup Guide

### 1. Get Your API Keys

**HuggingFace Token** (required for speaker detection):
1. Go to https://huggingface.co/settings/tokens
2. Create a token with "read" access
3. Accept the user agreement at https://huggingface.co/pyannote/speaker-diarization-3.1

**OpenCode Zen Key** (optional, for AI analysis):
1. Go to https://opencode.ai/auth
2. Sign in and add billing
3. Copy your API key from the dashboard

### 2. Configure Environment

```bash
# Create .env file
cat > .env << 'EOF'
HF_ACCESS_TOKEN=hf_your_token_here
LLM_API_KEY=oc_your_key_here
EOF
```

**Security note**: Never commit this file. It's already git-ignored by convention.

### 3. Install and Run

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
```

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
# English interview with custom output
pdm run transcribe interviews/candidate-john.mp4 --output ~/Documents/

# Portuguese meeting, skip analysis
pdm run transcribe meetings/team-sync.mov --language pt --skip-analysis

# Spanish presentation
pdm run transcribe presentations/quarterly-review.mp4 --language es
```

## Understanding the Output

### Transcript Format

```markdown
# Job Interview Transcript: candidate-john

**[00:15] SPEAKER_00:** Welcome to the interview. Can you tell me about your experience?

**[00:22] SPEAKER_01:** Thank you for having me. I've worked in software development for 5 years...

**[01:30] SPEAKER_00:** That's impressive. What technologies are you most comfortable with?
```

**Notes**:
- `[00:15]` = minutes:seconds timestamp
- `SPEAKER_00`, `SPEAKER_01` = automatically detected speakers (usually interviewer/candidate)
- `[inaudible]` = segments the AI couldn't understand (filtered for hallucinations)

### Analysis Format (Optional)

```markdown
# Interview Analysis

1. **Strengths**: What did the candidate do well?
   - Clear technical explanations
   - Good use of examples from past experience
   
2. **Areas for Improvement**: Where could they improve?
   - Could be more concise in responses
   - More specific metrics in project descriptions
   
3. **Communication Style**: How was their clarity, confidence, professionalism?
4. **Technical Answers**: If applicable, assess depth and accuracy
5. **Actionable Recommendations**: Specific tips for future interviews
```

## Key Decisions Explained

### Why MLX-Whisper Instead of OpenAI's API?

**The Tradeoff**:
- **MLX-Whisper** (what we use): Free, runs locally, super fast on Apple Silicon. Mac-only.
- **OpenAI API**: Works everywhere, but costs money (~$0.36/hour) and sends your audio to their servers.

**We chose MLX-Whisper because**:
- Job interviews are private - local processing is better
- Once set up, it's free forever
- 3-5x faster on M-series Macs (minutes vs hours)

**If you don't have a Mac**: You'd need to modify the code to use OpenAI Whisper API or another alternative.

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

**Problem**: You didn't set `LLM_API_KEY` in `.env`, or it's invalid.

**Check**: 
```bash
cat .env | grep LLM  # Should show your key
```

If missing, analysis is skipped gracefully - you still get the transcript.

### 4. "Speakers are labeled wrong"

**Problem**: Pyannote assigns `SPEAKER_00`, `SPEAKER_01` arbitrarily. It doesn't know which is interviewer vs candidate.

**Fix**: Manual fix in the markdown, or rename them based on context (who asks questions vs who answers).

### 5. Long videos take forever

**Problem**: Processing scales with video length.

**Reality check**:
- 30 min video = ~5-10 minutes processing
- 1 hour video = ~15-20 minutes
- Uses significant RAM (8GB+ recommended)

### 6. Transcript has weird repetitions

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

If using OpenCode Zen LLM analysis:

| Video Length | Input Tokens | Cost |
|--------------|--------------|------|
| 30 min interview | ~10,000 | $0.005 |
| 1 hour interview | ~20,000 | $0.01 |

Gemini 3 Flash is priced at $0.50/1M input tokens, $3/1M output tokens. Most interview analysis costs under a cent.

## Troubleshooting

### "ffmpeg not found"
Install ffmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux)

### "HF_ACCESS_TOKEN not found"
Create a `.env` file with: `HF_ACCESS_TOKEN=your_token_here`

### "Access to pyannote/speaker-diarization-3.1: GATED"
Visit the model page on Hugging Face and accept the user agreement

### "LLM_API_KEY not found"
The transcript will still be generated, but AI analysis will be skipped with a warning.

### Out of memory errors
- Close other applications
- Use shorter video segments
- Upgrade RAM if consistently hitting limits

## Supported Formats

- **Input**: MP4, MOV (case-insensitive)
- **Output**: Markdown (.md) - Transcript and optional Analysis

## Architecture Details

### File Structure

```
jobs/
├── transcribe_diarize.py   # Main script (274 lines)
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
- [OpenCode Zen](https://opencode.ai/zen/) - LLM API gateway
