#!/bin/bash
# Installs the meeting recorder on macOS: system tools, Swift recorder, Python venv.
set -euo pipefail

have() { command -v "$1" >/dev/null 2>&1; }

echo "==> Checking Homebrew packages (uv, ffmpeg, llama.cpp)..."
if ! have brew; then
  echo "error: Homebrew is required: https://brew.sh" >&2
  exit 1
fi
for formula in uv ffmpeg llama.cpp; do
  if brew list "$formula" >/dev/null 2>&1; then
    echo "    $formula already installed"
  else
    echo "    installing $formula..."
    brew install "$formula"
  fi
done

echo "==> Checking Swift toolchain..."
if ! have swift; then
  echo "error: Swift is required (Xcode or Command Line Tools)" >&2
  exit 1
fi
swift --version 2>&1 | head -1

echo "==> Building the recorder..."
swift build -c release --package-path recorder
echo "    recorder: recorder/.build/release/meetrec"

echo "==> Installing Python dependencies..."
uv sync

echo "==> Local summary model..."
echo "    Qwen3.5-9B (Q4_K_M, ~6 GB) downloads on first server start, then is reused."
echo "    Start the summary server with:"
echo "      llama-server -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M --port 8080 --chat-template-kwargs '{"enable_thinking": false}'"

echo "==> Checking the meet alias..."
if grep -q "alias meet=" ~/.zshrc 2>/dev/null; then
  echo "    meet alias already present"
else
  cat >> ~/.zshrc << 'ALIAS'

# Meeting Recorder: start capturing on entering a meeting.
# Usage: meet [--language pt] [--name "planning"] [--mic "External"]
alias meet='uv run --project $HOME/projects/video-to-text meeting record'
ALIAS
  echo "    meet alias added (restart the terminal or run 'source ~/.zshrc')"
fi

echo ""
echo "Done. Record with: meet   (Ctrl-C stops, processing starts; the summary server starts and stops automatically)"
