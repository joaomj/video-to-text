#!/bin/bash
# Installs the meeting recorder on macOS: system tools, Swift recorder, Python venv.
set -euo pipefail

have() { command -v "$1" >/dev/null 2>&1; }

echo "==> Checking Homebrew packages (uv, ffmpeg, proton-pass-cli)..."
if ! have brew; then
  echo "error: Homebrew is required: https://brew.sh" >&2
  exit 1
fi
for formula in uv ffmpeg proton-pass-cli; do
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

echo "==> Checking Proton Pass session..."
if pass-cli info >/dev/null 2>&1; then
  echo "    pass-cli session is active"
else
  echo "    no pass-cli session: run 'pass-cli login' once to enable runtime key loading"
fi

echo ""
echo "Done. Record with: uv run meeting record   (Ctrl-C stops, processing starts)"
