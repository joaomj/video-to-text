# Development

## Requirements

Apple Silicon Mac, macOS 14.4+, Python 3.11+, UV, FFmpeg, Swift 6 toolchain.

## Checks

```bash
uv sync --group test
uv run ruff check .
uv run pytest tests/test_hallucinations.py tests/test_meeting_e2e.py -q
swift build -c release --package-path recorder
```

The end-to-end test fakes the model at the `transcribe_track` boundary and
the summary HTTP call at the `httpx` boundary. No GPU, money, or vault needed.

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests:

- `python` (ubuntu-latest): sync, ruff, pytest.
- `swift` (macos-15): release build of `recorder/`.

## Releases

Releases are tag-based. Push a `v*` tag and `.github/workflows/release.yml`
verifies the checks above, then creates the GitHub release with
automatically generated notes (categories in `.github/release.yml`).

```bash
git tag v0.5.0
git push origin v0.5.0
gh release view v0.5.0
```

Version the package in `pyproject.toml` before tagging.
