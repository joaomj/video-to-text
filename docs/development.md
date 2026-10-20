# Development

## Requirements

Apple Silicon Mac, macOS 14.4+, Python 3.11+, UV, FFmpeg, Swift 6 toolchain.

## Checks

```bash
uv sync --group test
uv run ruff check .
uv run pytest tests/test_hallucinations.py tests/test_meeting_e2e.py tests/test_resources.py -q
swift build -c release --package-path recorder
```

The end-to-end test pins `ASR_ENGINE=torch` and fakes the model at the
`transcribe_track` boundary, plus the summary HTTP call at the `httpx`
boundary. No GPU and no `speech` CLI needed.

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests:

- `python` (ubuntu-latest): sync, ruff, pytest.
- `swift` (macos-15): release build of `recorder/`.

## Releases

Releases are tag-based. Push a `v*` tag and `.github/workflows/release.yml`
verifies the checks above, then creates the GitHub release with
automatically generated notes (categories in `.github/release.yml`).

```bash
git tag v0.7.0
git push origin v0.7.0
gh release view v0.7.0
```

Version the package in `pyproject.toml` before tagging.
