# Packaging Phase Plan

## Phase 1: Pluggable transcription backends

- Refactor the CLI into a small package with typed modules.
- Add backend selection with `auto`, `mlx`, and `faster` modes.
- Keep MLX as the native Apple Silicon path.
- Add a faster-whisper backend for Docker and cross-platform use.

Gate:
- `python -m unittest`
- `python -m compileall transcribe_diarize.py transcribe_diarize_app`

## Phase 2: Docker packaging

- Add a CPU-first Dockerfile with a non-root user.
- Add `docker-compose.yml` with read-only root filesystem, mounted cache, and Docker secrets.
- Add `.dockerignore` and `.env.example` for container-friendly setup.

Gate:
- `docker build -t transcribe-diarize .`

## Phase 3: Packaging docs

- Update onboarding docs for native and Docker workflows.
- Document backend selection, model caching, and secret file support.

Gate:
- Review docs for consistency with implemented commands and files.
