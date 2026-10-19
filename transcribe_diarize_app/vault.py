"""Runtime secret loading from Proton Pass.

The OpenRouter API key lives in the vault, not in `.env`. It is read through
`pass-cli` into memory only: never a command-line argument, never logged, never
written to disk. The `pass://` reference names the vault, item, and field; the
default below is overridable with `OPENROUTER_PASS_REF`.
"""

import os
import shutil
import subprocess

DEFAULT_OPENROUTER_REF = "pass://mac-mini/openrouter-macmini-video-to-text/API Key"
REASON = "Loading the OpenRouter API key for meeting recording transcription and summary"

_cache: dict[str, str] = {}


def read_pass_ref(reference: str) -> str:
    """Returns the secret at a `pass://` reference, cached for the process lifetime."""
    if reference in _cache:
        return _cache[reference]
    binary = shutil.which("pass-cli")
    if binary is None:
        raise RuntimeError(
            "pass-cli is not installed, so the vault reference cannot be read. "
            "Install it with `brew install proton-pass-cli` and run `pass-cli login`."
        )
    environment = {
        **os.environ,
        "PROTON_PASS_AGENT_REASON": REASON,
    }
    try:
        completed = subprocess.run(  # noqa: S603
            [binary, "item", "view", reference],
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"pass-cli did not respond while reading the vault: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(
            "Could not read the vault reference. "
            f"Run `pass-cli login`, then retry. ({detail[-1] if detail else 'no detail'})"
        )
    secret = completed.stdout.strip()
    if not secret:
        raise RuntimeError(
            "The vault reference returned an empty value. "
            "Check the item field in Proton Pass, then retry."
        )
    _cache[reference] = secret
    return secret
