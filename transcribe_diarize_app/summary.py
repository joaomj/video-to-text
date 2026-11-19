"""Meeting summary generation through OpenRouter's OpenAI-compatible chat API."""

import os
import subprocess
import time

import httpx

from .constants import MEETING_SUMMARY_PROMPT
from .logging_utils import logger
from .settings import Settings

MAX_ATTEMPTS = 3
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
REQUEST_TIMEOUT_SECONDS = 300.0


class RetryableResponseError(RuntimeError):
    pass


def api_key(settings: Settings) -> str:
    """Returns the OpenRouter key from the environment or from Proton Pass."""
    if settings.openrouter_api_key:
        return settings.openrouter_api_key
    ref = settings.openrouter_api_key_ref
    if not ref:
        raise RuntimeError(
            "No OpenRouter key: set OPENROUTER_API_KEY, or OPENROUTER_API_KEY_REF "
            "to a Proton Pass reference, in .env."
        )
    try:
        result = subprocess.run(  # noqa: S603
            ["pass-cli", "item", "view", ref],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PROTON_PASS_AGENT_REASON": "video-to-text meeting summary"},
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not run pass-cli for OPENROUTER_API_KEY_REF: {exc}") from exc
    key = result.stdout.strip()
    if result.returncode != 0 or not key:
        raise RuntimeError(
            f"pass-cli could not read OPENROUTER_API_KEY_REF (exit {result.returncode}): "
            f"{result.stderr.strip()[:300]}. Check `pass-cli info` and the reference."
        )
    return key


def summarize(transcript: str, settings: Settings, model: str | None = None) -> str:
    """Returns the summary markdown. Retries transient failures and raises the final error."""
    key = api_key(settings)
    prompt = MEETING_SUMMARY_PROMPT.replace("{transcript}", transcript)
    payload: dict[str, object] = {
        "model": model or settings.summary_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8000,
    }
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = httpx.post(
                f"{settings.llm_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {key}"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code in RETRYABLE_STATUS:
                raise RetryableResponseError(f"HTTP {response.status_code}: {response.text[:300]}")
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") == "length":
                raise RuntimeError(
                    f"The summary hit max_tokens ({payload['max_tokens']}) and was cut off."
                )
            content = choice["message"]["content"]
            if not content or not content.strip():
                raise RuntimeError("The summary model returned an empty response.")
            return content.strip()
        except (httpx.TransportError, RetryableResponseError) as exc:
            last_error = exc
            logger.warning("Summary attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, exc)
            if attempt < MAX_ATTEMPTS:
                time.sleep(2**attempt)
    raise RuntimeError(f"Summary request failed after {MAX_ATTEMPTS} attempts: {last_error}")
