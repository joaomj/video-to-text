"""Meeting summary generation through a local OpenAI-compatible chat server.

The default endpoint is llama-server on this Mac serving Qwen3.5-9B.
Start it with: llama-server -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M --port 8080
"""

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


def summarize(transcript: str, settings: Settings, model: str | None = None) -> str:
    """Returns the summary markdown. Retries transient failures and raises the final error."""
    prompt = MEETING_SUMMARY_PROMPT.replace("{transcript}", transcript)
    payload: dict[str, object] = {
        "model": model or settings.summary_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
    }
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = httpx.post(
                f"{settings.llm_base_url.rstrip('/')}/chat/completions",
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code in RETRYABLE_STATUS:
                raise RetryableResponseError(f"HTTP {response.status_code}: {response.text[:300]}")
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise RuntimeError("The summary model returned an empty response.")
            return content.strip()
        except (httpx.TransportError, RetryableResponseError) as exc:
            last_error = exc
            logger.warning("Summary attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, exc)
            if attempt < MAX_ATTEMPTS:
                time.sleep(2**attempt)
    raise RuntimeError(f"Summary request failed after {MAX_ATTEMPTS} attempts: {last_error}")
