"""Small server-side client for OpenRouter's OpenAI-compatible chat completions API."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import (
    MAX_NEW_TOKENS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MAX_RETRIES,
    OPENROUTER_MODEL,
    OPENROUTER_TIMEOUT_SECONDS,
    TEMPERATURE,
)

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Safe category-only error from the configured inference provider."""

    def __init__(self, category: str, retryable: bool = False) -> None:
        super().__init__(category)
        self.category = category
        self.retryable = retryable


@dataclass(frozen=True)
class Completion:
    text: str
    model: str


def chat_completion(messages: list[dict[str, str]]) -> Completion:
    """Call the configured OpenRouter endpoint."""
    if not OPENROUTER_API_KEY:
        raise ProviderError("OPENROUTER_API_KEY is not configured")
    if not OPENROUTER_MODEL:
        raise ProviderError("OPENROUTER_MODEL is not configured")

    payload = json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
        "stream": False,
    }).encode("utf-8")
    request = Request(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        try:
            with urlopen(request, timeout=OPENROUTER_TIMEOUT_SECONDS) as response:
                result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            choices = result.get("choices") or []
            text = choices[0].get("message", {}).get("content", "").strip() if choices else ""
            if not text:
                raise ProviderError("OpenRouter returned an empty response")
            return Completion(text=text, model=OPENROUTER_MODEL)
        except HTTPError as exc:
            if exc.code == 429:
                if attempt < OPENROUTER_MAX_RETRIES:
                    retry_after = exc.headers.get("Retry-After", "1") if exc.headers else "1"
                    try:
                        delay = min(max(float(retry_after), 0), 3)
                    except ValueError:
                        delay = 1
                    time.sleep(delay)
                    continue
                raise ProviderError("OpenRouter rate limit reached", retryable=True) from None
            if exc.code in (401, 403):
                raise ProviderError("OpenRouter authentication or access was rejected") from None
            if exc.code >= 500:
                if attempt < OPENROUTER_MAX_RETRIES:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise ProviderError("OpenRouter service is temporarily unavailable", retryable=True) from None
            raise ProviderError("OpenRouter rejected the request") from None
        except (TimeoutError, URLError):
            if attempt < OPENROUTER_MAX_RETRIES:
                time.sleep(0.25 * (attempt + 1))
                continue
            raise ProviderError("OpenRouter request timed out or could not connect", retryable=True) from None
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, AttributeError, TypeError):
            raise ProviderError("OpenRouter returned an invalid response") from None
    raise ProviderError("OpenRouter service is temporarily unavailable", retryable=True)
