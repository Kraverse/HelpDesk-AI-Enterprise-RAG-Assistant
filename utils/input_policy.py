"""Input and session bounds shared by the UI and offline tests."""

from __future__ import annotations

from config import (
    MAX_CHAT_MESSAGES,
    MAX_QUESTION_CHARS,
    MAX_SESSION_CHARS,
    SESSION_REQUESTS_PER_MINUTE,
)


def validate_question(question: str) -> tuple[str | None, str | None]:
    """Return normalized question and safe validation error, if any."""
    normalized = question.strip()
    if not normalized:
        return None, "Enter a question before sending."
    if len(normalized) > MAX_QUESTION_CHARS:
        return None, f"Please keep questions under {MAX_QUESTION_CHARS} characters."
    if not any(char.isalnum() for char in normalized):
        return None, "Please include words or numbers in your question."
    return normalized, None


def bound_history(history: list[dict]) -> list[dict]:
    """Retain a bounded suffix of complete user/assistant message pairs."""
    valid = [
        item for item in history
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ]
    valid = valid[-MAX_CHAT_MESSAGES:]
    total = sum(len(item["content"]) for item in valid)
    while valid and total > MAX_SESSION_CHARS:
        total -= len(valid.pop(0)["content"])
    # Avoid starting with an assistant answer detached from its question.
    if valid and valid[0]["role"] == "assistant":
        valid.pop(0)
    return valid


def register_request(timestamps: list[float], now: float) -> tuple[list[float], bool]:
    """Apply a session-local rolling one-minute request limit."""
    recent = [stamp for stamp in timestamps if 0 <= now - stamp < 60]
    if len(recent) >= SESSION_REQUESTS_PER_MINUTE:
        return recent, False
    return [*recent, now], True
