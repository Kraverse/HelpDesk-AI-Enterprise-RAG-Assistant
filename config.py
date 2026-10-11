"""Application settings sourced from environment variables or Streamlit secrets."""

from __future__ import annotations

import os
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Production can supply settings through process environment or Streamlit secrets.
    pass


def _secret(name: str, default: str = "") -> str:
    """Read a secret without requiring Streamlit during tests or CLI use."""
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        import streamlit as st

        value = st.secrets.get(name, default)
    except Exception:  # noqa: BLE001 - absent/unconfigured Streamlit secrets fall back to environment.
        value = default
    return str(value).strip()


def _int_setting(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


def _float_setting(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


APP_NAME = "HelpDesk AI"
APP_SUBTITLE = "Enterprise RAG Assistant"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
FAISS_PATH = os.getenv("FAISS_PATH", os.path.join(ROOT_DIR, "faiss_index"))
DATA_PATH = os.getenv("DATA_PATH", os.path.join(ROOT_DIR, "data"))
CHUNK_SIZE = _int_setting("CHUNK_SIZE", 500, 1, 10000)
CHUNK_OVERLAP = _int_setting("CHUNK_OVERLAP", 80, 0, 9999)
TOP_K = _int_setting("TOP_K", 3, 1, 8)
TEMPERATURE = _float_setting("TEMPERATURE", 0.2, 0, 2)
MAX_NEW_TOKENS = _int_setting("MAX_NEW_TOKENS", 512, 64, 2048)
MAX_QUESTION_CHARS = _int_setting("MAX_QUESTION_CHARS", 1000, 100, 5000)
MAX_CHAT_MESSAGES = _int_setting("MAX_CHAT_MESSAGES", 40, 4, 200)
MAX_SESSION_CHARS = _int_setting("MAX_SESSION_CHARS", 30000, 1000, 200000)
SESSION_REQUESTS_PER_MINUTE = _int_setting("SESSION_REQUESTS_PER_MINUTE", 20, 1, 120)
MAX_L2_DISTANCE = _float_setting("MAX_L2_DISTANCE", 1.5, 0.01, 100)
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "").strip()
NVIDIA_API_KEY = _secret("NVIDIA_API_KEY")
NVIDIA_TIMEOUT_SECONDS = _float_setting("NVIDIA_TIMEOUT_SECONDS", 30, 2, 120)
NVIDIA_MAX_RETRIES = _int_setting("NVIDIA_MAX_RETRIES", 1, 0, 2)
ASK_KRAVERSE_URL = os.getenv("ASK_KRAVERSE_URL", "").strip()

# Compatibility alias retained for older UI references; never contains a credential.
LLM_REPO = NVIDIA_MODEL or "NVIDIA model not configured"


def validate_settings() -> list[str]:
    """Return safe configuration errors without echoing secret values."""
    errors: list[str] = []
    parsed_base = urlparse(NVIDIA_BASE_URL)
    if parsed_base.scheme != "https" or not parsed_base.netloc:
        errors.append("NVIDIA_BASE_URL must use HTTPS.")
    if CHUNK_SIZE <= 0 or CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append("Chunk size and overlap are invalid.")
    if MAX_L2_DISTANCE <= 0:
        errors.append("MAX_L2_DISTANCE must be positive.")
    return errors
