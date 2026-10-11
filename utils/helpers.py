"""
Utility helper functions for HelpDesk AI.
"""

import os
import time
from datetime import datetime
from typing import List, Tuple, Optional

from langchain_core.documents import Document


def get_time() -> str:
    """Return current timestamp as a formatted string."""
    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")


def response_time(start: float) -> float:
    """Calculate elapsed seconds since *start*."""
    return round(time.time() - start, 2)


def ensure_folder(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def format_sources(documents: List[Document]) -> List[str]:
    """Extract unique source filenames from retrieved documents.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        Deduplicated list of source filenames.
    """
    if not documents:
        return []
    seen: set[str] = set()
    sources: list[str] = []
    for doc in documents:
        src = doc.metadata.get("source", "Unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


def save_chat(
    chat_history: List[dict],
    filename: str = "chat_history.txt",
) -> str:
    """Write the conversation to a text file.

    Args:
        chat_history: List of dicts with 'role' and 'content' keys.
        filename: Output file path.

    Returns:
        The path to the saved file.
    """
    ensure_folder(os.path.dirname(filename) or ".")

    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(f"HelpDesk AI — Chat Export\n")
        fh.write(f"Exported: {get_time()}\n")
        fh.write("=" * 50 + "\n\n")

        for entry in chat_history:
            role = entry.get("role", "unknown")
            content = entry.get("content", "")
            fh.write(f"[{role.upper()}]\n{content}\n\n")

    return filename


def chat_history_to_string(chat_history: List[dict]) -> str:
    """Serialize chat history into a downloadable text string.

    Args:
        chat_history: List of dicts with 'role' and 'content' keys.

    Returns:
        Formatted conversation text.
    """
    lines = [
        "HelpDesk AI — Chat Export",
        f"Exported: {get_time()}",
        "=" * 50,
        "",
    ]
    for entry in chat_history:
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        lines.append(f"[{role.upper()}]")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def get_confidence_color(confidence: float) -> str:
    """Return a CSS color based on the confidence percentage.

    Args:
        confidence: Value between 0 and 100.

    Returns:
        Hex color string.
    """
    if confidence >= 70:
        return "#3fb950"
    elif confidence >= 40:
        return "#d29922"
    return "#f85149"