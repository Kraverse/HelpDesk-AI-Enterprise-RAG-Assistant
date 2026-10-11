"""Retrieval-augmented generation using the existing FAISS index and embeddings."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import (
    DATA_PATH,
    EMBEDDING_MODEL,
    FAISS_PATH,
    MAX_L2_DISTANCE,
    NVIDIA_API_KEY,
    NVIDIA_MODEL,
    TOP_K,
    validate_settings,
)
from utils.input_policy import validate_question
from utils.nvidia_client import ProviderError, chat_completion
from utils.prompts import REFUSAL, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Safe, structured response for the UI."""

    answer: str = ""
    documents: list[Any] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    citations: list[dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0  # retained for compatibility; deliberately not calibrated/displayed
    relevance: str = "No evidence"
    response_time: float = 0.0
    error: str | None = None


_embeddings: Any = None
_vector_store: Any = None
_doc_count = 0
_chunk_count = 0


def get_embeddings() -> Any:
    """Load the original local sentence-transformer embedding model lazily."""
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def load_vector_store() -> Any:
    """Load the checked-in FAISS pair using only the trusted repository artifact."""
    global _vector_store, _chunk_count
    index_dir = Path(FAISS_PATH)
    if not (index_dir / "index.faiss").is_file() or not (index_dir / "index.pkl").is_file():
        logger.error("FAISS index artifacts are missing")
        return None
    try:
        from langchain_community.vectorstores import FAISS

        # The repository artifact is trusted and versioned with its source. Never load
        # user-uploaded pickle indexes; FAISS docstore persistence uses Python pickle.
        store = FAISS.load_local(
            str(index_dir), get_embeddings(), allow_dangerous_deserialization=True
        )
        _vector_store = store
        _chunk_count = int(store.index.ntotal)
        return store
    except Exception as exc:  # noqa: BLE001 - normalize variable FAISS/version errors.
        logger.warning("FAISS index load failed (%s)", type(exc).__name__)
        return None


def initialize_system() -> dict[str, Any]:
    """Initialize local retrieval and report safe, non-secret component status."""
    global _doc_count
    status: dict[str, Any] = {
        "embeddings_ready": False,
        "index_ready": False,
        "llm_ready": bool(NVIDIA_MODEL and NVIDIA_API_KEY),
        "llm_name": NVIDIA_MODEL or "NVIDIA model not configured",
        "embedding_model": EMBEDDING_MODEL,
        "doc_count": 0,
        "chunk_count": 0,
        "error": None,
    }
    issues = validate_settings()
    if issues:
        status["error"] = "Application configuration needs attention."
        return status
    try:
        get_embeddings()
        status["embeddings_ready"] = True
    except Exception as exc:  # noqa: BLE001 - normalize optional model initialization errors.
        logger.warning("Embedding initialization failed (%s)", type(exc).__name__)
        status["error"] = "The local retrieval model could not be initialized."
        return status

    store = load_vector_store()
    if store is not None:
        status["index_ready"] = True
        status["chunk_count"] = _chunk_count
    _doc_count = len(list(Path(DATA_PATH).glob("*.txt"))) if Path(DATA_PATH).exists() else 0
    status["doc_count"] = _doc_count
    if not status["index_ready"]:
        status["error"] = "The knowledge index is unavailable. Contact the application owner."
    elif not NVIDIA_MODEL:
        status["error"] = "NVIDIA_MODEL is not configured; generation is unavailable."
    elif not NVIDIA_API_KEY:
        status["error"] = "NVIDIA_API_KEY is not configured; generation is unavailable."
    return status


def _citations(documents: list[Any]) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    seen: set[str] = set()
    for doc in documents:
        source = str(getattr(doc, "metadata", {}).get("source", "Unknown"))
        if source not in seen:
            seen.add(source)
            citations.append({"id": f"S{len(citations) + 1}", "source": source})
    return citations


def query(question: str) -> RAGResponse:
    """Retrieve evidence, abstain on weak matches, then generate cited content."""
    started = time.perf_counter()
    response = RAGResponse()
    normalized, validation_error = validate_question(question)
    if validation_error:
        response.error = validation_error
        response.response_time = round(time.perf_counter() - started, 2)
        return response
    question = normalized
    if _vector_store is None:
        response.error = "The knowledge index is unavailable. Please try again later."
        response.response_time = round(time.perf_counter() - started, 2)
        return response

    try:
        results = _vector_store.similarity_search_with_score(question, k=TOP_K)
    except Exception as exc:  # noqa: BLE001 - normalize vector-store adapter errors.
        logger.warning("Retrieval failed (%s)", type(exc).__name__)
        response.error = "Knowledge retrieval failed. Please try again."
        response.response_time = round(time.perf_counter() - started, 2)
        return response

    if not results:
        response.answer = REFUSAL
        response.response_time = round(time.perf_counter() - started, 2)
        return response

    documents = [doc for doc, _score in results]
    distances = [float(score) for _doc, score in results]
    response.documents = documents
    response.citations = _citations(documents)
    response.sources = [citation["source"] for citation in response.citations]
    nearest_distance = min(distances)
    if nearest_distance > MAX_L2_DISTANCE:
        response.answer = REFUSAL
        response.relevance = "No sufficiently relevant evidence"
        response.citations = []
        response.sources = []
        response.response_time = round(time.perf_counter() - started, 2)
        return response
    response.relevance = "Evidence retrieved; relevance is not a probability of correctness"

    if not NVIDIA_MODEL:
        response.error = "Answer generation is not configured. Please contact the application owner."
        response.response_time = round(time.perf_counter() - started, 2)
        return response

    citation_by_source = {item["source"]: item["id"] for item in response.citations}
    evidence = []
    for doc in documents:
        source = str(getattr(doc, "metadata", {}).get("source", "Unknown"))
        marker = citation_by_source.get(source, "S1")
        evidence.append(f"[{marker}] Source: {source}\n{doc.page_content}")
    context = "\n\n---\n\n".join(evidence)
    try:
        result = chat_completion([
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ])
        answer = result.text.strip()
        if answer == REFUSAL:
            response.answer = REFUSAL
            response.sources = []
            response.citations = []
            response.response_time = round(time.perf_counter() - started, 2)
            return response
        markers = set(re.findall(r"\[(S\d+)\]", answer))
        valid_markers = {item["id"] for item in response.citations}
        if answer != REFUSAL and (not markers or not markers.issubset(valid_markers)):
            logger.info("Generation rejected because citations were absent or invalid")
            response.answer = REFUSAL
            response.sources = []
            response.citations = []
        else:
            response.answer = answer
    except ProviderError as exc:
        logger.warning("Generation unavailable (%s)", exc.category)
        response.error = f"{exc.category}. Please retry later or contact the application owner."
    except Exception as exc:  # noqa: BLE001 - final safe boundary for client failures.
        logger.error("Unexpected generation failure (%s)", type(exc).__name__)
        response.error = "Answer generation failed. Please retry later."
    response.response_time = round(time.perf_counter() - started, 2)
    return response
