"""Retrieval-augmented generation with evidence-grounded uploaded-document support."""

from __future__ import annotations

import logging
import math
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
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    TOP_K,
    validate_settings,
)
from utils.input_policy import validate_question
from utils.openrouter_client import ProviderError, chat_completion
from utils.prompts import REFUSAL, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# A broad resume question often has little literal overlap with a resume's individual
# chunks. Expand only explicit resume/CV requests, and search the session-upload index
# separately so the repository's help-centre content cannot outrank the user's file.
_RESUME_TERMS = re.compile(r"\b(resume|résumé|curriculum vitae|\bcv\b)\b", re.IGNORECASE)
_OVERVIEW_TERMS = re.compile(
    r"\b(tell me about|tell me something|summari[sz]e|overview|describe|what is in|"
    r"what does .* (say|contain)|skills|projects|experience|education|certifications?)\b",
    re.IGNORECASE,
)
_RESUME_EXPANSION = (
    "resume curriculum vitae professional summary career profile work experience "
    "education technical skills programming languages tools frameworks projects "
    "certifications achievements internships"
)
_RESUME_EVIDENCE_TERMS = {
    "summary", "profile", "objective", "experience", "education", "skills",
    "technical", "projects", "certifications", "achievements", "internship",
    "python", "machine", "learning", "generative", "ai", "ml", "developer",
}


@dataclass
class RAGResponse:
    """Safe, structured response for the UI."""

    answer: str = ""
    documents: list[Any] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    citations: list[dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    relevance: str = "No evidence"
    response_time: float = 0.0
    error: str | None = None


_embeddings: Any = None
_vector_store: Any = None
_doc_count = 0
_chunk_count = 0


def get_embeddings() -> Any:
    """Load the configured local sentence-transformer model lazily."""
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        # Keep the default (unnormalized) SentenceTransformer vectors explicit so
        # upload indexing and query embedding share the same normalization convention.
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": False},
        )
    return _embeddings


def load_vector_store() -> Any:
    """Load the trusted, checked-in FAISS index."""
    global _vector_store, _chunk_count
    index_dir = Path(FAISS_PATH)
    if not (index_dir / "index.faiss").is_file() or not (index_dir / "index.pkl").is_file():
        logger.error("FAISS index artifacts are missing")
        return None
    try:
        from langchain_community.vectorstores import FAISS

        store = FAISS.load_local(
            str(index_dir), get_embeddings(), allow_dangerous_deserialization=True
        )
        _vector_store = store
        _chunk_count = int(store.index.ntotal)
        logger.info(
            "Base index ready: chunks=%d dimension=%s metric=%s",
            _chunk_count,
            getattr(store.index, "d", "unknown"),
            _metric_name(store),
        )
        return store
    except Exception as exc:  # noqa: BLE001
        logger.warning("FAISS index load failed (%s)", type(exc).__name__)
        return None


def initialize_system() -> dict[str, Any]:
    """Initialize local retrieval and report safe component status."""
    global _doc_count
    status: dict[str, Any] = {
        "embeddings_ready": False,
        "index_ready": False,
        "llm_ready": bool(OPENROUTER_MODEL and OPENROUTER_API_KEY),
        "llm_name": OPENROUTER_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "doc_count": 0,
        "chunk_count": 0,
        "error": None,
    }
    if validate_settings():
        status["error"] = "Application configuration needs attention."
        return status
    try:
        get_embeddings()
        status["embeddings_ready"] = True
    except Exception as exc:  # noqa: BLE001
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
    elif not OPENROUTER_API_KEY:
        status["error"] = "OPENROUTER_API_KEY is not configured; answer generation is unavailable."
    return status


def _metric_name(store: Any) -> str:
    """Describe the FAISS score convention without assuming every index is L2."""
    metric = getattr(getattr(store, "index", None), "metric_type", None)
    if metric == 1:
        return "squared_l2_distance"
    if metric == 0:
        return "inner_product"
    return "unknown_or_test_store"


def _citations(documents: list[Any]) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    seen: set[str] = set()
    for doc in documents:
        source = str(getattr(doc, "metadata", {}).get("source", "Unknown"))
        if source not in seen:
            seen.add(source)
            citations.append({"id": f"S{len(citations) + 1}", "source": source})
    return citations


def _is_resume_query(question: str) -> bool:
    return bool(_RESUME_TERMS.search(question))


def _expanded_resume_query(question: str) -> str:
    return f"{question} {_RESUME_EXPANSION}"


def _resume_lexical_overlap(question: str, document: Any) -> int:
    """Count matching resume-domain terms; never inspect/log document text in diagnostics."""
    text = str(getattr(document, "page_content", "")).lower()
    query_terms = set(re.findall(r"[a-z]{2,}", question.lower()))
    relevant_terms = query_terms & _RESUME_EVIDENCE_TERMS
    return sum(1 for term in relevant_terms if re.search(rf"\b{re.escape(term)}\b", text))


def _retrieve(store: Any, question: str, k: int, label: str) -> list[tuple[Any, float]]:
    """Search using the configured query embedder and emit content-free diagnostics."""
    try:
        index = getattr(store, "index", None)
        metric = _metric_name(store)
        if index is not None and hasattr(store, "similarity_search_by_vector_with_score"):
            vector = get_embeddings().embed_query(question)
            dimension = getattr(index, "d", None)
            if dimension is not None and len(vector) != int(dimension):
                logger.warning(
                    "Retrieval rejected: store=%s reason=embedding_dimension_mismatch query_dim=%d index_dim=%d",
                    label, len(vector), int(dimension),
                )
                return []
            results = store.similarity_search_by_vector_with_score(vector, k=k)
            query_dimension = len(vector)
        else:
            # Compatibility for simple test doubles and older store adapters.
            results = store.similarity_search_with_score(question, k=k)
            query_dimension = "adapter-managed"
        safe_scores = [round(float(score), 4) for _doc, score in results if math.isfinite(float(score))]
        logger.info(
            "Retrieval diagnostics: store=%s model=%s normalized=false chunks=%s query_dim=%s metric=%s retrieved=%d scores=%s",
            label,
            EMBEDDING_MODEL,
            getattr(index, "ntotal", "unknown"),
            query_dimension,
            metric,
            len(results),
            safe_scores,
        )
        return [
            (doc, float(score))
            for doc, score in results
            if math.isfinite(float(score)) and str(getattr(doc, "page_content", "")).strip()
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Retrieval failed: store=%s exception=%s", label, type(exc).__name__)
        return []


def query(question: str, uploaded_store: Any = None) -> RAGResponse:
    """Retrieve grounded evidence, then generate an answer with validated citations."""
    started = time.perf_counter()
    response = RAGResponse()
    normalized, validation_error = validate_question(question)
    if validation_error:
        response.error = validation_error
        response.response_time = round(time.perf_counter() - started, 2)
        return response
    question = normalized
    resume_query = _is_resume_query(question)
    overview_query = resume_query and bool(_OVERVIEW_TERMS.search(question))

    # A question explicitly about the user's resume must be grounded in session uploads,
    # not in generic built-in help-centre documents. If no upload index exists, refuse.
    if resume_query and uploaded_store is not None:
        stores = [(uploaded_store, "session_uploads")]
        retrieval_question = _expanded_resume_query(question) if overview_query else question
    else:
        stores = [
            (store, label)
            for store, label in ((_vector_store, "base_index"), (uploaded_store, "session_uploads"))
            if store is not None
        ]
        retrieval_question = question

    if not stores:
        response.error = (
            "Upload and process your resume first to ask questions about it."
            if resume_query else "The knowledge index is unavailable. Please try again later."
        )
        response.response_time = round(time.perf_counter() - started, 2)
        return response

    results: list[tuple[Any, float, str]] = []
    for store, label in stores:
        results.extend((doc, score, label) for doc, score in _retrieve(store, retrieval_question, TOP_K, label))

    if not results:
        logger.info("Evidence rejected: reason=no_nonempty_chunks_retrieved")
        response.answer = REFUSAL
        response.relevance = "No sufficiently relevant evidence"
        response.response_time = round(time.perf_counter() - started, 2)
        return response

    # FAISS L2 scores are distances (lower is better); inner-product scores rank higher-first.
    # Sort within the index's declared convention instead of treating all scores as distances.
    def _rank_key(item: tuple[Any, float, str]) -> float:
        doc, score, label = item
        store = next((candidate for candidate, candidate_label in stores if candidate_label == label), None)
        return -float(score) if _metric_name(store) == "inner_product" else float(score)

    results.sort(key=_rank_key)
    accepted: list[tuple[Any, float, str]] = []
    rejected_scores: list[float] = []
    for doc, score, label in results:
        store = next((candidate for candidate, candidate_label in stores if candidate_label == label), None)
        metric = _metric_name(store) if store is not None else "unknown_or_test_store"
        if metric == "squared_l2_distance":
            if score <= MAX_L2_DISTANCE:
                accepted.append((doc, score, label))
            else:
                rejected_scores.append(score)
        elif metric == "inner_product":
            # Inner product ranks higher scores first. Do not apply an L2 distance cutoff.
            accepted.append((doc, score, label))
        elif score <= MAX_L2_DISTANCE:
            # Compatibility path for older adapters without a metric declaration.
            accepted.append((doc, score, label))
        else:
            rejected_scores.append(score)

    if resume_query and overview_query and uploaded_store is not None and not accepted:
        # For an explicit resume overview, the query may not match any one section closely.
        # Use only non-empty chunks from the user's uploaded index, and require either the
        # configured distance threshold or a resume-domain lexical match. This is a targeted
        # fallback, not a global threshold reduction; the base index remains excluded.
        fallback = [
            (doc, score, label)
            for doc, score, label in results
            if label == "session_uploads"
            and _resume_lexical_overlap(retrieval_question, doc) > 0
        ]
        if fallback:
            accepted = fallback[:TOP_K]
            logger.info(
                "Evidence fallback: mode=resume_overview_upload_only reason=domain_term_overlap accepted=%d",
                len(accepted),
            )

    logger.info(
        "Evidence filter: resume_query=%s overview_query=%s accepted=%d rejected=%d threshold_squared_l2=%.3f",
        resume_query, overview_query, len(accepted), len(rejected_scores), MAX_L2_DISTANCE,
    )
    if not accepted:
        if rejected_scores:
            logger.info("Evidence rejected: reason=score_threshold scores=%s", [round(s, 4) for s in rejected_scores])
        response.answer = REFUSAL
        response.relevance = "No sufficiently relevant evidence"
        response.response_time = round(time.perf_counter() - started, 2)
        return response

    documents = [doc for doc, _score, _label in accepted[:TOP_K]]
    response.documents = documents
    response.citations = _citations(documents)
    response.sources = [citation["source"] for citation in response.citations]
    response.relevance = "Evidence retrieved; relevance is not a probability of correctness"

    if not OPENROUTER_API_KEY:
        response.error = "OPENROUTER_API_KEY is not configured; answer generation is unavailable."
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
        if not markers or not markers.issubset(valid_markers):
            logger.info("Generation rejected: reason=missing_or_invalid_citations")
            response.answer = REFUSAL
            response.sources = []
            response.citations = []
        else:
            response.answer = answer
    except ProviderError as exc:
        logger.warning("Generation unavailable (%s)", exc.category)
        response.error = f"{exc.category}. Please retry later or contact the application owner."
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected generation failure (%s)", type(exc).__name__)
        response.error = "Answer generation failed. Please retry later."
    response.response_time = round(time.perf_counter() - started, 2)
    return response
