"""
==========================================================
HelpDesk AI – Enterprise RAG Assistant

RAG Engine — Retrieval & Generation Core
==========================================================
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS

from config import (
    HF_TOKEN,
    LLM_REPO,
    EMBEDDING_MODEL,
    FAISS_PATH,
    TOP_K,
    TEMPERATURE,
    MAX_NEW_TOKENS,
)
from utils.prompts import SYSTEM_PROMPT

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Response Data Class
# ============================================================

@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""

    answer: str = ""
    documents: List[Document] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    response_time: float = 0.0
    error: Optional[str] = None


# ============================================================
# Module-Level Caches
# ============================================================

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[FAISS] = None
_llm: Optional[HuggingFaceEndpoint] = None
_doc_count: int = 0
_chunk_count: int = 0


# ============================================================
# Embeddings
# ============================================================

def get_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache the sentence-transformer embedding model.

    Returns:
        Cached HuggingFaceEmbeddings instance.
    """
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embeddings


# ============================================================
# Vector Store
# ============================================================

def load_vector_store() -> Optional[FAISS]:
    """Load the FAISS vector store from disk.

    Returns:
        FAISS instance or None if the index is missing or corrupted.
    """
    global _vector_store, _chunk_count

    index_file = Path(FAISS_PATH) / "index.faiss"
    pkl_file = Path(FAISS_PATH) / "index.pkl"

    if not index_file.exists() or not pkl_file.exists():
        logger.error("FAISS index not found at %s/", FAISS_PATH)
        return None

    try:
        embeddings = get_embeddings()
        _vector_store = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        _chunk_count = _vector_store.index.ntotal
        logger.info("Vector store loaded — %d chunks.", _chunk_count)
        return _vector_store

    except Exception as exc:
        logger.error("Failed to load vector store: %s", exc)
        return None


# ============================================================
# LLM
# ============================================================

def get_llm() -> Optional[HuggingFaceEndpoint]:
    """Initialize and cache the HuggingFace Inference API LLM.

    Returns:
        HuggingFaceEndpoint instance or None if token is missing.
    """
    global _llm

    if _llm is None:
        if not HF_TOKEN or HF_TOKEN.strip() == "PASTE_YOUR_HUGGINGFACE_TOKEN_HERE":
            logger.warning("HF_TOKEN is not configured.")
            return None

        try:
            logger.info("Connecting to HuggingFace: %s", LLM_REPO)
            _llm = HuggingFaceEndpoint(
                repo_id=LLM_REPO,
                huggingfacehub_api_token=HF_TOKEN,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            logger.info("LLM connected.")
        except Exception as exc:
            logger.error("Failed to connect to LLM: %s", exc)
            return None

    return _llm


# ============================================================
# System Initialization
# ============================================================

def initialize_system() -> Dict[str, Any]:
    """Bootstrap every RAG component and return a status dict.

    Returns:
        Dictionary describing the readiness of each subsystem.
    """
    global _doc_count

    status: Dict[str, Any] = {
        "embeddings_ready": False,
        "index_ready": False,
        "llm_ready": False,
        "llm_name": LLM_REPO,
        "embedding_model": EMBEDDING_MODEL,
        "doc_count": 0,
        "chunk_count": 0,
        "error": None,
    }

    # --- Embeddings ---
    try:
        get_embeddings()
        status["embeddings_ready"] = True
    except Exception as exc:
        status["error"] = f"Embeddings failed: {exc}"
        logger.error("Embedding init failed: %s", exc)
        return status

    # --- Vector Store ---
    vs = load_vector_store()
    if vs is not None:
        status["index_ready"] = True
        status["chunk_count"] = _chunk_count

    # --- Document count ---
    data_path = Path("data")
    if data_path.exists():
        _doc_count = len(list(data_path.glob("*.txt")))
        status["doc_count"] = _doc_count

    # --- LLM ---
    llm = get_llm()
    if llm is not None:
        status["llm_ready"] = True
    elif not HF_TOKEN or HF_TOKEN.strip() == "PASTE_YOUR_HUGGINGFACE_TOKEN_HERE":
        status["error"] = "HF_TOKEN not configured. Set it in .env"
    else:
        status["error"] = "Could not connect to HuggingFace API."

    return status


# ============================================================
# Query
# ============================================================

def query(question: str) -> RAGResponse:
    """Execute the full RAG pipeline: retrieve → prompt → generate.

    Args:
        question: The user's natural-language question.

    Returns:
        RAGResponse with answer, sources, confidence, and timing.
    """
    start = time.time()
    response = RAGResponse()

    # --- Retrieve ---
    if _vector_store is None:
        response.error = "Vector store not loaded. Run ingest.py first."
        response.response_time = round(time.time() - start, 2)
        return response

    try:
        results = _vector_store.similarity_search_with_score(
            question, k=TOP_K
        )
    except Exception as exc:
        response.error = f"Retrieval failed: {exc}"
        response.response_time = round(time.time() - start, 2)
        return response

    if not results:
        response.answer = (
            "I couldn't find that information in the knowledge base."
        )
        response.response_time = round(time.time() - start, 2)
        return response

    # Separate documents and L2 distances
    documents: List[Document] = [doc for doc, _ in results]
    scores: List[float] = [score for _, score in results]

    response.documents = documents
    response.sources = list(
        {doc.metadata.get("source", "Unknown") for doc in documents}
    )

    # Compute confidence from L2 distances (normalized vectors)
    avg_distance = sum(scores) / len(scores)
    response.confidence = round(
        max(0.0, min(100.0, (1 - avg_distance / 1.5) * 100)), 1
    )

    # --- Generate ---
    llm = get_llm()
    if llm is None:
        response.error = "LLM not available. Check your HF_TOKEN in .env"
        response.response_time = round(time.time() - start, 2)
        return response

    try:
        context = "\n\n".join(doc.page_content for doc in documents)
        prompt = SYSTEM_PROMPT.format(context=context, question=question)

        raw_answer = llm.invoke(prompt)
        response.answer = raw_answer.strip() if raw_answer else "No response generated."

    except Exception as exc:
        response.error = f"Generation failed: {exc}"
        response.answer = (
            "An error occurred while generating a response. "
            "Please try again."
        )
        logger.error("LLM generation error: %s", exc)

    response.response_time = round(time.time() - start, 2)
    return response