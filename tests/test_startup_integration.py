"""Optional integration coverage for the installed FAISS/LangChain runtime."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

langchain_community = pytest.importorskip("langchain_community")
faiss = pytest.importorskip("faiss")
from langchain_core.embeddings import Embeddings

from utils import rag

APP = Path(__file__).resolve().parents[1] / "app.py"


class OfflineIndexEmbeddings(Embeddings):
    """Stand-in that lets the real index loader run without model downloads."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 384


def test_streamlit_starts_and_loads_checked_in_index_without_network(monkeypatch):
    monkeypatch.setattr(rag, "_vector_store", None)
    monkeypatch.setattr(rag, "_embeddings", None)
    monkeypatch.setattr(rag, "get_embeddings", lambda: OfflineIndexEmbeddings())
    at = AppTest.from_file(str(APP), default_timeout=15).run()
    status = at.session_state["system_status"]
    assert status["index_ready"] is True
    assert status["chunk_count"] == 3
    assert rag._vector_store.index.d == 384
    assert len(rag._vector_store.docstore._dict) == 3
    assert len(rag._vector_store.index_to_docstore_id) == 3
    assert not at.exception
