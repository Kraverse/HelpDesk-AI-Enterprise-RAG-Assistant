from types import SimpleNamespace

import pytest

from utils import rag


@pytest.fixture(autouse=True)
def configured_openrouter_key(monkeypatch):
    monkeypatch.setattr(rag, "OPENROUTER_API_KEY", "test-key")


class FakeStore:
    def __init__(self, results): self.results = results
    def similarity_search_with_score(self, _question, k): return self.results[:k]


def document(text, source):
    return SimpleNamespace(page_content=text, metadata={"source": source})


def test_retrieval_refuses_when_nearest_match_is_weak(monkeypatch):
    monkeypatch.setattr(rag, "_vector_store", FakeStore([(document("Shipping takes 5 days.", "shipping.txt"), 2.0)]))
    monkeypatch.setattr(rag, "MAX_L2_DISTANCE", 1.5)
    monkeypatch.setattr(rag, "OPENROUTER_MODEL", "approved-test-model")
    monkeypatch.setattr(
        rag,
        "chat_completion",
        lambda _messages: (_ for _ in ()).throw(AssertionError("LLM must not be called for weak evidence")),
    )
    response = rag.query("Tell me about returns")
    assert response.answer == rag.REFUSAL
    assert response.sources == []


def test_answer_requires_valid_source_citation(monkeypatch):
    monkeypatch.setattr(rag, "_vector_store", FakeStore([(document("Business hours are weekdays.", "faq.txt"), 0.2)]))
    monkeypatch.setattr(rag, "MAX_L2_DISTANCE", 1.5)
    monkeypatch.setattr(rag, "OPENROUTER_MODEL", "test-model")
    monkeypatch.setattr(rag, "chat_completion", lambda _messages: SimpleNamespace(text="Hours are weekdays [S1]"))
    response = rag.query("What are the business hours?")
    assert response.answer == "Hours are weekdays [S1]"
    assert response.citations == [{"id": "S1", "source": "faq.txt"}]


def test_uncited_or_invalid_citation_is_rejected(monkeypatch):
    monkeypatch.setattr(rag, "_vector_store", FakeStore([(document("Returns within 30 days.", "returns.txt"), 0.2)]))
    monkeypatch.setattr(rag, "MAX_L2_DISTANCE", 1.5)
    monkeypatch.setattr(rag, "OPENROUTER_MODEL", "test-model")
    monkeypatch.setattr(rag, "chat_completion", lambda _messages: SimpleNamespace(text="Returns are accepted within 30 days [S9]."))
    response = rag.query("What is the return policy?")
    assert response.answer == rag.REFUSAL
    assert response.citations == []


def test_provider_failure_is_user_safe(monkeypatch):
    monkeypatch.setattr(rag, "_vector_store", FakeStore([(document("Shipping takes 5 days.", "shipping.txt"), 0.2)]))
    monkeypatch.setattr(rag, "MAX_L2_DISTANCE", 1.5)
    monkeypatch.setattr(rag, "OPENROUTER_MODEL", "test-model")
    def fail(_messages): raise rag.ProviderError("OpenRouter authentication or access was rejected")
    monkeypatch.setattr(rag, "chat_completion", fail)
    response = rag.query("How long is shipping?")
    assert "authentication" in response.error
    assert "Authorization" not in response.error


def test_startup_reports_missing_openrouter_key_without_disclosing_secrets(monkeypatch):
    monkeypatch.setattr(rag, "get_embeddings", lambda: object())
    monkeypatch.setattr(rag, "load_vector_store", lambda: object())
    monkeypatch.setattr(rag, "_chunk_count", 3)
    monkeypatch.setattr(rag, "OPENROUTER_MODEL", "approved-test-model")
    monkeypatch.setattr(rag, "OPENROUTER_API_KEY", "")
    status = rag.initialize_system()
    assert status["index_ready"] is True
    assert status["llm_ready"] is False
    assert status["error"] == "OPENROUTER_API_KEY is not configured; answer generation is unavailable."
