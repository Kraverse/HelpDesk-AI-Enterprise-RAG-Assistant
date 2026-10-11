from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from utils import input_policy, rag

APP = Path(__file__).resolve().parents[1] / "app.py"


def ready_status():
    return {
        "embeddings_ready": True,
        "index_ready": True,
        "llm_ready": True,
        "llm_name": "test-model",
        "embedding_model": "test-embedding",
        "doc_count": 3,
        "chunk_count": 3,
        "error": None,
    }


def fake_response(question):
    return SimpleNamespace(
        answer=f"Answer to: {question} [S1]",
        sources=["faq.txt"],
        citations=[{"id": "S1", "source": "faq.txt"}],
        relevance="Evidence retrieved",
        response_time=0.01,
        error=None,
    )


def app_test(monkeypatch):
    monkeypatch.setattr(rag, "initialize_system", ready_status)
    monkeypatch.setattr(rag, "query", fake_response)
    return AppTest.from_file(str(APP), default_timeout=10).run()


def test_starter_question_creates_single_turn_and_citation(monkeypatch):
    at = app_test(monkeypatch)
    question = "What are your business hours?"
    next(button for button in at.button if button.label == question).click().run()
    history = at.session_state["chat_history"]
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[1]["citations"] == [{"id": "S1", "source": "faq.txt"}]
    assert not at.exception


def test_chat_input_and_export_are_available_after_question(monkeypatch):
    at = app_test(monkeypatch)
    at.chat_input[0].set_value("How can I contact support?").run()
    assert len(at.session_state["chat_history"]) == 2
    assert len(at.get("download_button")) == 1
    assert not at.exception


def test_clear_chat_resets_session(monkeypatch):
    at = app_test(monkeypatch)
    at.chat_input[0].set_value("Can I track my order?").run()
    next(button for button in at.button if button.label == "Clear chat").click().run()
    assert at.session_state["chat_history"] == []
    assert at.session_state["question_count"] == 0
    assert not at.exception


def test_invalid_question_does_not_create_a_chat_turn(monkeypatch):
    monkeypatch.setattr(rag, "initialize_system", ready_status)
    monkeypatch.setattr(input_policy, "validate_question", lambda _question: (None, "Please include words or numbers."))
    monkeypatch.setattr(rag, "query", lambda _question: (_ for _ in ()).throw(AssertionError("must not query")))
    at = AppTest.from_file(str(APP), default_timeout=10).run()
    at.chat_input[0].set_value("not actually accepted").run()
    assert at.session_state["chat_history"] == []
    assert len(at.warning) >= 1
    assert not at.exception


def test_unexpected_query_failure_is_safely_retained(monkeypatch):
    monkeypatch.setattr(rag, "initialize_system", ready_status)
    def fail(_question):
        raise RuntimeError("private transport detail")
    monkeypatch.setattr(rag, "query", fail)
    at = AppTest.from_file(str(APP), default_timeout=10).run()
    at.chat_input[0].set_value("How can I contact support?").run()
    assert at.session_state["chat_history"][-1]["content"] == "The request could not be completed. Please retry."
    assert "private transport detail" not in str(at)
    assert not at.exception


def test_document_upload_section_is_visible_and_empty_submit_is_safe(monkeypatch):
    at = app_test(monkeypatch)
    assert len(at.get("file_uploader")) == 1
    assert any(button.label == "Process selected documents" for button in at.button)
    next(button for button in at.button if button.label == "Process selected documents").click().run()
    assert any("Choose at least one" in element.value for element in at.warning)
    assert not at.exception
