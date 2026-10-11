from types import SimpleNamespace

from utils import rag


class FakeStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def similarity_search_with_score(self, question, k):
        self.calls.append((question, k))
        return self.results[:k]


def doc(text, source):
    return SimpleNamespace(page_content=text, metadata={"source": source})


def configure_llm(monkeypatch, answer):
    monkeypatch.setattr(rag, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(rag, "chat_completion", lambda _messages: SimpleNamespace(text=answer))


def test_broad_resume_question_uses_uploaded_chunks_and_cites_them(monkeypatch):
    resume = doc(
        "Professional Summary: AI/ML student. Technical Skills: Python, LangChain. "
        "Projects: MedRAG-X and HelpDesk AI. Education: BE Computer Science.",
        "Uploaded: synthetic_resume.txt",
    )
    uploaded = FakeStore([(resume, 2.4)])
    base = FakeStore([(doc("Returns are accepted within 30 days.", "returns.txt"), 0.1)])
    monkeypatch.setattr(rag, "_vector_store", base)
    monkeypatch.setattr(rag, "MAX_L2_DISTANCE", 1.5)
    configure_llm(monkeypatch, "The resume describes an AI/ML student with Python and LangChain skills and projects including MedRAG-X and HelpDesk AI [S1].")

    response = rag.query("Can you tell me something about my resume?", uploaded_store=uploaded)

    assert "AI/ML student" in response.answer
    assert response.citations == [{"id": "S1", "source": "Uploaded: synthetic_resume.txt"}]
    assert len(uploaded.calls) == 1
    assert base.calls == []


def test_resume_skills_question_uses_uploaded_index_and_citation(monkeypatch):
    resume = doc(
        "Technical Skills: Python, FastAPI, LangChain, FAISS. Projects: MedRAG-X.",
        "Uploaded: synthetic_resume.txt",
    )
    uploaded = FakeStore([(resume, 0.2)])
    base = FakeStore([(doc("Support is available by email.", "support.txt"), 0.1)])
    monkeypatch.setattr(rag, "_vector_store", base)
    monkeypatch.setattr(rag, "MAX_L2_DISTANCE", 1.5)
    configure_llm(monkeypatch, "The listed technical skills include Python, FastAPI, LangChain, and FAISS [S1].")

    response = rag.query("What skills are listed in my resume?", uploaded_store=uploaded)

    assert "Python" in response.answer
    assert response.citations[0]["source"] == "Uploaded: synthetic_resume.txt"
    assert base.calls == []


def test_irrelevant_question_still_abstains(monkeypatch):
    uploaded = FakeStore([
        (doc("Technical Skills: Python, FastAPI. Projects: MedRAG-X.", "Uploaded: synthetic_resume.txt"), 3.2)
    ])
    monkeypatch.setattr(rag, "_vector_store", None)
    monkeypatch.setattr(rag, "MAX_L2_DISTANCE", 1.5)
    configure_llm(monkeypatch, "This must not be returned [S1].")

    response = rag.query("What is the weather on Mars?", uploaded_store=uploaded)

    assert response.answer == rag.REFUSAL
    assert response.citations == []


def test_resume_projects_question_returns_source_citation(monkeypatch):
    resume = doc(
        "Projects: MedRAG-X, a cited medical document RAG assistant; HelpDesk AI, an enterprise support RAG assistant.",
        "Uploaded: synthetic_resume.txt",
    )
    uploaded = FakeStore([(resume, 0.25)])
    monkeypatch.setattr(rag, "_vector_store", None)
    monkeypatch.setattr(rag, "MAX_L2_DISTANCE", 1.5)
    configure_llm(monkeypatch, "The resume lists MedRAG-X and HelpDesk AI as projects [S1].")

    response = rag.query("What projects are mentioned in my resume?", uploaded_store=uploaded)

    assert "MedRAG-X" in response.answer
    assert "HelpDesk AI" in response.answer
    assert response.citations == [{"id": "S1", "source": "Uploaded: synthetic_resume.txt"}]


def test_resume_query_without_upload_does_not_search_base_index(monkeypatch):
    base = FakeStore([(doc("Generic company support information.", "support.txt"), 0.1)])
    monkeypatch.setattr(rag, "_vector_store", base)

    response = rag.query("Summarize my resume", uploaded_store=None)

    assert "Upload and process your resume" in response.error
    assert base.calls == []
