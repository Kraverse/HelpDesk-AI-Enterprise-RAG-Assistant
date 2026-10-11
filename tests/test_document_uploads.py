from io import BytesIO
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from utils import document_uploads, rag


def test_txt_is_extracted_and_chunked():
    result = document_uploads.parse_uploaded_file(
        "policy.txt", b"Returns are accepted within thirty days.\nContact support for help."
    )
    assert result.status == "processed"
    assert result.digest
    assert result.chunks
    assert {chunk.metadata["source"] for chunk in result.chunks} == {"Uploaded: policy.txt"}
    assert "Returns are accepted" in " ".join(chunk.page_content for chunk in result.chunks)


def test_pdf_text_is_extracted_and_chunked():
    pytest.importorskip("pypdf")
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
    output = BytesIO()
    canvas = reportlab.Canvas(output)
    canvas.drawString(30, 250, "Refund requests are reviewed within five business days.")
    canvas.save()
    pdf = document_uploads.parse_uploaded_file("refunds.pdf", output.getvalue())
    assert pdf.status == "processed"
    assert "Refund requests" in " ".join(chunk.page_content for chunk in pdf.chunks)


def test_empty_unreadable_and_unsupported_uploads():
    assert document_uploads.parse_uploaded_file("empty.txt", b" ").status == "empty"
    invalid_text = document_uploads.parse_uploaded_file("bad.txt", b"\xff\xfe\x00")
    assert invalid_text.status == "unreadable"
    invalid_pdf = document_uploads.parse_uploaded_file("bad.pdf", b"not a pdf")
    assert invalid_pdf.status == "unreadable"
    unsupported = document_uploads.parse_uploaded_file("data.docx", b"content")
    assert unsupported.status == "unsupported"


def test_pdf_without_extractable_text_is_reported_empty():
    pypdf = pytest.importorskip("pypdf")
    output = BytesIO()
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.write(output)
    result = document_uploads.parse_uploaded_file("scan.pdf", output.getvalue())
    assert result.status == "empty"
    assert "OCR" in result.message


def test_duplicate_uploads_and_session_document_limit():
    files = [
        ("first.txt", b"First policy text."),
        ("copy.txt", b"First policy text."),
        ("second.txt", b"Second policy text."),
    ]
    accepted, statuses = document_uploads.prepare_upload_batch(files, set(), max_documents=1)
    assert len(accepted) == 1
    assert [status["status"] for status in statuses] == ["ready", "duplicate", "limit"]


def test_build_uploaded_index_uses_in_memory_faiss(monkeypatch):
    class FakeFAISS:
        @staticmethod
        def from_documents(documents, embedding):
            return {"documents": documents, "embedding": embedding}
    monkeypatch.setattr("langchain_community.vectorstores.FAISS.from_documents", FakeFAISS.from_documents)
    documents = [Document(page_content="uploaded", metadata={"source": "Uploaded: file.txt"})]
    result = document_uploads.build_uploaded_index(documents, embeddings="existing-embedder")
    assert result["documents"] == documents
    assert result["embedding"] == "existing-embedder"
    assert document_uploads.build_uploaded_index([], embeddings=None) is None


def test_real_faiss_upload_index_build_and_search_without_model_download():
    pytest.importorskip("faiss")
    from langchain_core.embeddings import Embeddings

    class TinyEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[1.0, float(index), 0.0] for index, _ in enumerate(texts)]
        def embed_query(self, _text):
            return [1.0, 0.0, 0.0]

    chunks = [
        Document(page_content="refund terms", metadata={"source": "Uploaded: terms.txt"}),
        Document(page_content="support contacts", metadata={"source": "Uploaded: contact.txt"}),
    ]
    index = document_uploads.build_uploaded_index(chunks, TinyEmbeddings())
    matches = index.similarity_search_with_score("refund terms", k=1)
    assert matches[0][0].metadata["source"] == "Uploaded: terms.txt"


def test_uploaded_retrieval_is_combined_with_base_and_cited(monkeypatch):
    class FakeStore:
        def __init__(self, results):
            self.results = results
        def similarity_search_with_score(self, _question, k):
            return self.results[:k]

    uploaded_doc = SimpleNamespace(
        page_content="Refund processing takes five business days.",
        metadata={"source": "Uploaded: refund-policy.txt"},
    )
    weak_base_doc = SimpleNamespace(
        page_content="Shipping takes three days.",
        metadata={"source": "shipping.txt"},
    )
    monkeypatch.setattr(rag, "_vector_store", FakeStore([(weak_base_doc, 2.0)]))
    monkeypatch.setattr(rag, "MAX_L2_DISTANCE", 1.5)
    monkeypatch.setattr(rag, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(rag, "chat_completion", lambda _messages: SimpleNamespace(text="Refunds take five business days [S1]."))

    response = rag.query(
        "How long does refund processing take?",
        uploaded_store=FakeStore([(uploaded_doc, 0.2)]),
    )
    assert response.answer == "Refunds take five business days [S1]."
    assert response.sources == ["Uploaded: refund-policy.txt"]
    assert response.citations == [{"id": "S1", "source": "Uploaded: refund-policy.txt"}]
