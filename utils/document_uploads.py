"""Ephemeral document upload parsing and indexing helpers for the existing RAG app."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_UPLOAD_DOCUMENTS,
    MAX_UPLOAD_FILE_BYTES,
)

MAX_PDF_PAGES = 200


@dataclass
class UploadResult:
    filename: str
    status: str
    digest: str = ""
    chunks: list[Document] = field(default_factory=list)
    message: str = ""


def parse_uploaded_file(filename: str, content: bytes) -> UploadResult:
    """Extract and chunk a PDF/TXT payload without writing it to disk."""
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1].strip() or "unnamed"
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in {"pdf", "txt"}:
        return UploadResult(name, "unsupported", message="Only PDF and TXT files are supported.")
    if not content:
        return UploadResult(name, "empty", message="The file is empty.")
    if len(content) > MAX_UPLOAD_FILE_BYTES:
        return UploadResult(name, "too_large", message="File exceeds the upload size limit.")

    digest = hashlib.sha256(content).hexdigest()
    try:
        if suffix == "txt":
            text = content.decode("utf-8-sig").strip()
        else:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content), strict=False)
            if reader.is_encrypted:
                try:
                    if reader.decrypt("") == 0:
                        return UploadResult(name, "unreadable", digest, message="Encrypted PDFs are not supported.")
                except Exception:  # noqa: BLE001 - normalize varying encrypted-PDF failures.
                    return UploadResult(name, "unreadable", digest, message="Encrypted PDFs are not supported.")
            if len(reader.pages) > MAX_PDF_PAGES:
                return UploadResult(name, "too_large", digest, message=f"PDF exceeds {MAX_PDF_PAGES} pages.")
            text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
    except UnicodeDecodeError:
        return UploadResult(name, "unreadable", digest, message="TXT files must be valid UTF-8 text.")
    except Exception:  # noqa: BLE001 - parser errors vary by malformed PDF implementation.
        return UploadResult(name, "unreadable", digest, message="The file could not be read as a valid PDF/TXT document.")

    if not text:
        message = "No extractable text was found in the document. Scanned/image-only PDFs are not OCR processed."
        return UploadResult(name, "empty", digest, message=message)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=min(CHUNK_OVERLAP, max(0, CHUNK_SIZE - 1)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    source = f"Uploaded: {name}"
    document = Document(page_content=text, metadata={"source": source, "upload_sha256": digest})
    chunks = splitter.split_documents([document])
    return UploadResult(name, "processed", digest, chunks, f"Processed into {len(chunks)} chunks.")


def prepare_upload_batch(
    files: Iterable[tuple[str, bytes]],
    existing_digests: set[str],
    max_documents: int = MAX_UPLOAD_DOCUMENTS,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Parse a batch and return new per-session records plus safe file statuses."""
    accepted: dict[str, dict[str, Any]] = {}
    statuses: list[dict[str, str]] = []
    known = set(existing_digests)
    for filename, content in files:
        result = parse_uploaded_file(filename, content)
        if result.status != "processed":
            statuses.append({"filename": result.filename, "status": result.status, "message": result.message})
            continue
        if result.digest in known:
            statuses.append({"filename": result.filename, "status": "duplicate", "message": "This document is already uploaded in this session."})
            continue
        if len(existing_digests) + len(accepted) >= max_documents:
            statuses.append({"filename": result.filename, "status": "limit", "message": "The session document limit has been reached."})
            continue
        known.add(result.digest)
        accepted[result.digest] = {
            "filename": result.filename,
            "chunks": result.chunks,
            "chunk_count": len(result.chunks),
        }
        statuses.append({"filename": result.filename, "status": "ready", "message": result.message})
    return accepted, statuses


def build_uploaded_index(chunks: Iterable[Document], embeddings: Any) -> Any | None:
    """Build an in-memory FAISS index for this Streamlit session only."""
    documents = list(chunks)
    if not documents:
        return None
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(documents=documents, embedding=embeddings)
