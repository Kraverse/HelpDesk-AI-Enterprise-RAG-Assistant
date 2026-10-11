"""HelpDesk AI Streamlit interface."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    APP_NAME,
    APP_SUBTITLE,
    ASK_KRAVERSE_URL,
    EMBEDDING_MODEL,
    LLM_REPO,
    MAX_CHAT_MESSAGES,
    MAX_QUESTION_CHARS,
    MAX_UPLOAD_DOCUMENTS,
    MAX_UPLOAD_FILE_BYTES,
    OPENROUTER_MODEL,
)
from utils.document_uploads import build_uploaded_index, prepare_upload_batch
from utils.helpers import chat_history_to_string, get_time
from utils.input_policy import bound_history, register_request, validate_question
from utils.rag import get_embeddings, initialize_system
from utils.rag import query as rag_query

st.set_page_config(
    page_title=f"{APP_NAME} — {APP_SUBTITLE}",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root { --ink:#17223b; --muted:#68758b; --accent:#5b68df; --surface:#fff; --line:#e6eaf2; }
[data-testid="stAppViewContainer"] { background:linear-gradient(135deg,#f6f8fc 0%,#f2f5fb 55%,#f8f7ff 100%); color:var(--ink); }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
.block-container { max-width:1120px; padding-top:2rem; padding-bottom:2rem; }
.hero { padding:1.5rem 1.7rem; border:1px solid #e7eafd; border-radius:22px; background:linear-gradient(110deg,#fff 0%,#f4f6ff 100%); box-shadow:0 14px 42px rgba(36,49,96,.06); margin-bottom:1.25rem; }
.hero h1 { font-size:2rem; letter-spacing:-.04em; margin:0; color:#182542; }
.hero p { color:#68758b; margin:.45rem 0 0; }
[data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:18px; background:rgba(255,255,255,.92); box-shadow:0 7px 24px rgba(29,43,81,.04); }
[data-testid="stChatInput"] { border-color:#dce2f2; border-radius:16px; box-shadow:0 8px 26px rgba(42,56,105,.08); }
.stButton>button, .stDownloadButton>button, .stLinkButton> a { border-radius:12px!important; }
[data-testid="stMetric"] { border:1px solid var(--line); background:#fff; border-radius:14px; padding:.6rem .85rem; }
.small-muted { color:#68758b; font-size:.88rem; }
@media (max-width:700px) { .block-container { padding:1rem .75rem 1.2rem; } .hero { padding:1.1rem; border-radius:16px; } .hero h1 {font-size:1.55rem;} }
@media (prefers-reduced-motion: reduce) { *,*::before,*::after { scroll-behavior:auto!important; animation:none!important; transition:none!important; } }
</style>
"""

SUGGESTED_QUESTIONS = [
    "What are your business hours?",
    "How can I contact support?",
    "Do you ship internationally?",
    "What is the return policy?",
    "Can I track my order?",
    "How are refunds issued?",
]


def _verified_assistant_url(value: str) -> str | None:
    """Only allow an explicitly configured HTTPS assistant URL."""
    parsed = urlparse(value)
    if parsed.scheme == "https" and parsed.netloc and not parsed.username and not parsed.password:
        return value
    return None


def init_state() -> None:
    defaults = {
        "chat_history": [],
        "question_count": 0,
        "total_response_time": 0.0,
        "system_initialized": False,
        "system_status": {},
        "request_timestamps": [],
        "uploaded_files": {},
        "uploaded_store": None,
        "upload_statuses": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state.chat_history = bound_history(st.session_state.chat_history)


def render_sources(citations: list[dict[str, str]]) -> None:
    if not citations:
        return
    with st.expander("Sources and evidence", expanded=False):
        for item in citations:
            st.markdown(f"**[{item['id']}]** `{item['source']}`")


def render_sidebar(status: dict) -> None:
    with st.sidebar:
        st.markdown("## ✦ HelpDesk AI")
        st.caption("Enterprise RAG Assistant")
        st.divider()
        st.markdown("### System status")
        st.write(f"Embeddings: {'Ready' if status.get('embeddings_ready') else 'Unavailable'}")
        st.write(f"Knowledge index: {'Ready' if status.get('index_ready') else 'Unavailable'}")
        st.write(f"Answer generation: {'Configured' if status.get('llm_ready') else 'Needs configuration'}")
        if status.get("error"):
            st.info(status["error"])
        st.divider()
        st.markdown("### Knowledge base")
        st.caption(f"{status.get('doc_count', 0)} source documents · {status.get('chunk_count', 0)} indexed chunks")
        uploaded_count = len(st.session_state.uploaded_files)
        uploaded_chunks = sum(item["chunk_count"] for item in st.session_state.uploaded_files.values())
        st.caption(f"{uploaded_count} session uploads · {uploaded_chunks} upload chunks")
        st.caption(f"Embeddings: {EMBEDDING_MODEL}")
        st.caption(f"LLM: {OPENROUTER_MODEL or LLM_REPO}")
        if st.session_state.question_count:
            average = st.session_state.total_response_time / st.session_state.question_count
            left, right = st.columns(2)
            left.metric("Questions", st.session_state.question_count)
            right.metric("Avg time", f"{average:.2f}s")
        link = _verified_assistant_url(ASK_KRAVERSE_URL)
        if link:
            st.link_button("Open Ask KraVerse AI", link, use_container_width=True)
        elif ASK_KRAVERSE_URL:
            st.caption("Ask KraVerse link is hidden because the configured URL is not a valid HTTPS URL.")
        st.divider()
        render_upload_section()
        st.divider()
        first, second = st.columns(2)
        with first:
            if st.button("Clear chat", use_container_width=True, key="clear_chat"):
                st.session_state.chat_history = []
                st.session_state.question_count = 0
                st.session_state.total_response_time = 0.0
                st.rerun()
        with second:
            if st.session_state.chat_history:
                content = chat_history_to_string(st.session_state.chat_history)
                st.download_button(
                    "Export", content.encode("utf-8"),
                    file_name=f"helpdesk-chat-{get_time().replace(' ', '-')}.txt",
                    mime="text/plain", use_container_width=True, key="download_chat",
                )
        st.caption(f"Session limit: {MAX_CHAT_MESSAGES} messages · {MAX_QUESTION_CHARS} chars/question")


def render_upload_section() -> None:
    """Render safe ephemeral document upload controls in the existing sidebar."""
    st.markdown("### Add knowledge documents")
    st.caption(
        f"Upload multiple PDF/TXT files (up to {MAX_UPLOAD_FILE_BYTES // (1024 * 1024)} MB each, {MAX_UPLOAD_DOCUMENTS} per session). "
        "Files are processed in memory for this session only; built-in documents are preserved."
    )
    selected = st.file_uploader(
        "Choose PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="knowledge_uploads",
    )
    if st.button("Process selected documents", use_container_width=True, key="process_uploads"):
        if not selected:
            st.warning("Choose at least one PDF or TXT file first.")
        else:
            pending, statuses = prepare_upload_batch(
                [(file.name, file.getvalue()) for file in selected],
                set(st.session_state.uploaded_files),
                MAX_UPLOAD_DOCUMENTS,
            )

            if pending:
                merged = {**st.session_state.uploaded_files, **pending}
                try:
                    chunks = [chunk for item in merged.values() for chunk in item["chunks"]]
                    uploaded_store = build_uploaded_index(chunks, get_embeddings())
                    st.session_state.uploaded_files = merged
                    st.session_state.uploaded_store = uploaded_store
                except Exception:  # noqa: BLE001 - suppress parser/embedding details from UI.
                    statuses = [
                        {"filename": s["filename"], "status": "index_error", "message": "Could not embed/index this session's uploads. Try smaller or readable files."}
                        if s["status"] == "ready" else s
                        for s in statuses
                    ]
            st.session_state.upload_statuses = statuses

    for item in st.session_state.upload_statuses:
        message = f"**{item['filename']}** — {item['message']}"
        if item["status"] == "ready":
            st.success(message)
        elif item["status"] in {"duplicate", "limit", "removed"}:
            st.info(message)
        else:
            st.warning(message)

    if st.session_state.uploaded_files:
        st.markdown("**Uploaded this session**")
        for digest, item in list(st.session_state.uploaded_files.items()):
            left, right = st.columns([4, 1])
            left.caption(f"{item['filename']} · {item['chunk_count']} chunks")
            if right.button("Remove", key=f"remove_upload_{digest[:12]}", help=f"Remove {item['filename']}"):
                remaining = {key: value for key, value in st.session_state.uploaded_files.items() if key != digest}
                try:
                    chunks = [chunk for doc in remaining.values() for chunk in doc["chunks"]]
                    st.session_state.uploaded_store = build_uploaded_index(chunks, get_embeddings())
                    st.session_state.uploaded_files = remaining
                    st.session_state.upload_statuses = [
                        status for status in st.session_state.upload_statuses
                        if status["filename"] != item["filename"]
                    ]
                    st.session_state.upload_statuses.append({
                        "filename": item["filename"],
                        "status": "removed",
                        "message": "Removed from this session's knowledge base.",
                    })
                    st.rerun()
                except Exception:  # noqa: BLE001 - avoid surfacing embedding/index details.
                    st.warning("Could not update the session index. Uploaded documents were kept.")
        if st.button("Clear uploaded knowledge base", use_container_width=True, key="clear_uploads"):
            st.session_state.uploaded_files = {}
            st.session_state.uploaded_store = None
            st.session_state.upload_statuses = []
            st.rerun()


def _handle_question(raw_question: str) -> bool:
    question, validation_error = validate_question(raw_question)
    if validation_error:
        st.warning(validation_error)
        return False
    timestamps, allowed = register_request(
        st.session_state.request_timestamps, time.monotonic()
    )
    st.session_state.request_timestamps = timestamps
    if not allowed:
        st.warning("This session has reached its request limit. Please wait a minute and try again.")
        return False
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        started = time.perf_counter()
        with st.spinner("Searching the knowledge base and preparing a grounded answer…"):
            try:
                if st.session_state.uploaded_store is not None:
                    response = rag_query(question, uploaded_store=st.session_state.uploaded_store)
                else:
                    response = rag_query(question)
            except Exception:  # noqa: BLE001 - UI boundary must not expose uncaught provider details.
                # Never show the exception body (it may contain provider detail).
                safe_error = "The request could not be completed. Please retry."
                st.session_state.chat_history.append({
                    "role": "assistant", "content": safe_error, "error": safe_error,
                    "sources": [], "citations": [], "relevance": "Unavailable",
                    "response_time": round(time.perf_counter() - started, 2),
                })
                st.session_state.chat_history = bound_history(st.session_state.chat_history)
                st.warning(safe_error)
                return True
        elapsed = time.perf_counter() - started
        if response.answer:
            st.markdown(response.answer)
        elif response.error:
            st.warning(response.error)
        render_sources(response.citations)
        st.caption(f"{response.relevance} · {response.response_time:.2f}s")
        if response.error:
            st.info(response.error)
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response.answer,
        "sources": response.sources,
        "citations": response.citations,
        "relevance": response.relevance,
        "response_time": response.response_time,
        "error": response.error,
    })
    st.session_state.chat_history = bound_history(st.session_state.chat_history)
    st.session_state.question_count += 1
    st.session_state.total_response_time += elapsed
    return True


def render_chat() -> None:
    st.markdown(
        f'<div class="hero"><h1>{APP_NAME}</h1><p>{APP_SUBTITLE} · Answers are grounded in the configured help-center documents.</p></div>',
        unsafe_allow_html=True,
    )
    if not st.session_state.chat_history:
        st.markdown("#### What can I help you find?")
        st.caption("Try a suggested question, or ask about policies, shipping, and support.")
        cols = st.columns(3)
        for index, question in enumerate(SUGGESTED_QUESTIONS):
            with cols[index % 3]:
                if (
                    st.button(question, key=f"suggest_{index}", use_container_width=True)
                    and _handle_question(question)
                ):
                    st.rerun()
    for message in st.session_state.chat_history:
        with st.chat_message(message.get("role", "user")):
            st.markdown(message.get("content", ""))
            if message.get("role") == "assistant":
                render_sources(message.get("citations", []))
                if message.get("relevance"):
                    st.caption(f"{message['relevance']} · {message.get('response_time', 0):.2f}s")
                if message.get("error"):
                    st.info(message["error"])
    prompt = st.chat_input("Ask a question about your help-center documents…")
    if prompt and _handle_question(prompt):
        st.rerun()


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    if not st.session_state.system_initialized:
        with st.spinner("Initializing local retrieval components…"):
            st.session_state.system_status = initialize_system()
            st.session_state.system_initialized = True
    render_sidebar(st.session_state.system_status)
    render_chat()
    st.caption("HelpDesk AI · Session-only chat history · Do not enter sensitive personal information.")


if __name__ == "__main__":
    main()
