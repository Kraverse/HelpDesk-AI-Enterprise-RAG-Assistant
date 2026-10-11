"""
==========================================================
HelpDesk AI – Enterprise RAG Assistant

Streamlit Application — Professional Chat Interface
==========================================================
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so sibling imports work everywhere
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    APP_NAME,
    APP_SUBTITLE,
    LLM_REPO,
    EMBEDDING_MODEL,
    FAISS_PATH,
)
from utils.rag import initialize_system, query as rag_query
from utils.helpers import (
    get_time,
    response_time,
    format_sources,
    chat_history_to_string,
    get_confidence_color,
)

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title=f"{APP_NAME} — {APP_SUBTITLE}",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS — Dark Enterprise Theme
# ============================================================

CUSTOM_CSS = """
/* ---- Global ---- */
[data-testid="stAppViewContainer"] {
    background-color: #0d1117;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}

/* ---- Main Area ---- */
[data-testid="stMain"] {
    background-color: #0d1117;
}

/* ---- Chat Input ---- */
[data-testid="stChatInput"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
}
[data-testid="stChatInput"] textarea {
    color: #e6edf3 !important;
    background-color: transparent !important;
}

/* ---- Headers ---- */
h1, h2, h3, h4, h5, h6 {
    color: #e6edf3 !important;
}

/* ---- Status Cards ---- */
.status-card {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
}

.status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 6px 0;
    font-size: 14px;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-ready { background-color: #3fb950; box-shadow: 0 0 6px #3fb95088; }
.dot-error { background-color: #f85149; box-shadow: 0 0 6px #f8514988; }
.dot-warn  { background-color: #d29922; box-shadow: 0 0 6px #d2992288; }

/* ---- Metric Tiles ---- */
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin: 8px 0;
}

.metric-tile {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
}

.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #58a6ff;
    line-height: 1.2;
}

.metric-label {
    font-size: 10px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 2px;
}

/* ---- Source Badges ---- */
.source-badge {
    display: inline-block;
    background-color: rgba(31, 111, 235, 0.15);
    color: #58a6ff;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 12px;
    margin: 2px 4px 2px 0;
    border: 1px solid rgba(31, 111, 235, 0.25);
}

/* ---- Confidence Bar ---- */
.confidence-track {
    width: 100%;
    height: 6px;
    background-color: #21262d;
    border-radius: 3px;
    overflow: hidden;
    margin: 4px 0 2px;
}

.confidence-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}

/* ---- Meta Row (time + confidence) ---- */
.meta-row {
    display: flex;
    gap: 16px;
    align-items: center;
    margin-top: 8px;
    font-size: 12px;
    color: #8b949e;
}

.meta-item {
    display: flex;
    align-items: center;
    gap: 4px;
}

/* ---- Typing Animation ---- */
@keyframes pulse-dot {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1); }
}

.typing-indicator span {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #58a6ff;
    margin: 0 3px;
    animation: pulse-dot 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

/* ---- Suggested Question Buttons ---- */
.suggested-btn button {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #58a6ff !important;
    border-radius: 20px !important;
    padding: 8px 18px !important;
    font-size: 13px !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.suggested-btn button:hover {
    background-color: rgba(31, 111, 235, 0.12) !important;
    border-color: #58a6ff !important;
    color: #79c0ff !important;
}

/* ---- Action Buttons ---- */
.action-btn button {
    width: 100%;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    transition: all 0.2s ease !important;
}

/* ---- Hide Default Chrome ---- */
header[data-testid="stHeader"] {
    background-color: rgba(13, 17, 23, 0.8);
    backdrop-filter: blur(12px);
}
[data-testid="stToolbar"] {
    visibility: hidden;
}
footer[data-testid="stFooter"] {
    background-color: #0d1117;
    padding: 8px 0;
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* ---- Divider ---- */
.sidebar-divider {
    border: none;
    border-top: 1px solid #30363d;
    margin: 16px 0;
}

/* ---- Tech tag ---- */
.tech-tag {
    display: inline-block;
    background-color: rgba(31, 111, 235, 0.1);
    color: #8b949e;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-family: monospace;
    margin: 2px 0;
}
"""


def inject_custom_css() -> None:
    """Inject the dark enterprise theme."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================

def init_session_state() -> None:
    """Seed session-state defaults on first run."""
    defaults: Dict[str, Any] = {
        "chat_history": [],
        "question_count": 0,
        "total_response_time": 0.0,
        "system_initialized": False,
        "system_status": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# Sidebar
# ============================================================

SUGGESTED_QUESTIONS: List[str] = [
    "What are your business hours?",
    "How can I contact support?",
    "Do you ship internationally?",
    "What is the return policy?",
    "Can I track my order?",
    "How are refunds issued?",
]


def _status_icon(ready: bool, warning: bool = False) -> str:
    """Return an HTML status dot."""
    if warning:
        return '<span class="status-dot dot-warn"></span>'
    if ready:
        return '<span class="status-dot dot-ready"></span>'
    return '<span class="status-dot dot-error"></span>'


def render_sidebar(status: Dict[str, Any]) -> None:
    """Draw the sidebar with dashboard, analytics, and actions."""
    with st.sidebar:

        # ---- Brand ----
        st.markdown(
            '<div style="margin: 4px 0 12px;">'
            '<span style="font-size:24px;">🤖</span> '
            '<span style="font-size:20px;font-weight:700;color:#e6edf3;">'
            'HelpDesk AI</span><br>'
            '<span style="font-size:12px;color:#8b949e;">'
            'Enterprise RAG Assistant</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ---- System Status ----
        llm_is_token_issue = (
            "TOKEN" in str(status.get("error", "")).upper()
        )

        st.markdown(
            '<div class="status-card">'
            '<div style="font-size:11px;color:#8b949e;'
            'text-transform:uppercase;letter-spacing:1px;'
            'margin-bottom:10px;">System Status</div>'

            f'<div class="status-row">'
            f'{_status_icon(status.get("embeddings_ready"))}'
            f'<span>Embeddings</span></div>'

            f'<div class="status-row">'
            f'{_status_icon(status.get("index_ready"))}'
            f'<span>Vector Index</span></div>'

            f'<div class="status-row">'
            f'{_status_icon(status.get("llm_ready"), warning=llm_is_token_issue)}'
            f'<span>LLM</span></div>'

            '</div>',
            unsafe_allow_html=True,
        )

        if status.get("error"):
            st.caption(
                f"⚠️ {status['error']}",
            )

        # ---- Knowledge Base ----
        doc_count = status.get("doc_count", 0)
        chunk_count = status.get("chunk_count", 0)

        st.markdown(
            '<div class="status-card">'
            '<div style="font-size:11px;color:#8b949e;'
            'text-transform:uppercase;letter-spacing:1px;'
            'margin-bottom:8px;">Knowledge Base</div>'

            f'<div style="font-size:13px;">📄 Documents: '
            f'<b style="color:#58a6ff;">{doc_count}</b></div>'

            f'<div style="font-size:13px;">🧩 Chunks: '
            f'<b style="color:#58a6ff;">{chunk_count}</b></div>'

            f'<div style="font-size:13px;">📊 Index: '
            f'<b style="color:#58a6ff;">'
            f'{"Ready" if status.get("index_ready") else "Missing"}'
            f'</b></div>'

            '</div>',
            unsafe_allow_html=True,
        )

        # ---- Analytics ----
        q_count = st.session_state.get("question_count", 0)
        avg_time = (
            round(st.session_state["total_response_time"] / q_count, 2)
            if q_count > 0
            else 0.0
        )

        st.markdown(
            '<div class="metrics-grid">'
            '<div class="metric-tile">'
            f'<div class="metric-value">{q_count}</div>'
            '<div class="metric-label">Questions</div>'
            '</div>'
            '<div class="metric-tile">'
            f'<div class="metric-value">{avg_time}s</div>'
            '<div class="metric-label">Avg Time</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ---- Tech Stack ----
        st.markdown(
            '<div class="status-card">'
            '<div style="font-size:11px;color:#8b949e;'
            'text-transform:uppercase;letter-spacing:1px;'
            'margin-bottom:8px;">Tech Stack</div>'

            f'<div class="tech-tag">{status.get("llm_name", LLM_REPO)}</div><br>'
            f'<div class="tech-tag">{status.get("embedding_model", EMBEDDING_MODEL)}</div>'

            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ---- Actions ----
        col_a, col_b = st.columns(2)

        with col_a:
            if st.button(
                "🗑️ Clear Chat",
                use_container_width=True,
                key="clear_chat",
            ):
                st.session_state["chat_history"] = []
                st.session_state["question_count"] = 0
                st.session_state["total_response_time"] = 0.0
                st.rerun()

        with col_b:
            if st.session_state.get("chat_history"):
                export_text = chat_history_to_string(
                    st.session_state["chat_history"]
                )
                st.download_button(
                    label="💾 Save",
                    data=export_text.encode("utf-8"),
                    file_name=f"helpdesk-chat-{get_time().replace(' ', '-')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="download_chat",
                )

        # ---- Footer ----
        st.markdown(
            '<div style="margin-top:24px;text-align:center;'
            'font-size:11px;color:#484f58;">'
            'HelpDesk AI v2.0 · RAG-Powered<br>'
            'Built with Streamlit + LangChain + FAISS'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# Response Metadata Renderer
# ============================================================

def render_response_metadata(
    sources: List[str],
    confidence: float,
    resp_time: float,
) -> None:
    """Show source badges, confidence bar, and response time."""
    # Sources
    if sources:
        badges_html = " ".join(
            f'<span class="source-badge">📄 {s}</span>' for s in sources
        )
        st.markdown(
            f'<div style="margin-top:8px;">{badges_html}</div>',
            unsafe_allow_html=True,
        )

    # Confidence + Time
    color = get_confidence_color(confidence)
    st.markdown(
        f'<div class="meta-row">'
        f'<div class="meta-item">'
        f'<div class="confidence-track">'
        f'<div class="confidence-fill" style="width:{confidence}%;'
        f'background-color:{color};"></div>'
        f'</div>'
        f'<span>Confidence {confidence}%</span>'
        f'</div>'
        f'<div class="meta-item">⏱️ {resp_time}s</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Chat Interface
# ============================================================

def render_chat() -> None:
    """Draw the main chat area with history and input."""
    # Title
    st.markdown(
        '<div style="margin: 16px 0 8px;">'
        f'<span style="font-size:28px;font-weight:700;color:#e6edf3;">'
        f'💬 {APP_NAME}</span><br>'
        f'<span style="font-size:14px;color:#8b949e;">'
        f'{APP_SUBTITLE} — Ask anything about our products and services</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Render history
    for msg in st.session_state.chat_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                render_response_metadata(
                    sources=msg.get("sources", []),
                    confidence=msg.get("confidence", 0.0),
                    resp_time=msg.get("response_time", 0.0),
                )
                if msg.get("error"):
                    st.caption(f"⚠️ {msg['error']}")

    # Suggested questions (only when chat is empty)
    if not st.session_state.chat_history:
        st.markdown(
            '<div style="margin: 20px 0 8px; font-size:13px; '
            'color:#8b949e;">💡 Try asking:</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for idx, question in enumerate(SUGGESTED_QUESTIONS):
            with cols[idx % 3]:
                if st.button(
                    question,
                    key=f"suggest_{idx}",
                    use_container_width=True,
                ):
                    _handle_question(question)

    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        _handle_question(prompt)


def _handle_question(prompt: str) -> None:
    """Process a user question through the RAG pipeline."""
    # User message
    st.session_state.chat_history.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant thinking
    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown(
            '<div class="typing-indicator">'
            '<span></span><span></span><span></span>'
            '</div>',
            unsafe_allow_html=True,
        )

        start = time.time()
        resp = rag_query(prompt)
        elapsed = response_time(start)

        thinking.empty()

        if resp.answer:
            st.markdown(resp.answer)

        render_response_metadata(
            sources=resp.sources,
            confidence=resp.confidence,
            resp_time=resp.response_time,
        )

        if resp.error:
            st.caption(f"⚠️ {resp.error}")

    # Persist in session state
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": resp.answer,
            "sources": resp.sources,
            "confidence": resp.confidence,
            "response_time": resp.response_time,
            "error": resp.error,
        }
    )
    st.session_state.question_count += 1
    st.session_state.total_response_time += elapsed


# ============================================================
# Footer
# ============================================================

def render_footer() -> None:
    """Application footer."""
    st.markdown(
        '<div style="text-align:center;margin:32px 0 8px;'
        'font-size:11px;color:#484f58;">'
        'HelpDesk AI — Enterprise RAG Assistant · v2.0<br>'
        'Retrieval-Augmented Generation · '
        f'{get_time()}'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Application entry point."""
    inject_custom_css()
    init_session_state()

    # Bootstrap RAG on first run
    if not st.session_state.system_initialized:
        with st.spinner("Initializing HelpDesk AI..."):
            st.session_state.system_status = initialize_system()
            st.session_state.system_initialized = True

    status = st.session_state.system_status
    render_sidebar(status)
    render_chat()
    render_footer()


if __name__ == "__main__":
    main()