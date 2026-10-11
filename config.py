"""
HelpDesk AI - Enterprise RAG Assistant
Application Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ==============================
# Hugging Face
# ==============================

HF_TOKEN = os.getenv("HF_TOKEN")

LLM_REPO = "Qwen/Qwen2.5-7B-Instruct"

# ==============================
# Embedding Model
# ==============================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==============================
# Vector Store
# ==============================

FAISS_PATH = "faiss_index"

# ==============================
# Text Splitting
# ==============================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# ==============================
# Retrieval
# ==============================

TOP_K = 3

# ==============================
# LLM Settings
# ==============================

TEMPERATURE = 0.2

MAX_NEW_TOKENS = 512

# ==============================
# UI
# ==============================

APP_NAME = "HelpDesk AI"

APP_SUBTITLE = "Enterprise RAG Assistant"