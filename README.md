# HelpDesk AI — Enterprise RAG Assistant

<p align="center">
  <img src="assets/chatbot.png" alt="HelpDesk AI" width="120" height="120" />

  **Intelligent Customer Support Powered by Retrieval-Augmented Generation**

  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.45%2B-red.svg)](https://streamlit.io)
  [![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green.svg)](https://langchain.com)
  [![FAISS](https://img.shields.io/badge/FAISS-1.11%2B-yellow.svg)](https://github.com/facebookresearch/faiss)
  [![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
</p>

---

## Overview

HelpDesk AI is an enterprise-grade customer support assistant that uses **Retrieval-Augmented Generation (RAG)** to answer questions exclusively from your company's knowledge base. It combines the power of large language models with precise document retrieval to deliver accurate, hallucination-free responses.

Unlike generic chatbots, HelpDesk AI will **never answer outside its knowledge base** — ensuring every response is grounded in your actual documentation.

---

## Features

- **RAG Pipeline** — Semantic search over your documents with FAISS vector indexing
- **Hallucination Prevention** — Strict context-only response generation via engineered prompts
- **Source Citations** — Every answer shows which documents were referenced
- **Confidence Scoring** — L2-distance-based confidence metrics displayed in real time
- **Professional Dashboard** — Dark enterprise theme with live analytics sidebar
- **Chat Export** — Download full conversation history as a text file
- **Modular Architecture** — Clean separation of config, ingestion, RAG engine, and UI
- **Error Resilience** — Graceful handling of missing tokens, corrupted indexes, and network failures
- **Production-Ready** — Type hints, docstrings, logging, and PEP 8 compliance throughout

---

## Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│   User       │────▶│  Streamlit    │────▶│  RAG Engine  │
│   Interface  │◀────│   app.py      │◀────│  rag.py      │
└──────────────┘     └───────────────┘     └──────┬───────┘
                                                  │
                              ┌───────────────────┼───────────────────┐
                              │                   │                   │
                       ┌──────▼──────┐    ┌───────▼──────┐    ┌──────▼──────┐
                       │   FAISS     │    │  HuggingFace │    │   Prompt    │
                       │  Vector DB  │    │  Inference   │    │  Template   │
                       └──────┬──────┘    │  API (LLM)   │    └─────────────┘
                              │            └──────────────┘
                       ┌──────▼──────┐
                       │  Sentence   │
                       │ Transformers│
                       │ (Embeddings)│
                       └─────────────┘
```

### How RAG Works

1. **Ingestion** — Knowledge-base `.txt` files are split into chunks and embedded into a FAISS vector index using sentence-transformers.
2. **Retrieval** — When a user asks a question, the query is embedded and the top-K most similar chunks are retrieved via L2-distance search.
3. **Generation** — Retrieved chunks are injected into a carefully engineered prompt and sent to the LLM, which produces a grounded answer.

---

## Folder Structure

```
HelpDesk-AI-Enterprise-RAG-Assistant/
├── app.py                # Streamlit chat interface
├── ingest.py             # Builds the FAISS vector database
├── config.py             # Centralized configuration
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── LICENSE               # MIT License
├── .gitignore            # Git ignore rules
├── .env.example          # Environment variable template
├── .env                  # Your secrets (never committed)
│
├── data/                 # Knowledge-base documents
│   ├── faq.txt
│   ├── shipping.txt
│   └── return_policy.txt
│
├── faiss_index/          # Generated vector index
│   ├── index.faiss
│   └── index.pkl
│
├── utils/                # Utility modules
│   ├── __init__.py
│   ├── helpers.py        # Helper functions
│   ├── prompts.py        # Prompt templates
│   └── rag.py            # RAG engine core
│
├── assets/               # Images and media
└── logs/                 # Application logs
```

---

## Tech Stack

| Layer            | Technology                          |
| ---------------- | ----------------------------------- |
| UI               | Streamlit 1.45+                     |
| LLM              | Qwen/Qwen2.5-7B-Instruct (HF API)  |
| Embeddings       | all-MiniLM-L6-v2 (Sentence TF)     |
| Vector Store     | FAISS (Facebook AI)                |
| Orchestration    | LangChain 0.3+                     |
| Environment      | python-dotenv                      |
| Language         | Python 3.10+                       |

---

## Installation

### Prerequisites

- Python 3.10 or newer
- A [Hugging Face](https://huggingface.co/) account with an API token
- (Optional) GPU for local inference

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/helpdesk-ai-enterprise-rag-assistant.git
cd helpdesk-ai-enterprise-rag-assistant

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and paste your Hugging Face token

# 5. Build the vector database
python ingest.py

# 6. Launch the application
streamlit run app.py
```

---

## Environment Variables

| Variable   | Description                        | Required |
| ---------- | ---------------------------------- | -------- |
| `HF_TOKEN` | Your Hugging Face API access token | Yes      |

Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## Deployment

### Streamlit Community Cloud

```bash
# requirements.txt already includes all dependencies
# Set HF_TOKEN as a Streamlit Community Cloud secret
# Deploy directly from your GitHub repo
```

### Hugging Face Spaces

```bash
# Create a new Space with Streamlit SDK
# Add HF_TOKEN as a Space secret
# Push your code
```

### Render / Railway

```bash
# Add a start command: streamlit run app.py --server.port $PORT
# Set HF_TOKEN as an environment variable
```

---

## Screenshots

<!-- Add screenshots here after deployment
<img src="assets/screenshot-1.png" alt="Chat Interface" width="800" />
<img src="assets/screenshot-2.png" alt="Dashboard Sidebar" width="300" />
-->

---

## Future Improvements

- [ ] Multi-format document support (PDF, DOCX, Markdown)
- [ ] Conversation memory with session history
- [ ] Admin panel for knowledge-base management
- [ ] A/B testing for prompt templates
- [ ] Rate limiting and usage analytics
- [ ] Multi-language support
- [ ] Integration with ticketing systems (Zendesk, Freshdesk)
- [ ] Local LLM fallback (Ollama / llama.cpp)

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

Please ensure your code passes `ruff check .` and includes appropriate docstrings.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ using Streamlit, LangChain & FAISS
</p>