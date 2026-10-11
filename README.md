# HelpDesk AI — Enterprise RAG Assistant

<p align="center">
  **Customer support answers grounded in your knowledge base**

  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.45%2B-red.svg)](https://streamlit.io)
  [![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green.svg)](https://langchain.com)
  [![FAISS](https://img.shields.io/badge/FAISS-1.11%2B-yellow.svg)](https://github.com/facebookresearch/faiss)
  [![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
</p>

---

## Overview

HelpDesk AI is a Streamlit customer-support assistant built around retrieval-augmented generation (RAG). It searches the repository's knowledge documents using the existing FAISS index and local `sentence-transformers/all-MiniLM-L6-v2` embeddings. When configured, it sends retrieved evidence and the question to NVIDIA's OpenAI-compatible hosted chat API. Answers are instructed to cite retrieved passages; the application refuses weak matches and rejects missing or unknown citation markers.

The quality of the answers depends on the contents of the knowledge base, the embedding/index compatibility, retrieval behavior, and the configured model. Citations and retrieval labels are not guarantees that an answer is correct. Review answers before using them for high-impact support decisions.

## Features

- Semantic retrieval over the existing FAISS index and text knowledge base.
- Grounded response instructions, explicit out-of-scope refusal, and evidence citations.
- NVIDIA-compatible chat-completions client with configurable model ID, bounded timeout/retries, and safe error categories.
- Premium responsive Streamlit chat UI, starter questions, status panel, session metrics, clear-chat, and text export.
- Input validation and bounded per-session conversation history.
- Local/offline automated tests for validation, RAG evidence/citations, and mocked provider success/failure paths.
- No NVIDIA model ID is assumed. Configure an exact model ID supported by your NVIDIA account.

## Architecture

```text
Streamlit UI (app.py)
  └── Input/session policy
      └── RAG service (utils/rag.py)
          ├── Existing FAISS index + local Hugging Face sentence embeddings
          ├── Weak-match abstention and source-ID mapping
          └── NVIDIA OpenAI-compatible API (utils/nvidia_client.py)

Offline ingestion: ingest.py → source documents → chunks → embeddings → FAISS
```

### Knowledge and index compatibility

The existing `data/*.txt` documents and `faiss_index/index.faiss` / `index.pkl` are preserved. Do not change the embedding model without rebuilding and evaluating the index. The checked-in LangChain FAISS persistence includes a pickle-based docstore; only load trusted repository-built artifacts. Do not upload or load user-provided pickle indexes. The application does not automatically rebuild/overwrite the index at web startup.

## Repository structure

```text
app.py                 Streamlit UI
config.py              Environment and Streamlit Secrets configuration
ingest.py              Offline FAISS ingestion
requirements.txt       Runtime dependencies
requirements-dev.txt   Test and lint tools
.env.example           Non-secret local configuration template
README.md              Project and setup documentation
LICENSE                Existing MIT license

data/                  Existing knowledge-base text files
faiss_index/           Existing FAISS and LangChain docstore artifacts
utils/rag.py           Retrieval, abstention, citations, and generation
utils/nvidia_client.py NVIDIA-compatible API client
utils/input_policy.py  Question and session bounds
utils/prompts.py       Grounded response policy
utils/helpers.py       Chat export and shared helpers
tests/                 Offline unit tests
```

## Installation and local setup

Python 3.10+ is required. The local embedding model is fetched through the Hugging Face ecosystem when first initialized; the LLM itself is not downloaded or run locally.

```bash
git clone https://github.com/Kraverse/HelpDesk-AI-Enterprise-RAG-Assistant.git
cd HelpDesk-AI-Enterprise-RAG-Assistant

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env                 # optional; never commit .env
```

Set `NVIDIA_MODEL` to the exact supported model ID confirmed for your NVIDIA API account. Set `NVIDIA_API_KEY` in `.env` for local use or in Streamlit Community Cloud Secrets for deployment. **Do not paste credentials into source code, commit history, logs, or chat.** Without both settings, the app can initialize local retrieval but answer generation reports a safe configuration message.

```bash
streamlit run app.py
```

### Existing FAISS index

The index is checked in. On first initialization, the local embedding model must be available and compatible. If the artifacts are missing or fail to load, review the error and the embedding/index provenance before running `python ingest.py`; preserve a backup of the existing index before any re-ingestion. Re-ingest only after reviewing the source corpus and model/chunk settings.

## Configuration

| Variable | Purpose | Default / required |
|---|---|---|
| `NVIDIA_API_KEY` | Secret for server-side NVIDIA hosted inference | Required only for live generation |
| `NVIDIA_MODEL` | Exact chat model identifier accepted by your NVIDIA account | **No default; configure explicitly** |
| `NVIDIA_BASE_URL` | OpenAI-compatible NVIDIA API base URL | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_TIMEOUT_SECONDS` | Upper-bounded request timeout | `30` seconds |
| `NVIDIA_MAX_RETRIES` | Bounded retry count | `1` |
| `EMBEDDING_MODEL` | Local query/document embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `FAISS_PATH` | FAISS artifact directory | `faiss_index` |
| `TOP_K` | Maximum retrieved passages (bounded by app) | `3` |
| `MAX_L2_DISTANCE` | Weak-match cutoff in the current FAISS distance space | `1.5`; evaluate/calibrate before changing |
| `MAX_QUESTION_CHARS` | Input length bound | `1000` |
| `MAX_CHAT_MESSAGES` | Maximum retained session messages | `40` |
| `MAX_SESSION_CHARS` | Maximum retained session text | `30000` |
| `SESSION_REQUESTS_PER_MINUTE` | Session-local request ceiling | `20` |
| `ASK_KRAVERSE_URL` | Optional, verified HTTPS link to existing assistant | Unset until canonical URL is confirmed |

The NVIDIA endpoint and API protocol are configurable, but this repository intentionally does not guess a model ID. The inspected `MedRag-X` project used Groq rather than NVIDIA; it is not a source for NVIDIA model configuration.

The `ASK_KRAVERSE_URL` setting is blank by default. The app only renders a link when a valid HTTPS URL is explicitly configured. Do not set it until the assistant's canonical address is verified; iframe embedding or API integration has not been established.

## Deployment: Streamlit Community Cloud

Keep the current Streamlit deployment unless a separately reviewed migration is justified.

1. Confirm the app's repository and branch selection before changing deployment configuration. `master` contains the app in the inspected repository; `main` has historically contained only a README.
2. Deploy from the reviewed feature branch/PR only after the required checks pass and the repository owner approves the change.
3. In the app's **Settings → Secrets**, configure `NVIDIA_API_KEY` and the confirmed `NVIDIA_MODEL`. Do not put secrets in GitHub Actions variables unless the workflow requires them.
4. Verify the deployment's build logs and perform a live end-to-end smoke query with known evidence. A successful build alone does not verify retrieval or generation.
5. Preserve the prior deployment revision and existing index so the deployment can be rolled back.

The app does not store durable chat history; the transcript and counters are session state. The request ceiling is per Streamlit session, not a substitute for account-wide abuse prevention. Questions and retrieved evidence are sent to NVIDIA when generation is enabled. Avoid entering sensitive personal or payment information.

## Tests

The test suite uses no live API key or network calls; provider requests are mocked.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q app.py config.py ingest.py utils tests
```

Tests cover input/session/request bounds, provider payload/auth/error sanitization, retrieval abstention, citation validation, and Streamlit AppTest flows for starter questions, chat input, export, and clearing a session. They do not replace a real compatibility check of the committed FAISS artifacts or a live NVIDIA smoke test after configuration.

## Screenshots

<!-- Add current screenshots after a reviewed deployment.
<img src="assets/screenshot-1.png" alt="Chat Interface" width="800" />
<img src="assets/screenshot-2.png" alt="Dashboard Sidebar" width="300" />
-->

## Contributing

1. Create a feature branch from the current application branch.
2. Keep secrets and generated customer data out of commits.
3. Run the tests and include actual test output in the pull request.
4. Explain any changes to the knowledge documents, embedding model, or FAISS index.
5. Open a pull request; do not deploy directly from an unreviewed branch.

Please run `ruff check .` where available and include appropriate tests and documentation.

## License

This project is licensed under the MIT License — see the preserved [LICENSE](LICENSE) file for details.

---

<p align="center">Built with Streamlit, LangChain, FAISS, and configurable inference providers</p>
