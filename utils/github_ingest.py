"""Import public GitHub source files as citation-aware RAG documents."""
from __future__ import annotations

import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ALLOWED = {".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".html", ".css", ".sql", ".sh", ".ipynb"}
SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "dist", "build", "__pycache__", ".next", "vendor", "coverage"}
SECRET_NAMES = re.compile(r"(^|/)(\.env[^/]*|.*secret.*|.*credential.*|.*private.?key.*|id_rsa|id_ed25519)(/|$)", re.I)
MAX_FILES = 80
MAX_FILE_BYTES = 120_000
MAX_TOTAL_BYTES = 1_500_000

class GitHubImportError(ValueError):
    pass

def parse_repo_url(value: str) -> tuple[str, str]:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"https", "http"} or parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise GitHubImportError("Enter a GitHub repository URL such as https://github.com/owner/repo.")
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2 or any(p in {".", ".."} for p in parts[:2]):
        raise GitHubImportError("The URL must identify an owner and repository.")
    return parts[0], parts[1].removesuffix(".git")

def _api_json(url: str, token: str = ""):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "HelpDeskAI-RepoImporter", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=15) as response:
            import json
            return json.loads(response.read(2_000_000).decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            raise GitHubImportError("Repository not found, inaccessible, or private. For private repositories configure a read-only GITHUB_TOKEN secret.") from None
        if exc.code == 403:
            raise GitHubImportError("GitHub API rate limit or access restriction. Try later or configure a read-only token.") from None
        raise GitHubImportError(f"GitHub returned HTTP {exc.code}.") from None
    except (URLError, TimeoutError):
        raise GitHubImportError("Could not reach GitHub. Check the URL and try again.") from None

def import_repository(value: str, embeddings, token: str = "") -> tuple[list[Document], dict]:
    owner, repo = parse_repo_url(value)
    root = f"https://api.github.com/repos/{owner}/{repo}"
    metadata = _api_json(root, token)
    branch = metadata.get("default_branch") or "main"
    tree = _api_json(f"{root}/git/trees/{branch}?recursive=1", token)
    items = tree.get("tree", [])
    if tree.get("truncated"):
        raise GitHubImportError("This repository is too large for a complete tree listing.")
    candidates = []
    for item in items:
        path = item.get("path", "")
        parts = path.split("/")
        if item.get("type") != "blob" or SECRET_NAMES.search(path) or any(p in SKIP_DIRS for p in parts):
            continue
        if any(p.lower().endswith((".lock", ".min.js", ".min.css")) for p in parts):
            continue
        dot = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if dot in ALLOWED and int(item.get("size", 0) or 0) <= MAX_FILE_BYTES:
            candidates.append(item)
    docs = []
    total = 0
    for item in candidates[:MAX_FILES]:
        path = item["path"]
        data = _api_json(f"{root}/contents/{path}?ref={branch}", token)
        if data.get("encoding") != "base64":
            continue
        import base64
        try:
            raw = base64.b64decode(data.get("content", ""))
            if total + len(raw) > MAX_TOTAL_BYTES:
                break
            text = raw.decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        total += len(raw)
        if not text.strip():
            continue
        lines = text.splitlines()
        source = f"https://github.com/{owner}/{repo}/blob/{branch}/{path}"
        docs.append(Document(page_content=text, metadata={"source": f"{owner}/{repo}/{path}", "url": source, "repo": f"{owner}/{repo}", "path": path, "branch": branch, "start_line": 1, "end_line": len(lines)}))
    if not docs:
        raise GitHubImportError("No eligible readable source or documentation files were found.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80, add_start_index=True, separators=["\n\n", "\n", " ", ""])
    chunks = []
    for doc in docs:
        for chunk in splitter.split_documents([doc]):
            start = int(chunk.metadata.get("start_index", 0))
            prefix = doc.page_content[:start]
            line_start = prefix.count("\n") + 1
            line_end = line_start + chunk.page_content.count("\n")
            chunk.metadata["start_line"] = line_start
            chunk.metadata["end_line"] = line_end
            chunk.metadata["source"] = f"{doc.metadata['source']}:{line_start}-{line_end}"
            chunks.append(chunk)
    return chunks, {"repo": f"{owner}/{repo}", "branch": branch, "files": len(docs), "chunks": len(chunks), "bytes": total}
