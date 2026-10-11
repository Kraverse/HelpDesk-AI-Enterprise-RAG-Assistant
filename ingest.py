"""
==========================================================
HelpDesk AI – Enterprise RAG Assistant

Builds the FAISS Vector Database
==========================================================
"""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, FAISS_PATH

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ==========================================================
# Load Documents
# ==========================================================

def load_documents() -> list[Document]:
    """Load all .txt files from the data/ directory.

    Returns:
        List of LangChain Document objects with source metadata.
    Raises:
        FileNotFoundError: If the data/ folder does not exist.
        ValueError: If no .txt files are found.
    """
    data_folder = Path("data")

    if not data_folder.exists():
        raise FileNotFoundError(
            "The data/ folder does not exist. "
            "Create it and add .txt knowledge-base files."
        )

    txt_files = sorted(data_folder.glob("*.txt"))

    if not txt_files:
        raise ValueError(
            "No .txt files found inside data/. "
            "Add at least one knowledge-base file."
        )

    documents: list[Document] = []
    logger.info("Loading documents from %s", data_folder.resolve())

    for file_path in txt_files:
        content = file_path.read_text(encoding="utf-8").strip()

        if not content:
            logger.warning("Skipping empty file: %s", file_path.name)
            continue

        doc = Document(
            page_content=content,
            metadata={"source": file_path.name},
        )
        documents.append(doc)
        logger.info("  Loaded: %s (%d chars)", file_path.name, len(content))

    if not documents:
        raise ValueError("All files in data/ are empty.")

    logger.info("Total documents loaded: %d", len(documents))
    return documents


# ==========================================================
# Split Documents
# ==========================================================

def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked Document objects.
    """
    logger.info("Splitting documents (chunk_size=%d, overlap=%d)", CHUNK_SIZE, CHUNK_OVERLAP)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info("Created %d chunks from %d documents.", len(chunks), len(documents))
    return chunks


# ==========================================================
# Load Embeddings
# ==========================================================

def load_embeddings() -> HuggingFaceEmbeddings:
    """Initialize the sentence-transformer embedding model.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info("Embedding model ready.")
    return embeddings


# ==========================================================
# Create FAISS
# ==========================================================

def build_vector_database(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """Build a FAISS vector store from document chunks.

    Args:
        chunks: List of chunked Document objects.
        embeddings: Embedding model instance.

    Returns:
        FAISS vector store.
    """
    logger.info("Building FAISS vector store from %d chunks...", len(chunks))
    vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vector_db


# ==========================================================
# Save FAISS
# ==========================================================

def save_database(vector_db: FAISS) -> None:
    """Persist the FAISS index to disk.

    Args:
        vector_db: FAISS vector store to save.
    """
    index_path = Path(FAISS_PATH)
    index_path.mkdir(parents=True, exist_ok=True)

    vector_db.save_local(FAISS_PATH)
    logger.info("Vector database saved to %s/", FAISS_PATH)


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    """End-to-end pipeline: load → split → embed → index → save."""
    print("\n===================================")
    print("  HelpDesk AI")
    print("  Enterprise RAG Assistant")
    print("  Building Vector Database")
    print("===================================\n")

    try:
        documents = load_documents()
        chunks = split_documents(documents)
        embeddings = load_embeddings()
        vector_db = build_vector_database(chunks, embeddings)
        save_database(vector_db)

        print("\nVector Database Ready.")
        print(f"  Documents : {len(documents)}")
        print(f"  Chunks    : {len(chunks)}")
        print(f"  Model     : {EMBEDDING_MODEL}")
        print(f"  Location  : {FAISS_PATH}/\n")

    except FileNotFoundError as e:
        logger.error("%s", e)
    except ValueError as e:
        logger.error("%s", e)
    except Exception as e:
        logger.exception("Unexpected error during ingestion: %s", e)


if __name__ == "__main__":
    main()