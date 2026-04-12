"""retrieval.retriever — Chroma-backed indexing and retrieval.

Each store is persisted on disk under:
  <app_root>/.chroma/<store_id>/

This means the vector index survives process restarts.

Structured JSON log emitted for each index and retrieve call.
"""

import hashlib
import json
import logging
import re
import time
import traceback
from pathlib import Path
from typing import Optional, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import Config

logger = logging.getLogger(__name__)

# store_id must consist only of alphanumeric characters, hyphens, and underscores.
# This prevents path traversal attacks when the value is used as a directory name.
_STORE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")

# Type alias — a list of (Document, relevance_score) pairs.
# Relevance scores from Chroma are in [0, 1], where 1.0 is most similar.
Chunk = tuple[Document, float]


# ── Public API ────────────────────────────────────────────────────────────────


def index(
    documents: list[Document],
    store_id: str,
    embedding_model: Optional[str] = None,
) -> Chroma:
    """Embed *documents* and write them into a persistent Chroma collection.

    Args:
        documents:       Chunked LangChain Document objects (from ingestion).
        store_id:        Logical name for the collection; used as sub-directory.
        embedding_model: Override the default embedding model from Config.

    Returns:
        The loaded ``Chroma`` vector store instance (ready for retrieval).
    """
    t0 = time.monotonic()
    persist_dir = _store_path(store_id)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = _make_embeddings(embedding_model)

    try:
        store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(persist_dir),
            collection_name=store_id,
        )
    except Exception as exc:
        _log_error("indexing", str(exc), traceback.format_exc())
        raise

    duration_ms = int((time.monotonic() - t0) * 1000)
    _log_indexing(store_id, len(documents), duration_ms)
    return store


def load_store(
    store_id: str,
    embedding_model: Optional[str] = None,
) -> Chroma:
    """Load an existing persistent Chroma store from disk.

    Args:
        store_id:        The store name used when it was originally indexed.
        embedding_model: Override the default embedding model from Config.

    Returns:
        The loaded ``Chroma`` instance.

    Raises:
        FileNotFoundError: If the store directory does not exist.
    """
    persist_dir = _store_path(store_id)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No Chroma store found for store_id='{store_id}' at {persist_dir}. "
            "Run indexing first."
        )
    embeddings = _make_embeddings(embedding_model)
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=store_id,
    )


def retrieve(
    query: str,
    store: Chroma,
    k: int = Config.RETRIEVAL_K,
) -> list[Chunk]:
    """Retrieve the *k* most relevant chunks from *store* for *query*.

    Args:
        query: Natural-language question.
        store: A loaded ``Chroma`` instance.
        k:     Number of chunks to return.

    Returns:
        List of ``(Document, relevance_score)`` tuples, ordered by descending score.
    """
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:12]

    try:
        results: list[Chunk] = store.similarity_search_with_relevance_scores(
            query, k=k
        )
    except Exception as exc:
        _log_error("retrieval", str(exc), traceback.format_exc())
        raise

    top_score = results[0][1] if results else 0.0
    _log_retrieval(query_hash, len(results), top_score)
    return results


def store_exists(store_id: str) -> bool:
    """Return True if a persisted Chroma store exists for *store_id*."""
    return _store_path(store_id).exists()


# ── Private helpers ───────────────────────────────────────────────────────────


def _store_path(store_id: str) -> Path:
    """Return the absolute path for a store, validating against path traversal."""
    if not _STORE_ID_RE.match(store_id):
        raise ValueError(
            f"Invalid store_id '{store_id}'. "
            "Use only letters, digits, hyphens, and underscores (max 128 chars)."
        )
    base = Config.CHROMA_BASE_DIR.resolve()
    candidate = (base / store_id).resolve()
    # Guard: ensure the resolved path is strictly within CHROMA_BASE_DIR
    if not str(candidate).startswith(str(base) + "/") and candidate != base:
        raise ValueError(
            f"store_id '{store_id}' resolves outside the Chroma base directory."
        )
    return candidate


def _make_embeddings(model: Optional[str] = None) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=model or Config.OPENAI_EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY,
    )


def _log_indexing(store_id: str, vectors_written: int, duration_ms: int) -> None:
    record = {
        "status": "ok",
        "store_id": store_id,
        "vectors_written": vectors_written,
        "duration_ms": duration_ms,
    }
    logger.info(json.dumps(record))


def _log_retrieval(query_hash: str, chunks_returned: int, top_score: float) -> None:
    record = {
        "status": "ok",
        "query_hash": query_hash,
        "chunks_returned": chunks_returned,
        "top_score": round(top_score, 4),
    }
    logger.info(json.dumps(record))


def _log_error(stage: str, message: str, tb: str) -> None:
    record = {"status": "error", "stage": stage, "message": message, "traceback": tb}
    logger.error(json.dumps(record))
