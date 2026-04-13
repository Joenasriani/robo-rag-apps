"""retrieval.retriever — Chroma-backed indexing and retrieval for api_service."""

import json
import logging
import re
import time
import traceback
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from config import Config

logger = logging.getLogger(__name__)

_STORE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")

Chunk = tuple[Document, float]


def index(
    documents: list[Document],
    store_id: str,
    embeddings: Optional[Embeddings] = None,
) -> Chroma:
    """Embed documents and write into a persistent Chroma collection."""
    t0 = time.monotonic()
    persist_dir = _store_path(store_id)
    persist_dir.mkdir(parents=True, exist_ok=True)

    if embeddings is None:
        embeddings = _make_embeddings()

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
    logger.info(json.dumps({"status": "ok", "store_id": store_id, "vectors": len(documents), "ms": duration_ms}))
    return store


def load_store(store_id: str, embeddings: Optional[Embeddings] = None) -> Chroma:
    """Load an existing persistent Chroma store from disk."""
    persist_dir = _store_path(store_id)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No Chroma store found for store_id='{store_id}'."
        )
    if embeddings is None:
        embeddings = _make_embeddings()
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
    """Retrieve the k most relevant chunks."""
    try:
        results: list[Chunk] = store.similarity_search_with_relevance_scores(query, k=k)
    except Exception as exc:
        _log_error("retrieval", str(exc), traceback.format_exc())
        raise
    top_score = results[0][1] if results else 0.0
    logger.info(json.dumps({"status": "ok", "chunks": len(results), "top_score": round(top_score, 4)}))
    return results


def store_exists(store_id: str) -> bool:
    return _store_path(store_id).exists()


def _store_path(store_id: str) -> Path:
    if not _STORE_ID_RE.match(store_id):
        raise ValueError(f"Invalid store_id '{store_id}'.")
    base = Config.CHROMA_BASE_DIR.resolve()
    candidate = (base / store_id).resolve()
    if not str(candidate).startswith(str(base) + "/") and candidate != base:
        raise ValueError(f"store_id '{store_id}' resolves outside Chroma base dir.")
    return candidate


def _make_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=Config.OPENAI_EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY,
    )


def _log_error(stage: str, message: str, tb: str) -> None:
    logger.error(json.dumps({"status": "error", "stage": stage, "message": message, "traceback": tb}))
