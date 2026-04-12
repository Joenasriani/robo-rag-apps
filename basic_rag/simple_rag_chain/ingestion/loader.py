"""ingestion.loader — document loading and chunking for simple_rag_chain.

Supported file types:
  .pdf  — PyPDFLoader (preserves page numbers)
  .txt  — TextLoader
  .md   — TextLoader

Chunking strategy:
  RecursiveCharacterTextSplitter with configurable chunk_size and overlap.
  Every chunk receives metadata:
    - source:      original filename (basename)
    - page:        page number for PDFs (0-indexed from PyPDFLoader); -1 for text/md
    - chunk_index: position of the chunk within its source document

Structured JSON log emitted for each ingest call.
"""

import hashlib
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Union

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from config import Config

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def ingest(
    source: Union[str, Path],
    chunk_size: int = Config.CHUNK_SIZE,
    chunk_overlap: int = Config.CHUNK_OVERLAP,
) -> list[Document]:
    """Load and chunk a single document file.

    Args:
        source:        Path to the file to ingest.
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of characters overlapping between adjacent chunks.

    Returns:
        List of LangChain ``Document`` objects, each with metadata:
        ``source``, ``page``, ``chunk_index``.

    Raises:
        ValueError: Unsupported file type.
        FileNotFoundError: File does not exist.
        RuntimeError: Loader or splitter failure.
    """
    source = Path(source)
    suffix = source.suffix.lower()
    t0 = time.monotonic()

    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {sorted(_SUPPORTED_SUFFIXES)}"
        )

    try:
        raw_docs = _load(source, suffix)
    except Exception as exc:
        _log_error("ingestion", str(exc), traceback.format_exc())
        raise RuntimeError(f"Failed to load '{source.name}': {exc}") from exc

    try:
        chunks = _chunk(raw_docs, source.name, chunk_size, chunk_overlap)
    except Exception as exc:
        _log_error("ingestion", str(exc), traceback.format_exc())
        raise RuntimeError(f"Failed to chunk '{source.name}': {exc}") from exc

    duration_ms = int((time.monotonic() - t0) * 1000)
    _log_ingestion(source.name, len(chunks), duration_ms)
    return chunks


# ── Private helpers ───────────────────────────────────────────────────────────


def _load(source: Path, suffix: str) -> list[Document]:
    """Return raw (un-split) documents from the given file."""
    if suffix == ".pdf":
        loader = PyPDFLoader(str(source))
        return loader.load()
    # .txt and .md both use TextLoader
    loader = TextLoader(str(source), encoding="utf-8", autodetect_encoding=True)
    docs = loader.load()
    # Assign page=-1 for non-PDF sources so downstream code can check safely
    for doc in docs:
        doc.metadata.setdefault("page", -1)
    return docs


def _chunk(
    docs: list[Document],
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split raw documents into chunks and enrich metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    split_docs = splitter.split_documents(docs)

    for idx, doc in enumerate(split_docs):
        # Normalise source to basename so paths don't leak into the store
        doc.metadata["source"] = filename
        # PyPDFLoader stores page as int; for text files we defaulted to -1 above
        doc.metadata.setdefault("page", -1)
        doc.metadata["chunk_index"] = idx

    return split_docs


def _log_ingestion(filename: str, chunks_produced: int, duration_ms: int) -> None:
    record = {
        "status": "ok",
        "file": filename,
        "chunks_produced": chunks_produced,
        "duration_ms": duration_ms,
    }
    logger.info(json.dumps(record))


def _log_error(stage: str, message: str, tb: str) -> None:
    record = {"status": "error", "stage": stage, "message": message, "traceback": tb}
    logger.error(json.dumps(record))
