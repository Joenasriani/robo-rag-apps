"""ingestion.loader — document loading and chunking for api_service."""

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Union

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def ingest(
    source: Union[str, Path],
    chunk_size: int = Config.CHUNK_SIZE,
    chunk_overlap: int = Config.CHUNK_OVERLAP,
) -> list[Document]:
    """Load and chunk a document file.

    Returns chunks with metadata: source, page, chunk_index.
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
    logger.info(json.dumps({"status": "ok", "file": source.name, "chunks": len(chunks), "ms": duration_ms}))
    return chunks


def ingest_text(
    text: str,
    source_name: str,
    chunk_size: int = Config.CHUNK_SIZE,
    chunk_overlap: int = Config.CHUNK_OVERLAP,
) -> list[Document]:
    """Chunk a raw text string into Documents.

    Args:
        text:        Raw text content.
        source_name: Logical filename to store in metadata.

    Returns:
        List of Document objects with source, page=-1, chunk_index metadata.
    """
    doc = Document(page_content=text, metadata={"source": source_name, "page": -1})
    return _chunk([doc], source_name, chunk_size, chunk_overlap)


def _load(source: Path, suffix: str) -> list[Document]:
    if suffix == ".pdf":
        return PyPDFLoader(str(source)).load()
    docs = TextLoader(str(source), encoding="utf-8", autodetect_encoding=True).load()
    for doc in docs:
        doc.metadata.setdefault("page", -1)
    return docs


def _chunk(
    docs: list[Document],
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    split_docs = splitter.split_documents(docs)
    for idx, doc in enumerate(split_docs):
        doc.metadata["source"] = filename
        doc.metadata.setdefault("page", -1)
        doc.metadata["chunk_index"] = idx
    return split_docs


def _log_error(stage: str, message: str, tb: str) -> None:
    logger.error(json.dumps({"status": "error", "stage": stage, "message": message, "traceback": tb}))
