"""generation.chain — Ollama LLM RAG chain for ollama_rag.

Runs entirely locally: Ollama LLM + Ollama Embeddings + Chroma.
No external API keys required at runtime.

Answer schema:
{
  "answer":           str,
  "sources":          [{"document": str, "page_or_chunk": int, "score": float, "excerpt": str}],
  "confidence":       "high" | "medium" | "low",
  "retrieved_chunks": int
}
"""

import json
import logging
import time
import traceback
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import Config

logger = logging.getLogger(__name__)

_EXCERPT_LENGTH = 220

_SYSTEM_PROMPT = (
    "You are a precise research assistant for the RoboMarket domain. "
    "Answer questions based ONLY on the provided context. "
    "If the context does not contain enough information, say so explicitly. "
    "Do not fabricate information. Be concise and factual."
)

_USER_TEMPLATE = """\
Context (retrieved from indexed documents):
{context}

Question: {question}

Answer using only the information in the context above."""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("human", _USER_TEMPLATE)]
)


class SourceRef(TypedDict):
    document: str
    page_or_chunk: int
    score: float
    excerpt: str


class Answer(TypedDict):
    answer: str
    sources: list[SourceRef]
    confidence: str
    retrieved_chunks: int


# ── Public API ────────────────────────────────────────────────────────────────


def generate(
    query: str,
    context: list[tuple[Document, float]],
    model: str | None = None,
) -> Answer:
    """Generate a grounded answer from retrieved context using Ollama LLM.

    Args:
        query:   The user question.
        context: List of (Document, relevance_score) from retrieval.
        model:   Override the Ollama model from Config.
    """
    from langchain_ollama import ChatOllama  # type: ignore

    t0 = time.monotonic()

    llm = ChatOllama(
        model=model or Config.OLLAMA_LLM_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
        temperature=0,
    )
    chain = _PROMPT | llm | StrOutputParser()
    context_text = _format_context(context)

    try:
        answer_text: str = chain.invoke({"context": context_text, "question": query})
    except Exception as exc:
        _log_error("generation", str(exc), traceback.format_exc())
        raise

    sources = _build_sources(context)
    confidence = _confidence(context)

    duration_ms = int((time.monotonic() - t0) * 1000)
    logger.info(json.dumps({"status": "ok", "confidence": confidence, "ms": duration_ms}))

    return Answer(
        answer=answer_text.strip(),
        sources=sources,
        confidence=confidence,
        retrieved_chunks=len(context),
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _format_context(context: list[tuple[Document, float]]) -> str:
    parts: list[str] = []
    for i, (doc, score) in enumerate(context, start=1):
        meta = doc.metadata
        source = meta.get("source", "unknown")
        page = meta.get("page", -1)
        chunk_idx = meta.get("chunk_index", i - 1)
        loc = f"page {page}" if page >= 0 else f"chunk {chunk_idx}"
        parts.append(
            f"[{i}] Source: {source} ({loc}) | Relevance: {score:.2f}\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def _build_sources(context: list[tuple[Document, float]]) -> list[SourceRef]:
    sources: list[SourceRef] = []
    for doc, score in context:
        meta = doc.metadata
        page = meta.get("page", -1)
        chunk_idx = meta.get("chunk_index", 0)
        page_or_chunk = page if page >= 0 else chunk_idx
        excerpt = doc.page_content[:_EXCERPT_LENGTH].replace("\n", " ").strip()
        if len(doc.page_content) > _EXCERPT_LENGTH:
            excerpt += "…"
        sources.append(
            SourceRef(
                document=meta.get("source", "unknown"),
                page_or_chunk=page_or_chunk,
                score=round(score, 4),
                excerpt=excerpt,
            )
        )
    return sources


def _confidence(context: list[tuple[Document, float]]) -> str:
    if not context:
        return "low"
    top_score = context[0][1]
    if top_score >= Config.CONFIDENCE_HIGH:
        return "high"
    if top_score >= Config.CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def _log_error(stage: str, message: str, tb: str) -> None:
    logger.error(json.dumps({"status": "error", "stage": stage, "message": message, "traceback": tb}))
