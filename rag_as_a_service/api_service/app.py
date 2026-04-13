"""api_service — FastAPI RAG service.

Endpoints:
  POST /index   — Ingest and index document text into a named store.
  POST /query   — Query a named store and return a grounded answer.
  GET  /health  — Health check.

Run:
    python app.py
    # or
    uvicorn app:app --host 0.0.0.0 --port 8000

Requires:
    OPENAI_API_KEY set in .env (see .env.example)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Path setup ────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from config import Config
from generation.chain import generate
from ingestion.loader import ingest_text
from retrieval.retriever import index as index_docs
from retrieval.retriever import load_store, retrieve, store_exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RoboMarket RAG Service",
    description=(
        "A production-grade Retrieval-Augmented Generation API for the RoboMarket domain. "
        "Index documents via POST /index, then query them via POST /query."
    ),
    version="1.0.0",
)


# ── Request / Response schemas ────────────────────────────────────────────────


class IndexRequest(BaseModel):
    """Request body for POST /index."""

    store_id: str = Field(
        ...,
        description="Logical name for the Chroma collection (alphanumeric, hyphens, underscores).",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_\-]+$",
    )
    documents: list[str] = Field(
        ...,
        description="List of plain-text document contents to ingest.",
        min_length=1,
    )
    source_names: Optional[list[str]] = Field(
        default=None,
        description="Optional list of source filenames, one per document. "
        "Defaults to 'document_0.txt', 'document_1.txt', …",
    )


class IndexResponse(BaseModel):
    store_id: str
    chunks_indexed: int
    message: str


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    store_id: str = Field(
        ...,
        description="The store to query (must have been indexed first).",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_\-]+$",
    )
    query: str = Field(
        ...,
        description="Natural-language question to answer from the indexed documents.",
        min_length=1,
    )
    k: int = Field(
        default=Config.RETRIEVAL_K,
        description="Number of chunks to retrieve.",
        ge=1,
        le=20,
    )


class SourceRef(BaseModel):
    document: str
    page_or_chunk: int
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    store_id: str
    query: str
    answer: str
    sources: list[SourceRef]
    confidence: str
    retrieved_chunks: int


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok", "service": "robomarket-rag-api"}


@app.post(
    "/index",
    response_model=IndexResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest and index documents",
)
async def index_endpoint(req: IndexRequest) -> IndexResponse:
    """Chunk and embed the provided document texts into a named Chroma store.

    - Each string in ``documents`` is treated as a separate document.
    - The store persists on disk and can be queried after this call.
    - Calling this endpoint again with the same ``store_id`` **adds** new documents
      to the existing collection (does not replace it).
    """
    if not Config.OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY is not configured on the server.",
        )

    source_names = req.source_names or [
        f"document_{i}.txt" for i in range(len(req.documents))
    ]
    if len(source_names) != len(req.documents):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="source_names length must match documents length.",
        )

    all_chunks = []
    for text, name in zip(req.documents, source_names):
        try:
            chunks = ingest_text(text, name)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to chunk document '{name}': {exc}",
            ) from exc
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No chunks produced from the provided documents.",
        )

    try:
        index_docs(all_chunks, req.store_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Indexing failed for store_id=%s", req.store_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {exc}",
        ) from exc

    return IndexResponse(
        store_id=req.store_id,
        chunks_indexed=len(all_chunks),
        message=f"Successfully indexed {len(all_chunks)} chunks into store '{req.store_id}'.",
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query a store and get a grounded answer",
)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Retrieve relevant chunks from the named store and generate a grounded answer.

    Returns:
        - ``answer``: LLM-generated answer grounded in retrieved chunks.
        - ``sources``: Citations tracing back to the retrieved chunks.
        - ``confidence``: Derived from the top relevance score (high/medium/low).
    """
    if not store_exists(req.store_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Store '{req.store_id}' not found. Index documents first via POST /index.",
        )

    if not Config.OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY is not configured on the server.",
        )

    try:
        store = load_store(req.store_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load store: {exc}",
        ) from exc

    try:
        chunks = retrieve(req.query, store, k=req.k)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    if not chunks:
        return QueryResponse(
            store_id=req.store_id,
            query=req.query,
            answer="No relevant documents found in the knowledge base for this query.",
            sources=[],
            confidence="low",
            retrieved_chunks=0,
        )

    try:
        answer = generate(req.query, chunks)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {exc}",
        ) from exc

    return QueryResponse(
        store_id=req.store_id,
        query=req.query,
        answer=answer["answer"],
        sources=[SourceRef(**s) for s in answer["sources"]],
        confidence=answer["confidence"],
        retrieved_chunks=answer["retrieved_chunks"],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=False,
    )
