"""tests/test_retrieval.py — validate retrieval and FastAPI endpoints for api_service.

Uses FakeEmbeddings — no OpenAI API key required.
FastAPI endpoint tests use httpx.AsyncClient with mocked embeddings and generation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from langchain_core.documents import Document

_TEST_DOCS = [
    Document(
        page_content="The RoboMarket API uses OAuth 2.0 for authentication.",
        metadata={"source": "api_docs.txt", "page": -1, "chunk_index": 0},
    ),
    Document(
        page_content="Rate limits: 1,000 req/min for standard, 10,000 for enterprise.",
        metadata={"source": "api_docs.txt", "page": -1, "chunk_index": 1},
    ),
    Document(
        page_content="GET /products lists all products with optional filters.",
        metadata={"source": "api_docs.txt", "page": -1, "chunk_index": 2},
    ),
    Document(
        page_content="POST /orders creates a new order with items and shipping address.",
        metadata={"source": "api_docs.txt", "page": -1, "chunk_index": 3},
    ),
]


@pytest.fixture(scope="module")
def fake_embeddings():
    try:
        from langchain_community.embeddings.fake import FakeEmbeddings
    except ImportError:
        from langchain_core.embeddings.fake import FakeEmbeddings  # type: ignore
    return FakeEmbeddings(size=256)


@pytest.fixture(scope="module")
def chroma_store(tmp_path_factory, fake_embeddings):
    from langchain_chroma import Chroma

    persist_dir = tmp_path_factory.mktemp("chroma_api") / "store"
    persist_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(
        documents=_TEST_DOCS,
        embedding=fake_embeddings,
        persist_directory=str(persist_dir),
        collection_name="test_api_service",
    )
    return store, persist_dir


def test_index_creates_store(chroma_store) -> None:
    store, _ = chroma_store
    assert store is not None


def test_store_persists_to_disk(chroma_store) -> None:
    _, persist_dir = chroma_store
    assert persist_dir.exists()


def test_retrieve_returns_results(chroma_store) -> None:
    store, _ = chroma_store
    results = store.similarity_search_with_relevance_scores("authentication", k=2)
    assert len(results) > 0


def test_result_has_document_and_score(chroma_store) -> None:
    store, _ = chroma_store
    results = store.similarity_search_with_relevance_scores("rate limits", k=2)
    for doc, score in results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert doc.page_content.strip()


def test_chunk_metadata_preserved(chroma_store) -> None:
    store, _ = chroma_store
    results = store.similarity_search_with_relevance_scores("products API", k=4)
    sources = [doc.metadata.get("source") for doc, _ in results]
    assert any(s == "api_docs.txt" for s in sources)


def test_retrieve_module_function(chroma_store) -> None:
    from retrieval.retriever import retrieve

    store, _ = chroma_store
    results = retrieve("OAuth authentication", store, k=2)
    assert len(results) > 0
    for doc, score in results:
        assert doc.page_content.strip()
        assert isinstance(score, float)


def test_index_function_with_fake_embeddings(tmp_path, fake_embeddings) -> None:
    from retrieval.retriever import index

    docs = [
        Document(
            page_content="RoboMarket provides REST and GraphQL APIs.",
            metadata={"source": "test.txt", "page": -1, "chunk_index": 0},
        )
    ]
    import config as cfg
    original = cfg.Config.CHROMA_BASE_DIR
    cfg.Config.CHROMA_BASE_DIR = tmp_path / ".chroma"
    try:
        store = index(docs, "test-api-fake", embeddings=fake_embeddings)
        assert store is not None
        results = store.similarity_search_with_relevance_scores("REST API", k=1)
        assert len(results) > 0
    finally:
        cfg.Config.CHROMA_BASE_DIR = original


# ── FastAPI endpoint tests ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def api_app():
    """Import the FastAPI app after path is set up."""
    import app as api_module
    return api_module.app


@pytest.mark.asyncio
async def test_health_endpoint(api_app) -> None:
    """GET /health must return 200 and status ok."""
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=api_app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_query_unknown_store_returns_404(api_app) -> None:
    """POST /query with a non-existent store_id must return 404."""
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=api_app), base_url="http://test") as client:
        resp = await client.post(
            "/query",
            json={"store_id": "nonexistent-store-xyz", "query": "test query"},
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_index_without_api_key_returns_503(api_app, monkeypatch) -> None:
    """POST /index without OPENAI_API_KEY must return 503."""
    import config as cfg
    monkeypatch.setattr(cfg.Config, "OPENAI_API_KEY", "")
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=api_app), base_url="http://test") as client:
        resp = await client.post(
            "/index",
            json={"store_id": "test-store", "documents": ["Some document text."]},
        )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_index_mismatched_source_names_returns_422(api_app, monkeypatch) -> None:
    """source_names length mismatch must return 422."""
    import config as cfg
    monkeypatch.setattr(cfg.Config, "OPENAI_API_KEY", "sk-test")
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=api_app), base_url="http://test") as client:
        resp = await client.post(
            "/index",
            json={
                "store_id": "test-store",
                "documents": ["Doc 1", "Doc 2"],
                "source_names": ["only_one.txt"],
            },
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_index_and_query_end_to_end(api_app, tmp_path, fake_embeddings, monkeypatch) -> None:
    """Full end-to-end: index a document, then query it."""
    import config as cfg
    monkeypatch.setattr(cfg.Config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(cfg.Config, "CHROMA_BASE_DIR", tmp_path / ".chroma")

    from httpx import ASGITransport, AsyncClient

    # Mock the OpenAI embeddings and LLM to avoid real API calls
    with patch("retrieval.retriever._make_embeddings", return_value=fake_embeddings), \
         patch("generation.chain.ChatOpenAI") as mock_llm_cls:

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_chain_output = "The API uses OAuth 2.0 for authentication."
        # Simulate chain.invoke returning a string
        mock_llm.__or__ = MagicMock(return_value=mock_llm)
        mock_llm.invoke = MagicMock(return_value=mock_chain_output)

        # Patch the full chain pipeline
        with patch("app.generate") as mock_generate:
            mock_generate.return_value = {
                "answer": "The API uses OAuth 2.0 for authentication.",
                "sources": [
                    {
                        "document": "inline_doc.txt",
                        "page_or_chunk": 0,
                        "score": 0.85,
                        "excerpt": "OAuth 2.0 authentication.",
                    }
                ],
                "confidence": "high",
                "retrieved_chunks": 1,
            }

            async with AsyncClient(transport=ASGITransport(app=api_app), base_url="http://test") as client:
                # Index
                idx_resp = await client.post(
                    "/index",
                    json={
                        "store_id": "e2e-test-store",
                        "documents": [
                            "OAuth 2.0 is used for authentication. "
                            "Rate limits are 1000 per minute. " * 5
                        ],
                        "source_names": ["inline_doc.txt"],
                    },
                )
                assert idx_resp.status_code == 201, idx_resp.text
                assert idx_resp.json()["chunks_indexed"] > 0

                # Query
                q_resp = await client.post(
                    "/query",
                    json={"store_id": "e2e-test-store", "query": "How does authentication work?"},
                )
                assert q_resp.status_code == 200, q_resp.text
                body = q_resp.json()
                assert body["answer"]
                assert body["confidence"] in ("high", "medium", "low")
