"""tests/test_retrieval.py — validate indexing and retrieval pipeline.

These tests build a temporary persistent Chroma store using FakeEmbeddings
so they do NOT require an OpenAI API key.  The FakeEmbeddings class generates
deterministic random vectors; the tests verify structural correctness (non-empty
results, metadata round-trip) rather than semantic ranking quality.
"""

import sys
from pathlib import Path

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from langchain_core.documents import Document

# ── Test documents ────────────────────────────────────────────────────────────

_TEST_DOCS = [
    Document(
        page_content=(
            "RoboMarket is a B2B marketplace that connects buyers and sellers "
            "of industrial robots, cobots, and automation components."
        ),
        metadata={"source": "overview.txt", "page": -1, "chunk_index": 0},
    ),
    Document(
        page_content=(
            "The platform supports FANUC, KUKA, ABB, Yaskawa, and Universal Robots. "
            "Over 50,000 SKUs are listed across six product categories."
        ),
        metadata={"source": "overview.txt", "page": -1, "chunk_index": 1},
    ),
    Document(
        page_content=(
            "Suppliers must pass a three-stage verification: legal entity check, "
            "product authenticity audit, and reference customer validation."
        ),
        metadata={"source": "overview.txt", "page": -1, "chunk_index": 2},
    ),
    Document(
        page_content=(
            "RoboMarket charges a 1.5 to 3.5 percent transaction fee on completed "
            "orders. Buyers with annual spend above $1M qualify for negotiated rates."
        ),
        metadata={"source": "pricing.txt", "page": -1, "chunk_index": 0},
    ),
]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fake_embeddings():
    """FakeEmbeddings that generate deterministic vectors — no API key needed."""
    try:
        from langchain_community.embeddings.fake import FakeEmbeddings
    except ImportError:
        from langchain_core.embeddings.fake import FakeEmbeddings  # type: ignore

    return FakeEmbeddings(size=256)


@pytest.fixture(scope="module")
def chroma_store(tmp_path_factory, fake_embeddings):
    """Build a temporary Chroma store from test documents."""
    from langchain_chroma import Chroma

    persist_dir = tmp_path_factory.mktemp("chroma_test") / "store"
    persist_dir.mkdir(parents=True, exist_ok=True)

    store = Chroma.from_documents(
        documents=_TEST_DOCS,
        embedding=fake_embeddings,
        persist_directory=str(persist_dir),
        collection_name="test_store",
    )
    return store, persist_dir


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_index_creates_store(chroma_store) -> None:
    """Chroma store fixture must be non-None after indexing."""
    store, _ = chroma_store
    assert store is not None


def test_store_persists_to_disk(chroma_store) -> None:
    """The persist directory must exist after indexing."""
    _, persist_dir = chroma_store
    assert persist_dir.exists(), f"Chroma persist dir not found: {persist_dir}"


def test_store_can_be_reloaded(chroma_store, fake_embeddings) -> None:
    """A store saved to disk must be loadable in a fresh Chroma instance."""
    from langchain_chroma import Chroma

    _, persist_dir = chroma_store
    reloaded = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=fake_embeddings,
        collection_name="test_store",
    )
    assert reloaded is not None


def test_retrieve_returns_results(chroma_store) -> None:
    """similarity_search_with_relevance_scores must return non-empty results."""
    store, _ = chroma_store
    results = store.similarity_search_with_relevance_scores(
        "What brands are on RoboMarket?", k=2
    )
    assert len(results) > 0, "Expected at least one result from the store"


def test_retrieve_returns_correct_k(chroma_store) -> None:
    """Requesting k=3 must return at most 3 results (may be fewer if store is small)."""
    store, _ = chroma_store
    k = 3
    results = store.similarity_search_with_relevance_scores("robots", k=k)
    assert len(results) <= k


def test_result_contains_document_and_score(chroma_store) -> None:
    """Each result must be a (Document, float) tuple with non-empty page_content."""
    store, _ = chroma_store
    results = store.similarity_search_with_relevance_scores("marketplace", k=2)
    for doc, score in results:
        assert isinstance(doc, Document), f"Expected Document, got {type(doc)}"
        assert isinstance(score, float), f"Expected float score, got {type(score)}"
        assert doc.page_content.strip(), "Chunk has empty page_content"


def test_chunk_metadata_preserved(chroma_store) -> None:
    """Metadata stored during indexing must be retrievable from the store."""
    store, _ = chroma_store
    results = store.similarity_search_with_relevance_scores("transaction fee pricing", k=4)
    assert len(results) > 0
    sources = [doc.metadata.get("source") for doc, _ in results]
    assert any(s in ("overview.txt", "pricing.txt") for s in sources), (
        f"Expected source metadata to be one of overview.txt/pricing.txt, got {sources}"
    )


def test_retrieve_with_retriever_module(chroma_store) -> None:
    """The retrieval.retriever.retrieve() wrapper must work with the store."""
    from retrieval.retriever import retrieve  # noqa: E402

    store, _ = chroma_store
    results = retrieve("FANUC KUKA ABB robot brands", store, k=2)
    assert len(results) > 0, "retrieve() returned no results"
    for doc, score in results:
        assert doc.page_content.strip()
        assert isinstance(score, float)


def test_top_result_from_module_has_metadata(chroma_store) -> None:
    """The top result from retrieve() must include source metadata."""
    from retrieval.retriever import retrieve  # noqa: E402

    store, _ = chroma_store
    results = retrieve("supplier verification", store, k=3)
    assert results, "retrieve() returned empty list"
    top_doc, _ = results[0]
    assert "source" in top_doc.metadata, "Top result missing 'source' metadata"
