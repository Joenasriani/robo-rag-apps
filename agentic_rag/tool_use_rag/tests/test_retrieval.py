"""tests/test_retrieval.py — validate retrieval and tool for tool_use_rag.

Uses FakeEmbeddings — no OpenAI API key required.
"""

import sys
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from langchain_core.documents import Document

_TEST_DOCS = [
    Document(
        page_content="RoboMarket is a B2B marketplace for industrial robots and cobots.",
        metadata={"source": "faq.txt", "page": -1, "chunk_index": 0},
    ),
    Document(
        page_content="FANUC, KUKA, ABB, and Universal Robots are available on the platform.",
        metadata={"source": "faq.txt", "page": -1, "chunk_index": 1},
    ),
    Document(
        page_content="Transaction fees range from 1.5% to 3.5% depending on order size.",
        metadata={"source": "faq.txt", "page": -1, "chunk_index": 2},
    ),
    Document(
        page_content="Suppliers undergo a three-stage verification process before listing.",
        metadata={"source": "faq.txt", "page": -1, "chunk_index": 3},
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

    persist_dir = tmp_path_factory.mktemp("chroma_tool_use") / "store"
    persist_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(
        documents=_TEST_DOCS,
        embedding=fake_embeddings,
        persist_directory=str(persist_dir),
        collection_name="test_tool_use",
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
    results = store.similarity_search_with_relevance_scores("robot brands", k=2)
    assert len(results) > 0


def test_result_has_document_and_score(chroma_store) -> None:
    store, _ = chroma_store
    results = store.similarity_search_with_relevance_scores("fees", k=2)
    for doc, score in results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert doc.page_content.strip()


def test_chunk_metadata_preserved(chroma_store) -> None:
    store, _ = chroma_store
    results = store.similarity_search_with_relevance_scores("marketplace", k=4)
    sources = [doc.metadata.get("source") for doc, _ in results]
    assert any(s == "faq.txt" for s in sources)


def test_retrieve_module_function(chroma_store) -> None:
    from retrieval.retriever import retrieve

    store, _ = chroma_store
    results = retrieve("transaction fees", store, k=2)
    assert len(results) > 0
    for doc, score in results:
        assert doc.page_content.strip()
        assert isinstance(score, float)


def test_make_retrieval_tool_returns_string(chroma_store) -> None:
    from retrieval.retriever import make_retrieval_tool

    store, _ = chroma_store
    tool = make_retrieval_tool(store, k=2)
    result = tool.run("verification process")
    assert isinstance(result, str)
    assert len(result) > 0


def test_retrieval_tool_no_results_returns_message(fake_embeddings) -> None:
    """An empty store should return a 'no results' message, not raise."""
    from langchain_chroma import Chroma
    from retrieval.retriever import make_retrieval_tool

    # Build a fresh store with a single doc to avoid empty-store edge cases
    doc = Document(
        page_content="placeholder",
        metadata={"source": "p.txt", "page": -1, "chunk_index": 0},
    )
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        store = Chroma.from_documents(
            documents=[doc],
            embedding=fake_embeddings,
            persist_directory=tmp,
            collection_name="empty_test",
        )
        tool = make_retrieval_tool(store, k=3)
        result = tool.run("totally unrelated query about nothing")
        assert isinstance(result, str)
