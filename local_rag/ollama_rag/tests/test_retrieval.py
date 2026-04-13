"""tests/test_retrieval.py — validate retrieval for ollama_rag.

Uses FakeEmbeddings via the embeddings parameter — no Ollama required.
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
        page_content="Articulated robots have 4–6 rotary joints and are used for welding and assembly.",
        metadata={"source": "kb.txt", "page": -1, "chunk_index": 0},
    ),
    Document(
        page_content="Cobots comply with ISO/TS 15066 and can share workspace with humans.",
        metadata={"source": "kb.txt", "page": -1, "chunk_index": 1},
    ),
    Document(
        page_content="OPC UA is the recommended protocol for Industry 4.0 integrations.",
        metadata={"source": "kb.txt", "page": -1, "chunk_index": 2},
    ),
    Document(
        page_content="FANUC, KUKA, ABB, Yaskawa, and Universal Robots are top brands on RoboMarket.",
        metadata={"source": "kb.txt", "page": -1, "chunk_index": 3},
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

    persist_dir = tmp_path_factory.mktemp("chroma_ollama") / "store"
    persist_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(
        documents=_TEST_DOCS,
        embedding=fake_embeddings,
        persist_directory=str(persist_dir),
        collection_name="test_ollama_rag",
    )
    return store, persist_dir


def test_index_creates_store(chroma_store) -> None:
    store, _ = chroma_store
    assert store is not None


def test_store_persists_to_disk(chroma_store) -> None:
    _, persist_dir = chroma_store
    assert persist_dir.exists()


def test_store_can_be_reloaded(chroma_store, fake_embeddings) -> None:
    from langchain_chroma import Chroma

    _, persist_dir = chroma_store
    reloaded = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=fake_embeddings,
        collection_name="test_ollama_rag",
    )
    assert reloaded is not None


def test_retrieve_returns_results(chroma_store) -> None:
    store, _ = chroma_store
    results = store.similarity_search_with_relevance_scores("robot brands", k=2)
    assert len(results) > 0


def test_result_has_document_and_score(chroma_store) -> None:
    store, _ = chroma_store
    results = store.similarity_search_with_relevance_scores("cobots safety", k=2)
    for doc, score in results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert doc.page_content.strip()


def test_chunk_metadata_preserved(chroma_store) -> None:
    store, _ = chroma_store
    results = store.similarity_search_with_relevance_scores("integration protocol", k=4)
    sources = [doc.metadata.get("source") for doc, _ in results]
    assert any(s == "kb.txt" for s in sources)


def test_retrieve_module_function(chroma_store) -> None:
    from retrieval.retriever import retrieve

    store, _ = chroma_store
    results = retrieve("industry 4.0 protocol", store, k=2)
    assert len(results) > 0
    for doc, score in results:
        assert doc.page_content.strip()
        assert isinstance(score, float)


def test_index_function_with_fake_embeddings(tmp_path, fake_embeddings) -> None:
    """retrieval.retriever.index() must accept an embeddings override."""
    from retrieval.retriever import index

    docs = [
        Document(
            page_content="Test document for index function.",
            metadata={"source": "test.txt", "page": -1, "chunk_index": 0},
        )
    ]
    store_id = "test-fake-index"
    # Override CHROMA_BASE_DIR via monkey-patching config
    import config as cfg
    original = cfg.Config.CHROMA_BASE_DIR
    cfg.Config.CHROMA_BASE_DIR = tmp_path / ".chroma"
    try:
        store = index(docs, store_id, embeddings=fake_embeddings)
        assert store is not None
        results = store.similarity_search_with_relevance_scores("test document", k=1)
        assert len(results) > 0
    finally:
        cfg.Config.CHROMA_BASE_DIR = original
