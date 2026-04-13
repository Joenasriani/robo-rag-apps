"""tests/test_ingestion.py — validate ingestion pipeline for tool_use_rag."""

import sys
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from ingestion.loader import ingest  # noqa: E402


@pytest.fixture()
def sample_txt(tmp_path: Path) -> Path:
    content = (
        "RoboMarket is a B2B marketplace for industrial robots.\n"
        "It supports FANUC, KUKA, ABB, and Universal Robots.\n"
        "Suppliers must pass a three-stage verification process.\n"
        "The platform charges a 1.5 to 3.5 percent transaction fee.\n"
        "Integration APIs include REST and GraphQL endpoints.\n" * 10
    )
    p = tmp_path / "test_doc.txt"
    p.write_text(content, encoding="utf-8")
    return p


def test_txt_ingest_produces_chunks(sample_txt: Path) -> None:
    chunks = ingest(sample_txt)
    assert len(chunks) > 0


def test_txt_chunks_have_source_metadata(sample_txt: Path) -> None:
    chunks = ingest(sample_txt)
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == sample_txt.name


def test_txt_chunks_have_chunk_index_metadata(sample_txt: Path) -> None:
    chunks = ingest(sample_txt)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_txt_chunks_have_page_metadata(sample_txt: Path) -> None:
    chunks = ingest(sample_txt)
    for chunk in chunks:
        assert chunk.metadata.get("page") == -1


def test_chunk_content_is_non_empty(sample_txt: Path) -> None:
    chunks = ingest(sample_txt)
    for chunk in chunks:
        assert chunk.page_content.strip()


def test_unsupported_type_raises_value_error(tmp_path: Path) -> None:
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("col1,col2\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        ingest(bad_file)


def test_missing_file_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        ingest(Path("/nonexistent/path/doc.txt"))


def test_sample_data_file_ingests() -> None:
    sample = _APP_DIR / "sample_data" / "robomarket_faq.txt"
    if not sample.exists():
        pytest.skip("sample_data/robomarket_faq.txt not found")
    chunks = ingest(sample)
    assert len(chunks) > 0
    texts = [c.page_content for c in chunks]
    assert any("RoboMarket" in t for t in texts)
