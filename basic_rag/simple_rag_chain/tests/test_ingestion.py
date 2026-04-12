"""tests/test_ingestion.py — validate real ingestion of .txt files.

These tests do NOT require an OpenAI API key; they only exercise
the document loading and chunking pipeline.
"""

import sys
from pathlib import Path

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow importing the app modules (config, ingestion.*) when running pytest
# from the app directory or from the repo root.
_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from ingestion.loader import ingest  # noqa: E402

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_txt(tmp_path: Path) -> Path:
    """Create a small .txt file with known content."""
    content = (
        "RoboMarket is a B2B marketplace for industrial robots.\n"
        "It supports FANUC, KUKA, ABB, and Universal Robots.\n"
        "Suppliers must pass a three-stage verification process.\n"
        "The platform charges a 1.5 to 3.5 percent transaction fee.\n"
        "Integration APIs include REST and GraphQL endpoints.\n" * 10  # make it large enough to chunk
    )
    p = tmp_path / "test_doc.txt"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def sample_md(tmp_path: Path) -> Path:
    """Create a small .md file."""
    content = (
        "# RoboMarket Overview\n\n"
        "RoboMarket connects buyers and sellers of automation equipment.\n\n"
        "## Key Features\n\n"
        "- AI-powered sourcing engine\n"
        "- Verified supplier network\n"
        "- REST and GraphQL APIs\n\n"
        "## Supported Brands\n\n"
        "FANUC, KUKA, ABB, Yaskawa, Universal Robots, Doosan.\n" * 8
    )
    p = tmp_path / "overview.md"
    p.write_text(content, encoding="utf-8")
    return p


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_txt_ingest_produces_chunks(sample_txt: Path) -> None:
    """Ingesting a .txt file must return at least one chunk."""
    chunks = ingest(sample_txt)
    assert len(chunks) > 0, "Expected at least one chunk from .txt ingestion"


def test_txt_chunks_have_source_metadata(sample_txt: Path) -> None:
    """Every chunk must carry a 'source' metadata key equal to the filename."""
    chunks = ingest(sample_txt)
    for chunk in chunks:
        assert "source" in chunk.metadata, "Chunk is missing 'source' metadata"
        assert chunk.metadata["source"] == sample_txt.name, (
            f"Expected source='{sample_txt.name}', "
            f"got '{chunk.metadata['source']}'"
        )


def test_txt_chunks_have_chunk_index_metadata(sample_txt: Path) -> None:
    """Every chunk must carry a 'chunk_index' metadata key."""
    chunks = ingest(sample_txt)
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata, "Chunk is missing 'chunk_index' metadata"
    # chunk_index values should be sequential starting from 0
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks))), (
        f"Expected sequential chunk indices 0..{len(chunks)-1}, got {indices}"
    )


def test_txt_chunks_have_page_metadata(sample_txt: Path) -> None:
    """Text file chunks must carry a 'page' metadata key (value -1 for non-PDFs)."""
    chunks = ingest(sample_txt)
    for chunk in chunks:
        assert "page" in chunk.metadata, "Chunk is missing 'page' metadata"
        assert chunk.metadata["page"] == -1, (
            f"Expected page=-1 for .txt file, got {chunk.metadata['page']}"
        )


def test_md_ingest_produces_chunks(sample_md: Path) -> None:
    """Ingesting a .md file must return at least one chunk."""
    chunks = ingest(sample_md)
    assert len(chunks) > 0, "Expected at least one chunk from .md ingestion"


def test_md_chunks_have_source_metadata(sample_md: Path) -> None:
    """MD file chunks must carry the correct 'source' filename."""
    chunks = ingest(sample_md)
    for chunk in chunks:
        assert chunk.metadata.get("source") == sample_md.name


def test_chunk_content_is_non_empty(sample_txt: Path) -> None:
    """No chunk should have empty page_content."""
    chunks = ingest(sample_txt)
    for chunk in chunks:
        assert chunk.page_content.strip(), "Chunk has empty page_content"


def test_unsupported_type_raises_value_error(tmp_path: Path) -> None:
    """Passing an unsupported file extension must raise ValueError."""
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("col1,col2\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        ingest(bad_file)


def test_missing_file_raises_file_not_found() -> None:
    """Passing a non-existent path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ingest(Path("/nonexistent/path/doc.txt"))


def test_sample_data_file_ingests(sample_data_dir: Path = None) -> None:
    """The bundled sample data file must ingest successfully."""
    sample = _APP_DIR / "sample_data" / "robomarket_overview.txt"
    if not sample.exists():
        pytest.skip("sample_data/robomarket_overview.txt not found")
    chunks = ingest(sample)
    assert len(chunks) > 0
    # Check that at least one chunk mentions "RoboMarket"
    texts = [c.page_content for c in chunks]
    assert any("RoboMarket" in t for t in texts), (
        "Expected at least one chunk to contain 'RoboMarket'"
    )
