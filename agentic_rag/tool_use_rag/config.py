"""App configuration — env vars and constants for tool_use_rag."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


class Config:
    # ── OpenAI ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # ── Chroma persistence ───────────────────────────────────────────────────
    CHROMA_BASE_DIR: Path = Path(__file__).parent / ".chroma"

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # ── Retrieval ────────────────────────────────────────────────────────────
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "4"))

    # ── Agent ────────────────────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "6"))

    # ── Answer confidence thresholds (relevance score, 0–1) ──────────────────
    CONFIDENCE_HIGH: float = 0.75
    CONFIDENCE_MEDIUM: float = 0.50
