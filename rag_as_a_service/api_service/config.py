"""App configuration — env vars and constants for api_service."""

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

    # ── Server ───────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # ── Answer confidence thresholds ─────────────────────────────────────────
    CONFIDENCE_HIGH: float = 0.75
    CONFIDENCE_MEDIUM: float = 0.50
