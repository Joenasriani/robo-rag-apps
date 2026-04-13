"""App configuration — env vars and constants for ollama_rag."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


class Config:
    # ── Ollama ───────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "llama3")

    # ── Chroma persistence ───────────────────────────────────────────────────
    CHROMA_BASE_DIR: Path = Path(__file__).parent / ".chroma"

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # ── Retrieval ────────────────────────────────────────────────────────────
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "4"))

    # ── Answer confidence thresholds ─────────────────────────────────────────
    CONFIDENCE_HIGH: float = 0.75
    CONFIDENCE_MEDIUM: float = 0.50
