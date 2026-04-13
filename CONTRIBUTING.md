# Contributing to robo-rag-apps

Thank you for contributing. This repository holds production-grade RAG applications. Every contribution must meet the global standard below — no exceptions.

---

## Global standard (non-negotiable)

Every app in this repo must:

1. **Ingest real documents or structured data** — no hardcoded strings passed off as documents.
2. **Chunk with a real strategy** — document chunker with explicit size, overlap, and metadata (source, page, chunk index).
3. **Embed using a real embedding model** — OpenAI, a local Ollama model, or another real provider.
4. **Index into a real, persistent vector store** — Chroma, Qdrant, FAISS, or similar. The store must survive process restarts.
5. **Retrieve relevant chunks at query time** — similarity search returning actual scored results.
6. **Generate answers grounded in retrieved context** — the LLM prompt must contain the retrieved chunks; the answer must not be fabricated.
7. **Return real source references** — every citation must trace back to a chunk returned by the retrieval step.
8. **Run independently** — `pip install -r requirements.txt && python app.py` (or `streamlit run app.py`) is the only command a developer needs.

### Prohibited — without exception

- Placeholder functions that print `TODO` or `pass`
- Mocked retrieval returning hardcoded strings
- Simulated citations or fake sources
- UI buttons wired to nothing
- Sample outputs presented as live results
- Features present in the UI that have no backend implementation

If you cannot implement something for real, **document it as a known limitation** in the app README instead of shipping a fake version.

---

## App folder structure

Every app must follow this layout:

```
<category>/<app_name>/
├── README.md            # Use case, setup, chunking strategy, known limitations
├── app.py               # Entry point (Streamlit or CLI)
├── config.py            # Env var loading and constants
├── requirements.txt     # Pinned dependencies
├── .env.example         # Template — never commit a real .env
├── ingestion/
│   └── loader.py        # ingest(source) → List[Document]
├── retrieval/
│   └── retriever.py     # index() and retrieve()
├── generation/
│   └── chain.py         # generate(query, context) → Answer
├── tests/
│   ├── test_ingestion.py
│   └── test_retrieval.py
└── sample_data/         # Optional small real sample file for smoke-testing
```

---

## Design system

All apps share one visual identity defined in `shared/ui/`. **Never recreate the theme inside an app.** Import and apply it:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.ui.theme import apply_theme
apply_theme()
```

Design tokens are defined in `shared/ui/theme.py`. Do not hardcode colors or fonts.

---

## Logging

Emit structured JSON logs at each stage using the pattern established in App 1:

| Stage | Required fields |
|-------|----------------|
| ingestion | `status`, `file`, `chunks_produced`, `duration_ms` |
| indexing | `status`, `store_id`, `vectors_written`, `duration_ms` |
| retrieval | `status`, `query_hash`, `chunks_returned`, `top_score` |
| generation | `status`, `tokens_used`, `confidence` |
| error | `stage`, `message`, `traceback` |

---

## Answer schema

Every `generate()` function must return a dict matching:

```json
{
  "answer": "string",
  "sources": [
    {
      "document": "filename.pdf",
      "page_or_chunk": 3,
      "score": 0.87,
      "excerpt": "first 200 chars of the chunk..."
    }
  ],
  "confidence": "high | medium | low",
  "retrieved_chunks": 4
}
```

---

## Tests

Every app must include:
- `tests/test_ingestion.py` — validates real file ingestion produces chunks with correct metadata.
- `tests/test_retrieval.py` — builds a temporary vector store, indexes test documents, and asserts that a query returns relevant, non-empty results.

Tests must pass without requiring external API keys (mock or skip API-dependent steps).

Run tests **per-app** from the repo root (this is also what CI does):
```bash
pytest -q basic_rag/simple_rag_chain/tests
pytest -q agentic_rag/tool_use_rag/tests
pytest -q local_rag/ollama_rag/tests
pytest -q rag_as_a_service/api_service/tests
```

Or from within the app folder:
```bash
cd basic_rag/simple_rag_chain && pytest tests/
```

> **Note:** Running `pytest` from the repo root without specifying a path will fail.
> All apps share identical internal package names (`ingestion/`, `retrieval/`, `generation/`), which clash in a single Python process.
> Always run tests one app at a time.

---

## PR checklist

Before opening a PR, confirm:

- [ ] App runs via `streamlit run app.py` following only its README
- [ ] No `TODO`, `pass`, or hardcoded fake data in shipped code
- [ ] All UI elements are wired to real backend behavior
- [ ] Vector store persists across restarts (verify by restarting and querying)
- [ ] Sources in answers trace to real retrieved chunks
- [ ] `pytest tests/` passes
- [ ] `.env.example` is present; real `.env` is not committed
- [ ] App uses `shared/ui/theme.py` — no inline design tokens
