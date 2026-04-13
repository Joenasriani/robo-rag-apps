# api_service

A **FastAPI-based RAG-as-a-Service** — expose your document knowledge base as a REST API with two endpoints: `POST /index` to ingest documents and `POST /query` to retrieve and answer questions.

## What it does

`api_service` wraps the complete RAG pipeline behind HTTP endpoints:

```
POST /index   →  Chunk + Embed + Store in Chroma
POST /query   →  Retrieve + Generate + Return grounded answer with citations
GET  /health  →  Service health check
```

---

## Tech stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI + uvicorn |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Chroma (persistent, on disk) |
| LLM | OpenAI `gpt-4o-mini` |
| RAG framework | LangChain (LCEL chain) |

---

## Setup

```bash
cd rag_as_a_service/api_service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY
python app.py
# → Uvicorn starts on http://localhost:8000
# → Interactive docs: http://localhost:8000/docs
```

---

## API Reference

### `GET /health`
```json
{ "status": "ok", "service": "robomarket-rag-api" }
```

### `POST /index`
**Request:**
```json
{
  "store_id": "my-knowledge-base",
  "documents": ["Full text of document 1...", "Full text of document 2..."],
  "source_names": ["doc1.txt", "doc2.txt"]
}
```
**Response (201):**
```json
{
  "store_id": "my-knowledge-base",
  "chunks_indexed": 42,
  "message": "Successfully indexed 42 chunks into store 'my-knowledge-base'."
}
```

### `POST /query`
**Request:**
```json
{
  "store_id": "my-knowledge-base",
  "query": "What authentication method does the API use?",
  "k": 4
}
```
**Response (200):**
```json
{
  "store_id": "my-knowledge-base",
  "query": "What authentication method does the API use?",
  "answer": "The RoboMarket API uses OAuth 2.0 bearer tokens for authentication.",
  "sources": [
    {
      "document": "api_docs.txt",
      "page_or_chunk": 0,
      "score": 0.91,
      "excerpt": "Authentication uses OAuth 2.0 bearer tokens..."
    }
  ],
  "confidence": "high",
  "retrieved_chunks": 4
}
```

---

## Running tests

```bash
pytest tests/ -v
```

Tests do **not** require an OpenAI API key. Retrieval tests use `FakeEmbeddings`. Endpoint tests mock the OpenAI calls with `unittest.mock`.

| Test file | What it validates |
|-----------|-------------------|
| `tests/test_ingestion.py` | File ingestion, `ingest_text()`, chunking, metadata |
| `tests/test_retrieval.py` | Chroma indexing, retrieval, FastAPI endpoint contracts |

---

## Known limitations

- Requires `OPENAI_API_KEY` at runtime for `/index` and `/query`.
- The Chroma store persists in `.chroma/<store_id>/` relative to the app directory.
- No authentication is implemented on the API itself — add a reverse proxy or middleware for production deployments.
