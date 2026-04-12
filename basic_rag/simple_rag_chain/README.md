# simple_rag_chain

> **Category:** basic_rag &nbsp;|&nbsp; **Complexity:** ⭐ Beginner  
> **Use case:** Upload documents, index them into a local vector store, and ask questions answered grounded in your content — with traceable source citations.

---

## What it does

`simple_rag_chain` is the foundational RAG application in the `robo-rag-apps` library.
It demonstrates the complete RAG pipeline:

```
Upload → Ingest → Chunk → Embed → Index → Retrieve → Generate → Cite
```

Everything runs locally on your machine. The only external service is the OpenAI API
(embeddings + chat completion). The vector index persists on disk across restarts.

---

## Tech stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit (themed via `shared/ui`) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Chroma (persistent, on disk) |
| LLM | OpenAI `gpt-4o-mini` |
| RAG framework | LangChain (LCEL chain) |

---

## Setup

### Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Steps

```bash
# 1. Navigate to the app folder (from repo root)
cd basic_rag/simple_rag_chain

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Open .env and set OPENAI_API_KEY=sk-...

# 5. Run
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Using the app

1. **Upload documents** — drag `.pdf`, `.txt`, or `.md` files into the file uploader.
2. **Optionally set a Store ID** — name the collection. Using the same ID on restart
   will reuse the existing index (no need to re-ingest).
3. **Click "Ingest & Index"** — files are chunked and embedded. The status bar shows
   `INGESTING → INDEXING → INDEXED`.
4. **Type a question** in the query box.
5. **Click "Ask"** — the app retrieves the most relevant chunks, generates a grounded
   answer, and displays it alongside source citations.

### Persistent store

The Chroma vector store is saved under:
```
basic_rag/simple_rag_chain/.chroma/<store_id>/
```
On restart, set the same Store ID and skip the ingest step — the index is already there.

---

## Chunking strategy

Documents are split with LangChain's `RecursiveCharacterTextSplitter`:

| Parameter | Default | Override |
|-----------|---------|---------|
| `chunk_size` | 800 chars | `CHUNK_SIZE` env var |
| `chunk_overlap` | 150 chars | `CHUNK_OVERLAP` env var |

**Why recursive character splitting?**  
It tries multiple separators in order (`\n\n`, `\n`, ` `, `""`) and falls back to smaller
units, preserving sentence and paragraph boundaries where possible. Overlap ensures
cross-boundary context is not lost.

Every chunk is tagged with:
- `source` — original filename
- `page` — page number for PDFs (0-indexed); `-1` for text/markdown files
- `chunk_index` — sequential position across all chunks from that source

---

## Answer schema

Every answer includes:

```json
{
  "answer": "...",
  "sources": [
    {
      "document": "filename.pdf",
      "page_or_chunk": 2,
      "score": 0.87,
      "excerpt": "first 220 characters of the chunk..."
    }
  ],
  "confidence": "high | medium | low",
  "retrieved_chunks": 4
}
```

**Confidence** is derived from the top retrieval relevance score:

| Score | Confidence |
|-------|------------|
| ≥ 0.75 | high |
| ≥ 0.50 | medium |
| < 0.50 | low |

Sources are traceable: each `excerpt` is a verbatim extract from the chunk returned by
the retrieval step — no fabricated citations.

---

## Running tests

```bash
# From the app directory
pytest tests/ -v
```

Tests do **not** require an OpenAI API key. Retrieval tests use `FakeEmbeddings` from
`langchain-community` to build a temporary Chroma store.

| Test file | What it validates |
|-----------|-------------------|
| `tests/test_ingestion.py` | `.txt`/`.md` ingestion: chunk count, metadata (source, page, chunk_index), error handling |
| `tests/test_retrieval.py` | Chroma indexing, persistence, reload, retrieval non-empty results, metadata round-trip |

---

## Configuration

All settings are loaded from `.env` (or environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat completion model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `RETRIEVAL_K` | `4` | Number of chunks to retrieve per query |

---

## Known limitations

- **PDF page metadata**: `PyPDFLoader` provides page numbers on a best-effort basis.
  Complex PDFs (scanned, multi-column) may produce inaccurate page assignments.
- **Large files**: Ingesting very large PDFs (> 200 pages) may take 30–90 seconds
  depending on chunking. The status bar will show `INGESTING` during this time.
- **Semantic precision**: `text-embedding-3-small` is a high-quality model for most
  use cases. For specialized technical domains, consider `text-embedding-3-large`.
- **No cross-session deduplication**: Re-ingesting the same file into the same store
  will add duplicate vectors. Clear the `.chroma/<store_id>` folder to start fresh.
