# ollama_rag

A **fully local** RAG application — no cloud API keys required. Embeddings and LLM inference run entirely via [Ollama](https://ollama.com) on your machine.

## What it does

`ollama_rag` demonstrates the complete RAG pipeline running locally:

```
Upload → Ingest → Chunk → Embed (Ollama) → Index (Chroma) → Retrieve → Generate (Ollama) → Cite
```

---

## Tech stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit (themed via `shared/ui`) |
| Embeddings | Ollama `nomic-embed-text` (local) |
| Vector store | Chroma (persistent, on disk) |
| LLM | Ollama `llama3` (local) |
| RAG framework | LangChain (LCEL chain) |

---

## Prerequisites

1. Install [Ollama](https://ollama.com/download).
2. Pull the required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```

---

## Setup

```bash
cd local_rag/ollama_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # optional: adjust model names or base URL
streamlit run app.py
```

---

## Using the app

1. **Upload documents** — drag `.pdf`, `.txt`, or `.md` files into the uploader.
2. **Set a Store ID** — name the Chroma collection.
3. **Click "Ingest & Index"** — files are chunked and embedded locally with Ollama.
4. **Type a question** and click **"Ask"** — the app retrieves relevant chunks and generates a grounded answer using the local Ollama LLM.

---

## Chunking strategy

| Parameter | Value |
|-----------|-------|
| Splitter | `RecursiveCharacterTextSplitter` |
| Chunk size | 800 chars |
| Chunk overlap | 150 chars |
| Metadata | `source`, `page`, `chunk_index` |

---

## Running tests

```bash
pytest tests/ -v
```

Tests do **not** require Ollama to be running. Retrieval tests use `FakeEmbeddings` and pass an explicit `embeddings` override to `index()`.

| Test file | What it validates |
|-----------|-------------------|
| `tests/test_ingestion.py` | Chunking, metadata, error handling |
| `tests/test_retrieval.py` | Chroma indexing with FakeEmbeddings, retrieval, metadata |

---

## Known limitations

- Requires Ollama running locally at runtime (app won't start without it).
- LLM quality depends on the local model; `llama3` is recommended but `mistral` also works.
- Large documents may be slow to embed depending on hardware.
