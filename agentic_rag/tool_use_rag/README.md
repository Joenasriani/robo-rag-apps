# tool_use_rag

An **agentic RAG** application where an LLM agent autonomously decides when and how to invoke the retrieval tool — iterating with refined queries until it has sufficient context to answer the question.

## What it does

`tool_use_rag` implements a **ReAct (Reason + Act) agent** loop:

```
Question → Agent thinks → Calls retrieve_documents tool (may repeat) → Synthesises grounded answer
```

Unlike simple RAG chains, the agent can:
- Issue **multiple retrieval queries** with progressively refined terms.
- Decide that **no retrieval is needed** if the question is unanswerable.
- Explicitly **cite sources** in its final answer.

---

## Tech stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit (themed via `shared/ui`) |
| Agent framework | LangChain ReAct agent |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Chroma (persistent, on disk) |
| LLM | OpenAI `gpt-4o-mini` |

---

## Setup

```bash
cd agentic_rag/tool_use_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY
streamlit run app.py
```

---

## Using the app

1. **Upload documents** — drag `.pdf`, `.txt`, or `.md` files into the uploader.
2. **Set a Store ID** — name the Chroma collection.
3. **Click "Ingest & Index"** — documents are chunked, embedded, and stored.
4. **Type a question** in the query box.
5. **Click "Ask Agent"** — the agent runs its ReAct loop, calls retrieval one or more times, and returns a grounded answer with source citations.

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

Tests do **not** require an OpenAI API key. `FakeEmbeddings` is used to build a temporary Chroma store.

| Test file | What it validates |
|-----------|-------------------|
| `tests/test_ingestion.py` | Chunking, metadata (source, page, chunk_index), error handling |
| `tests/test_retrieval.py` | Chroma indexing, retrieval, retrieval tool, metadata round-trip |

---

## Known limitations

- The ReAct agent requires `OPENAI_API_KEY` at runtime; it cannot run offline.
- Long documents may require increasing `AGENT_MAX_ITERATIONS` in `.env`.
