# robo-rag-apps

> A curated library of production-grade RAG applications focused on the RoboMarket domain.
> Every app ingests real data, embeds with a real model, stores in a real vector DB, retrieves real chunks, and generates grounded answers — no placeholders, no mocks.

---

## What this repo is

`robo-rag-apps` is an end-to-end reference library for Retrieval-Augmented Generation (RAG) built around RoboMarket use cases.

---

## How it is organized

Apps are grouped by RAG complexity tier. Every app lives in its own folder with its own `requirements.txt`, `.env.example`, and `README.md`.

### Current repo structure (implemented)

```text
robo-rag-apps/
├── shared/                     # Design system, reusable helpers (import, don't copy)
│   └── ui/                     # Streamlit theme, design tokens
├── basic_rag/                  # Entry-level RAG patterns
│   └── simple_rag_chain/       # Classic RAG chain with Streamlit UI
├── agentic_rag/                # Agent-driven RAG patterns
│   └── tool_use_rag/           # ReAct agent with retrieval tool
├── local_rag/                  # Fully local RAG (no cloud API keys)
│   └── ollama_rag/             # Ollama embeddings + Ollama LLM + Chroma
└── rag_as_a_service/           # RAG exposed as a REST API
    └── api_service/            # FastAPI service with /index and /query
```

> `advanced_rag/` is planned but not yet implemented.

---

## App catalog

| App | Category | Complexity | Embeddings | Vector Store | Run command | Status |
|-----|----------|------------|------------|--------------|-------------|--------|
| [simple_rag_chain](basic_rag/simple_rag_chain/) | basic_rag | ⭐ Beginner | OpenAI | Chroma | `streamlit run app.py` | ✅ Ready |
| [tool_use_rag](agentic_rag/tool_use_rag/) | agentic_rag | ⭐⭐ Intermediate | OpenAI | Chroma | `streamlit run app.py` | ✅ Ready |
| [ollama_rag](local_rag/ollama_rag/) | local_rag | ⭐⭐ Intermediate | Ollama (local) | Chroma | `streamlit run app.py` | ✅ Ready |
| [api_service](rag_as_a_service/api_service/) | rag_as_a_service | ⭐⭐ Intermediate | OpenAI | Chroma | `python app.py` | ✅ Ready |

---

## Quick start — run an app in under 5 minutes

### Streamlit apps (simple_rag_chain, tool_use_rag, ollama_rag)

```bash
# 1. Clone the repo
git clone https://github.com/Joenasriani/robo-rag-apps.git
cd robo-rag-apps

# 2. Pick an app (example: simple_rag_chain)
cd basic_rag/simple_rag_chain

# 3. Create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY (or Ollama settings for ollama_rag)

# 6. Run
streamlit run app.py
```

### FastAPI service (api_service)

```bash
cd rag_as_a_service/api_service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY
python app.py
# -> http://localhost:8000/docs   (interactive API docs)
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full global standard every app in this repo must meet.

**Key rules at a glance:**
- Every app must ingest real data, embed with a real model, and store in a real persistent vector DB.
- No placeholder functions, mocked retrieval, or fake citations — ever.
- Each app must run independently via `pip install -r requirements.txt && python app.py` (or `streamlit run app.py`).
- All apps share the design system defined in `shared/ui/`. Never reinvent it.
- PRs without passing tests will not be merged.