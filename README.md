# robo-rag-apps

> A curated library of production-grade RAG applications focused on the RoboMarket domain.  
> Every app ingests real data, embeds with a real model, stores in a real vector DB, retrieves real chunks, and generates grounded answers — no placeholders, no mocks.

---

## What this repo is

`robo-rag-apps` is an end-to-end reference library for Retrieval-Augmented Generation (RAG) built around RoboMarket use cases. Each application is **self-contained**: clone the repo, `cd` into any app folder, install its requirements, and run it — all in under five minutes.

## Why it exists

RoboMarket integrations require grounded, traceable answers drawn from product catalogs, technical manuals, supplier contracts, and market intelligence. This library provides battle-tested, production-ready RAG patterns that can be lifted directly into RoboMarket services. It is organized so that engineers can study individual patterns in isolation, then compose them for production deployment.

## How it is organized

Apps are grouped by RAG complexity tier. Every app lives in its own folder with its own `requirements.txt`, `.env.example`, and `README.md`.

```
robo-rag-apps/
├── shared/            # Design system, reusable helpers (import, don't copy)
│   └── ui/            # Streamlit theme, design tokens
├── basic_rag/         # Entry-level RAG patterns
├── advanced_rag/      # Hybrid search, knowledge graphs, vision, diagnostics
├── agentic_rag/       # Autonomous agents with RAG cores
├── rag_as_a_service/  # Deployable RAG APIs
└── local_rag/         # Fully offline, open-weight model RAG
```

---

## App catalog

| App | Category | Complexity | Embeddings | Vector Store | Status |
|-----|----------|------------|------------|--------------|--------|
| [simple_rag_chain](basic_rag/simple_rag_chain/) | basic_rag | ⭐ Beginner | OpenAI | Chroma (local) | ✅ Ready |
| rag_with_database_routing | basic_rag | ⭐⭐ Intermediate | — | — | 🔜 Planned |
| hybrid_search_rag | advanced_rag | ⭐⭐ Intermediate | — | — | 🔜 Planned |
| knowledge_graph_rag | advanced_rag | ⭐⭐⭐ Advanced | — | — | 🔜 Planned |
| vision_rag | advanced_rag | ⭐⭐⭐ Advanced | — | — | 🔜 Planned |
| rag_failure_diagnostics | advanced_rag | ⭐⭐⭐ Advanced | — | — | 🔜 Planned |
| agentic_rag_with_reasoning | agentic_rag | ⭐⭐⭐ Advanced | — | — | 🔜 Planned |
| corrective_rag | agentic_rag | ⭐⭐⭐ Advanced | — | — | 🔜 Planned |
| contextual_rag_agent | agentic_rag | ⭐⭐⭐ Advanced | — | — | 🔜 Planned |
| autonomous_rag | agentic_rag | ⭐⭐⭐⭐ Expert | — | — | 🔜 Planned |
| llama_local_rag | local_rag | ⭐⭐ Intermediate | Ollama | Chroma (local) | 🔜 Planned |
| deepseek_local_rag | local_rag | ⭐⭐ Intermediate | Ollama | Chroma (local) | 🔜 Planned |

---

## Quick start — run any app in under 5 minutes

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
# Edit .env — add your OPENAI_API_KEY

# 6. Run
streamlit run app.py
```

Open the URL shown in your terminal (default: http://localhost:8501).  
See the app's own **README.md** for detailed instructions and configuration options.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full global standard every app in this repo must meet.

**Key rules at a glance:**
- Every app must ingest real data, embed with a real model, and store in a real persistent vector DB.
- No placeholder functions, mocked retrieval, or fake citations — ever.
- Each app must run independently via `pip install -r requirements.txt && python app.py` (or `streamlit run app.py`).
- All apps share the design system defined in `shared/ui/`. Never reinvent it.
- PRs without passing tests will not be merged.
