"""ollama_rag — Streamlit entry point.

Run:
    streamlit run app.py

Requires:
    Ollama running locally with nomic-embed-text and llama3 pulled.
    No external API keys needed.
"""

import sys
import tempfile
import traceback
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from config import Config
from generation.chain import generate
from ingestion.loader import ingest
from retrieval.retriever import index as index_docs
from retrieval.retriever import load_store, retrieve, store_exists
from shared.ui.theme import apply_theme, status_badge

st.set_page_config(
    page_title="Local RAG (Ollama) — RoboMarket",
    page_icon="🦙",
    layout="wide",
)
apply_theme()


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "store_id": "ollama_rag_default",
        "status": "idle",
        "store": None,
        "last_answer": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _set_status(state: str, message: str = "") -> None:
    st.session_state["status"] = state
    st.sidebar.markdown(status_badge(state, message), unsafe_allow_html=True)


def _check_ollama() -> bool:
    """Ping Ollama health endpoint and warn if not reachable."""
    import urllib.request
    try:
        urllib.request.urlopen(Config.OLLAMA_BASE_URL, timeout=3)
        return True
    except Exception:
        st.warning(
            f"⚠️ Ollama not reachable at `{Config.OLLAMA_BASE_URL}`. "
            "Start Ollama and pull the required models:\n"
            f"`ollama pull {Config.OLLAMA_EMBED_MODEL}` and "
            f"`ollama pull {Config.OLLAMA_LLM_MODEL}`"
        )
        return False


# ── Layout ────────────────────────────────────────────────────────────────────

st.markdown("# 🦙 Local RAG — Ollama")
st.markdown(
    "A **fully local** RAG app — no cloud API keys. "
    "Embeddings and generation run via [Ollama](https://ollama.com) on your machine."
)

col_left, col_right = st.columns([1, 2], gap="large")

# ── Left column — ingestion ───────────────────────────────────────────────────

with col_left:
    st.markdown("### 📥 Ingest Documents")

    st.markdown(
        f"**Models:** embed=`{Config.OLLAMA_EMBED_MODEL}` | llm=`{Config.OLLAMA_LLM_MODEL}`"
    )

    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    store_id_input = st.text_input(
        "Store ID",
        value=st.session_state["store_id"],
    )

    ingest_btn = st.button(
        "⚡ Ingest & Index",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if ingest_btn:
        if not _check_ollama():
            st.stop()

        store_id = store_id_input.strip() or "ollama_rag_default"
        st.session_state["store_id"] = store_id
        all_chunks = []

        with st.spinner("Ingesting documents…"):
            _set_status("ingesting", "INGESTING")
            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix.lower()
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = Path(tmp.name)
                    chunks = ingest(tmp_path)
                    for c in chunks:
                        c.metadata["source"] = uploaded_file.name
                    all_chunks.extend(chunks)
                    tmp_path.unlink(missing_ok=True)
                except Exception as exc:
                    st.error(f"❌ Failed to ingest **{uploaded_file.name}**: {exc}")
                    _set_status("error", "ERROR")

        if all_chunks:
            with st.spinner("Embedding with Ollama…"):
                _set_status("indexing", "INDEXING")
                try:
                    store = index_docs(all_chunks, store_id)
                    st.session_state["store"] = store
                    _set_status("success", "INDEXED")
                    st.success(f"✅ Indexed **{len(all_chunks)}** chunks into `{store_id}`")
                except Exception as exc:
                    st.error(f"❌ Indexing failed: {exc}")
                    _set_status("error", "ERROR")

    st.markdown("---")
    st.markdown("#### Or load an existing store")
    load_id = st.text_input("Existing Store ID", placeholder="my-store-id")
    if st.button("Load Store", use_container_width=True):
        sid = load_id.strip()
        if not sid:
            st.warning("Enter a Store ID.")
        elif not store_exists(sid):
            st.error(f"No store for `{sid}`.")
        else:
            try:
                st.session_state["store"] = load_store(sid)
                st.session_state["store_id"] = sid
                st.success(f"✅ Loaded `{sid}`")
                _set_status("success", "LOADED")
            except Exception as exc:
                st.error(f"❌ {exc}")

# ── Right column — query ──────────────────────────────────────────────────────

with col_right:
    st.markdown("### 🔍 Ask a Question")

    query = st.text_area(
        "Your question",
        placeholder="What types of robots are available on RoboMarket?",
        height=100,
    )

    ask_btn = st.button(
        "🔎 Ask",
        use_container_width=True,
        disabled=not query,
    )

    if ask_btn:
        if st.session_state["store"] is None:
            st.warning("⚠️ Ingest or load a store first.")
        else:
            with st.spinner("Retrieving and generating with Ollama…"):
                _set_status("querying", "QUERYING")
                try:
                    chunks = retrieve(query, st.session_state["store"])
                    answer = generate(query, chunks)
                    st.session_state["last_answer"] = answer
                    _set_status("success", "DONE")
                except Exception as exc:
                    st.error(f"❌ {exc}")
                    _set_status("error", "ERROR")

    if st.session_state["last_answer"]:
        ans = st.session_state["last_answer"]
        conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(ans["confidence"], "⚪")

        st.markdown("#### Answer")
        st.markdown(ans["answer"])
        st.markdown(
            f"**Confidence:** {conf_color} {ans['confidence'].capitalize()} &nbsp;|&nbsp; "
            f"**Chunks retrieved:** {ans['retrieved_chunks']}"
        )

        if ans["sources"]:
            st.markdown("#### Sources")
            for src in ans["sources"]:
                loc = src["page_or_chunk"]
                loc_label = f"page {loc}" if loc >= 0 else f"chunk {abs(loc)}"
                with st.expander(
                    f"📄 {src['document']} — {loc_label} (score: {src['score']:.2f})"
                ):
                    st.markdown(f"*{src['excerpt']}*")
