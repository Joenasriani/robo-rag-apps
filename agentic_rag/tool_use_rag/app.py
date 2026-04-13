"""tool_use_rag — Streamlit entry point.

Run:
    streamlit run app.py

Requires:
    OPENAI_API_KEY set in .env (see .env.example)
"""

import sys
import tempfile
import traceback
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from config import Config
from generation.chain import run_agent
from ingestion.loader import ingest
from retrieval.retriever import index as index_docs
from retrieval.retriever import load_store, make_retrieval_tool, store_exists
from shared.ui.theme import apply_theme, status_badge

st.set_page_config(
    page_title="Tool-Use RAG — RoboMarket",
    page_icon="🤖",
    layout="wide",
)
apply_theme()


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "store_id": "tool_use_rag_default",
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
    badge_html = status_badge(state, message)
    st.sidebar.markdown(badge_html, unsafe_allow_html=True)


def _validate_api_key() -> bool:
    if not Config.OPENAI_API_KEY:
        st.error(
            "⚠️ **OPENAI_API_KEY** is not set. "
            "Copy `.env.example` to `.env` and add your key, then restart."
        )
        return False
    return True


# ── Layout ────────────────────────────────────────────────────────────────────

st.markdown("# 🤖 Tool-Use RAG")
st.markdown(
    "An **agentic RAG** app where an LLM agent decides when and how to call "
    "the retrieval tool — iterating until it has enough context to answer."
)

col_left, col_right = st.columns([1, 2], gap="large")

# ── Sidebar / left column — ingestion ────────────────────────────────────────

with col_left:
    st.markdown("### 📥 Ingest Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    store_id_input = st.text_input(
        "Store ID",
        value=st.session_state["store_id"],
        help="Logical name for the Chroma collection. Use the same ID to resume a session.",
    )

    ingest_btn = st.button(
        "⚡ Ingest & Index",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if ingest_btn:
        if not _validate_api_key():
            st.rerun()

        store_id = store_id_input.strip() or "tool_use_rag_default"
        st.session_state["store_id"] = store_id
        all_chunks = []

        with st.spinner("Ingesting documents…"):
            _set_status("ingesting", "INGESTING")
            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix.lower()
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
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
            with st.spinner("Indexing…"):
                _set_status("indexing", "INDEXING")
                try:
                    store = index_docs(all_chunks, store_id)
                    st.session_state["store"] = store
                    _set_status("success", "INDEXED")
                    st.success(f"✅ Indexed **{len(all_chunks)}** chunks into `{store_id}`")
                except Exception as exc:
                    st.error(f"❌ Indexing failed: {exc}")
                    _set_status("error", "ERROR")

    # Load existing store
    st.markdown("---")
    st.markdown("#### Or load an existing store")
    load_id = st.text_input("Existing Store ID", placeholder="my-store-id")
    if st.button("Load Store", use_container_width=True):
        if not _validate_api_key():
            st.rerun()
        sid = load_id.strip()
        if not sid:
            st.warning("Enter a Store ID first.")
        elif not store_exists(sid):
            st.error(f"No store found for ID `{sid}`.")
        else:
            try:
                st.session_state["store"] = load_store(sid)
                st.session_state["store_id"] = sid
                st.success(f"✅ Loaded store `{sid}`")
                _set_status("success", "LOADED")
            except Exception as exc:
                st.error(f"❌ {exc}")

# ── Right column — agent query ────────────────────────────────────────────────

with col_right:
    st.markdown("### 🔍 Ask the Agent")

    query = st.text_area(
        "Your question",
        placeholder="What transaction fees does RoboMarket charge for large orders?",
        height=100,
    )

    ask_btn = st.button(
        "🧠 Ask Agent",
        use_container_width=True,
        disabled=not query,
    )

    if ask_btn:
        if not _validate_api_key():
            st.rerun()
        if st.session_state["store"] is None:
            st.warning("⚠️ Ingest or load a store first.")
        else:
            with st.spinner("Agent is thinking…"):
                _set_status("querying", "QUERYING")
                try:
                    tool = make_retrieval_tool(st.session_state["store"])
                    answer = run_agent(query, tools=[tool])
                    st.session_state["last_answer"] = answer
                    _set_status("success", "DONE")
                except Exception as exc:
                    st.error(f"❌ Agent error: {exc}")
                    _set_status("error", "ERROR")

    if st.session_state["last_answer"]:
        ans = st.session_state["last_answer"]
        conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
            ans["confidence"], "⚪"
        )

        st.markdown("#### Answer")
        st.markdown(ans["answer"])

        st.markdown(
            f"**Confidence:** {conf_color} {ans['confidence'].capitalize()} &nbsp;|&nbsp; "
            f"**Chunks retrieved:** {ans['retrieved_chunks']} &nbsp;|&nbsp; "
            f"**Agent steps:** {ans['agent_steps']}"
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
