"""simple_rag_chain — Streamlit entry point.

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

# ── Path setup: allow importing shared/ from the repo root ───────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ── App-local imports ────────────────────────────────────────────────────────
from config import Config
from generation.chain import generate
from ingestion.loader import ingest
from retrieval.retriever import index as index_docs
from retrieval.retriever import load_store, retrieve, store_exists
from shared.ui.theme import apply_theme, status_badge

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Simple RAG Chain — RoboMarket",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_theme()

# ── Session state defaults ────────────────────────────────────────────────────
_defaults: dict = {
    "status": "idle",
    "status_msg": "Ready",
    "store": None,
    "store_id": getattr(Config, "DEFAULT_STORE_ID", "simple_rag_default"),
    "answer": None,
    "error": None,
    "chunks_indexed": 0,
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Helpers ───────────────────────────────────────────────────────────────────


def _set_status(state: str, msg: str = "") -> None:
    st.session_state["status"] = state
    st.session_state["status_msg"] = msg or state.upper()
    st.session_state["error"] = None


def _set_error(msg: str) -> None:
    st.session_state["status"] = "error"
    st.session_state["status_msg"] = "ERROR"
    st.session_state["error"] = msg


def _validate_api_key() -> bool:
    if not Config.OPENAI_API_KEY:
        _set_error(
            "OPENAI_API_KEY is not set. Add it to your .env file and restart the app."
        )
        return False
    return True


# ── Layout ────────────────────────────────────────────────────────────────────

# Header
st.markdown(
    "<h1>🤖 Simple RAG Chain</h1>"
    "<p style='color:#8888AA;font-size:16px;margin-top:-12px;'>"
    "Upload documents, index them into a persistent Chroma store, "
    "and ask grounded questions answered from your content."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# Status bar
status_col, _ = st.columns([3, 9])
with status_col:
    st.markdown(
        status_badge(st.session_state["status"], st.session_state["status_msg"]),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([5, 7], gap="large")

# ══ LEFT COLUMN — Ingest & Index ══════════════════════════════════════════════
with left_col:
    st.markdown("### 📂 Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supported formats: PDF, plain text (.txt), Markdown (.md)",
    )

    store_id_input = st.text_input(
        "Store ID",
        value=st.session_state["store_id"],
        help="Logical name for the Chroma collection. "
        "Use the same ID to resume a previous session.",
    )

    ingest_btn = st.button(
        "⚡ Ingest & Index",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if ingest_btn:
        if not _validate_api_key():
            st.rerun()

        store_id = store_id_input.strip() or "simple_rag_default"
        st.session_state["store_id"] = store_id
        all_chunks = []

        with st.spinner("Ingesting documents…"):
            _set_status("ingesting", "INGESTING")
            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix.lower()
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix, delete=False
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp.name)
                    chunks = ingest(tmp_path)
                    # Override source metadata with the original filename
                    for chunk in chunks:
                        chunk.metadata["source"] = uploaded_file.name
                    all_chunks.extend(chunks)
                    tmp_path.unlink(missing_ok=True)
                except Exception as exc:
                    _set_error(
                        f"Ingestion failed for '{uploaded_file.name}': {exc}\n\n"
                        + traceback.format_exc()
                    )
                    st.rerun()

        if all_chunks:
            with st.spinner(f"Indexing {len(all_chunks)} chunks into Chroma…"):
                _set_status("indexing", "INDEXING")
                try:
                    store = index_docs(all_chunks, store_id)
                    st.session_state["store"] = store
                    st.session_state["chunks_indexed"] = len(all_chunks)
                    _set_status(
                        "success",
                        f"INDEXED — {len(all_chunks)} chunks",
                    )
                except Exception as exc:
                    _set_error(
                        f"Indexing failed: {exc}\n\n" + traceback.format_exc()
                    )
                    st.rerun()

        st.rerun()

    # Show existing store if available
    if st.session_state["store"] is None and store_exists(store_id_input.strip()):
        st.markdown(
            f"<small style='color:#8888AA;'>Persistent store "
            f"<code>{store_id_input.strip()}</code> found on disk. "
            f"Enter a query to use it.</small>",
            unsafe_allow_html=True,
        )

    # Chunk count badge
    if st.session_state["chunks_indexed"] > 0:
        st.markdown(
            f"<small style='color:#22C55E;'>✓ {st.session_state['chunks_indexed']}"
            f" chunks indexed into <code>{st.session_state['store_id']}</code></small>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💬 Query")

    query_text = st.text_area(
        "Ask a question about your documents",
        placeholder="e.g. What robot brands are available on RoboMarket?",
        height=120,
    )

    ask_btn = st.button(
        "🔍 Ask",
        use_container_width=True,
        disabled=not query_text.strip(),
    )

    if ask_btn:
        if not _validate_api_key():
            st.rerun()

        query = query_text.strip()

        # Load store from disk if not already in session
        active_store = st.session_state["store"]
        sid = st.session_state["store_id"]
        if active_store is None:
            if store_exists(sid):
                try:
                    active_store = load_store(sid)
                    st.session_state["store"] = active_store
                except Exception as exc:
                    _set_error(f"Could not load store '{sid}': {exc}")
                    st.rerun()
            else:
                _set_error(
                    "No indexed documents found. "
                    "Upload files and click 'Ingest & Index' first."
                )
                st.rerun()

        with st.spinner("Retrieving relevant chunks…"):
            _set_status("querying", "QUERYING")
            try:
                chunks = retrieve(query, active_store, k=Config.RETRIEVAL_K)
            except Exception as exc:
                _set_error(f"Retrieval failed: {exc}\n\n" + traceback.format_exc())
                st.rerun()

        with st.spinner("Generating answer…"):
            try:
                answer = generate(query, chunks)
                st.session_state["answer"] = answer
                _set_status("success", "DONE")
            except Exception as exc:
                _set_error(
                    f"Generation failed: {exc}\n\n" + traceback.format_exc()
                )
                st.rerun()

        st.rerun()

# ══ RIGHT COLUMN — Answer & Sources ══════════════════════════════════════════
with right_col:
    st.markdown("### 📋 Answer")

    # Error display
    if st.session_state["error"]:
        st.markdown(
            f'<div style="background:#2A0D0D;border:1px solid #EF444455;'
            f'border-radius:8px;padding:1rem;color:#EF4444;'
            f'font-family:\'JetBrains Mono\',monospace;font-size:13px;'
            f'white-space:pre-wrap;">'
            f'⛔ {st.session_state["error"]}'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    answer_data = st.session_state["answer"]

    if answer_data is None:
        st.markdown(
            '<div style="background:#12121A;border:1px solid #2A2A3A;'
            'border-radius:8px;padding:2rem;color:#8888AA;text-align:center;">'
            "Answer will appear here after you ask a question."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # Confidence badge
        conf = answer_data["confidence"]
        conf_color = {"high": "#22C55E", "medium": "#F59E0B", "low": "#EF4444"}.get(
            conf, "#8888AA"
        )
        st.markdown(
            f'<span style="font-size:12px;color:{conf_color};'
            f'font-family:\'JetBrains Mono\',monospace;">'
            f"Confidence: {conf.upper()} &nbsp;|&nbsp; "
            f"Chunks retrieved: {answer_data['retrieved_chunks']}"
            f"</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Answer text
        st.markdown(
            f'<div style="background:#12121A;border:1px solid #2A2A3A;'
            f'border-radius:8px;padding:1.25rem;color:#F0F0FF;font-size:15px;'
            f'line-height:1.7;">{answer_data["answer"]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Sources panel
        st.markdown("### 🔗 Sources")
        sources = answer_data.get("sources", [])
        if not sources:
            st.markdown(
                '<small style="color:#8888AA;">No sources available.</small>',
                unsafe_allow_html=True,
            )
        else:
            for i, src in enumerate(sources, start=1):
                is_pdf = src["document"].lower().endswith(".pdf")
                page_label = (
                    f"page {src['page_or_chunk']}"
                    if is_pdf
                    else f"chunk {src['page_or_chunk']}"
                )
                with st.expander(
                    f"[{i}] {src['document']} — {page_label} "
                    f"(score {src['score']:.2f})"
                ):
                    st.markdown(
                        f'<div style="background:#1A1A26;border-radius:6px;'
                        f'padding:0.75rem;color:#F0F0FF;'
                        f'font-family:\'JetBrains Mono\',monospace;font-size:13px;">'
                        f'{src["excerpt"]}'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<small style='color:#8888AA;'>"
                        f"Document: <code>{src['document']}</code> &nbsp;|&nbsp; "
                        f"Location: {page_label} &nbsp;|&nbsp; "
                        f"Relevance score: <code>{src['score']:.4f}</code>"
                        f"</small>",
                        unsafe_allow_html=True,
                    )
