"""shared.ui — design tokens and Streamlit theme helper.

Usage in any Streamlit app:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from shared.ui.theme import apply_theme, status_badge

    apply_theme()   # call once, before any st.* calls that render content
"""

import streamlit as st

TOKENS: dict[str, str] = {
    "bg_primary":    "#0A0A0F",
    "bg_surface":    "#12121A",
    "bg_elevated":   "#1A1A26",
    "accent":        "#6C63FF",
    "accent_dim":    "#3D3880",
    "text_primary":  "#F0F0FF",
    "text_secondary":"#8888AA",
    "border":        "#2A2A3A",
    "success":       "#22C55E",
    "error":         "#EF4444",
    "warning":       "#F59E0B",
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-primary:     #0A0A0F;
  --bg-surface:     #12121A;
  --bg-elevated:    #1A1A26;
  --accent:         #6C63FF;
  --accent-dim:     #3D3880;
  --text-primary:   #F0F0FF;
  --text-secondary: #8888AA;
  --border:         #2A2A3A;
  --success:        #22C55E;
  --error:          #EF4444;
  --warning:        #F59E0B;
}

/* ── Base ──────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.main,
.block-container {
  background-color: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  font-family: system-ui, sans-serif !important;
}

[data-testid="stSidebar"] {
  background-color: var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
}

/* ── Typography ────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
  font-family: 'Space Grotesk', system-ui, sans-serif !important;
  color: var(--text-primary) !important;
}
h1 { font-size: 40px !important; font-weight: 700 !important; }
h2 { font-size: 28px !important; font-weight: 700 !important; }
h3 { font-size: 20px !important; font-weight: 600 !important; }
p, li { font-size: 16px !important; line-height: 1.6 !important; }

/* ── Buttons ───────────────────────────────────────────────── */
.stButton > button {
  background-color: var(--accent) !important;
  color: var(--text-primary) !important;
  border: none !important;
  border-radius: 6px !important;
  font-family: 'Space Grotesk', system-ui, sans-serif !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 0.5rem 1.5rem !important;
  transition: background-color 0.15s ease !important;
}
.stButton > button:hover {
  background-color: var(--accent-dim) !important;
}
.stButton > button:disabled {
  opacity: 0.45 !important;
  cursor: not-allowed !important;
}

/* ── Inputs ────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea  > div > div > textarea {
  background-color: var(--bg-elevated) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  font-family: system-ui, sans-serif !important;
  font-size: 14px !important;
}
.stTextArea > div > div > textarea:focus,
.stTextInput > div > div > input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(108,99,255,0.25) !important;
  outline: none !important;
}
.stTextInput label,
.stTextArea  label {
  color: var(--text-secondary) !important;
  font-size: 14px !important;
  font-weight: 500 !important;
}

/* ── File uploader ─────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background-color: var(--bg-elevated) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 8px !important;
  padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label {
  color: var(--text-secondary) !important;
  font-size: 14px !important;
}

/* ── Selectbox ─────────────────────────────────────────────── */
.stSelectbox > div > div {
  background-color: var(--bg-elevated) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
}

/* ── Code / Mono ───────────────────────────────────────────── */
code, pre, .stCodeBlock {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 13px !important;
  background-color: var(--bg-elevated) !important;
  color: var(--text-primary) !important;
  border-radius: 4px !important;
}

/* ── Expander ──────────────────────────────────────────────── */
.streamlit-expanderHeader {
  background-color: var(--bg-surface) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  font-size: 14px !important;
}
.streamlit-expanderContent {
  background-color: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 6px 6px !important;
}

/* ── Metrics ───────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background-color: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 1rem !important;
}
[data-testid="stMetricValue"] {
  color: var(--text-primary) !important;
  font-family: 'Space Grotesk', system-ui, sans-serif !important;
}
[data-testid="stMetricLabel"] {
  color: var(--text-secondary) !important;
}

/* ── Divider ───────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Spinner ───────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Success / info / warning / error alerts ───────────────── */
.stAlert { border-radius: 6px !important; }

/* ── Scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar            { width: 6px; height: 6px; }
::-webkit-scrollbar-track      { background: var(--bg-primary); }
::-webkit-scrollbar-thumb      { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover{ background: var(--accent-dim); }

/* ── Hide default Streamlit chrome ─────────────────────────── */
#MainMenu, footer, header { visibility: hidden !important; }
</style>
"""

_STATUS_COLORS: dict[str, tuple[str, str]] = {
    "idle":      ("#8888AA", "#1A1A26"),
    "ingesting": ("#F59E0B", "#2A1E0A"),
    "indexing":  ("#6C63FF", "#18162E"),
    "querying":  ("#6C63FF", "#18162E"),
    "error":     ("#EF4444", "#2A0D0D"),
    "success":   ("#22C55E", "#0D2A18"),
}


def apply_theme() -> None:
    """Inject the robo-rag-apps design system into a Streamlit page.

    Call once at the top of your Streamlit app, before rendering any content.
    """
    st.markdown(_CSS, unsafe_allow_html=True)


def status_badge(state: str, message: str = "") -> str:
    """Return an HTML inline badge for a pipeline status state.

    Args:
        state:   One of: idle | ingesting | indexing | querying | error | success
        message: Override label text.  Defaults to the state name uppercased.

    Returns:
        HTML string — render with ``st.markdown(..., unsafe_allow_html=True)``.
    """
    text_color, bg_color = _STATUS_COLORS.get(state, ("#8888AA", "#1A1A26"))
    label = message or state.upper()
    return (
        f'<span style="'
        f'display:inline-block;'
        f'padding:3px 10px;'
        f'border-radius:4px;'
        f'background:{bg_color};'
        f'color:{text_color};'
        f'font-family:\'JetBrains Mono\',monospace;'
        f'font-size:12px;'
        f'font-weight:500;'
        f'border:1px solid {text_color}55;'
        f'letter-spacing:0.04em;'
        f'">{label}</span>'
    )
