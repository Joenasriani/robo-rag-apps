"""tests/test_shared_theme.py — verify shared.ui.theme imports and functions correctly.

These tests do NOT require Streamlit to be running; they only verify that the
module can be imported and that its public functions return the expected types.
"""

import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# shared/ lives two levels above the app directory (repo root).
_APP_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _APP_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_theme_module_imports() -> None:
    """shared.ui.theme must be importable without errors."""
    from shared.ui import theme  # noqa: F401

    assert theme is not None


def test_apply_theme_is_callable() -> None:
    """apply_theme must be a callable exported from shared.ui.theme."""
    from shared.ui.theme import apply_theme

    assert callable(apply_theme)


def test_status_badge_is_callable() -> None:
    """status_badge must be a callable exported from shared.ui.theme."""
    from shared.ui.theme import status_badge

    assert callable(status_badge)


def test_status_badge_returns_html_string() -> None:
    """status_badge(state, message) must return a non-empty HTML string."""
    from shared.ui.theme import status_badge

    for state in ("idle", "ingesting", "indexing", "querying", "error", "success"):
        result = status_badge(state, state.upper())
        assert isinstance(result, str), f"Expected str for state={state}"
        assert len(result) > 0, f"Expected non-empty HTML for state={state}"
        assert "<span" in result, f"Expected HTML <span> element for state={state}"


def test_status_badge_unknown_state_still_returns_string() -> None:
    """status_badge with an unknown state must not raise and must return a string."""
    from shared.ui.theme import status_badge

    result = status_badge("unknown_state", "UNKNOWN")
    assert isinstance(result, str)
    assert len(result) > 0


def test_tokens_dict_has_required_keys() -> None:
    """TOKENS dict must contain the core design token keys."""
    from shared.ui.theme import TOKENS

    required_keys = {
        "bg_primary",
        "bg_surface",
        "accent",
        "text_primary",
        "text_secondary",
        "border",
        "success",
        "error",
    }
    for key in required_keys:
        assert key in TOKENS, f"TOKENS is missing key: {key!r}"
        assert isinstance(TOKENS[key], str), f"TOKENS[{key!r}] must be a str"
        assert TOKENS[key].startswith("#"), f"TOKENS[{key!r}] must be a hex colour"
