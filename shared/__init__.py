"""shared.ui — design system for robo-rag-apps.

Import apply_theme() in any Streamlit app to apply the shared visual identity.
"""
from shared.ui.theme import apply_theme, status_badge, TOKENS

__all__ = ["apply_theme", "status_badge", "TOKENS"]
