# conftest.py — add this app's directory to sys.path for test collection.
# This is required when running pytest from the repo root so that unqualified
# imports like `from ingestion.loader import ingest` resolve to this app.
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
