import os
from pathlib import Path


def _get_default_log_dir():
    return "/app/logs" if Path("/.dockerenv").exists() else "../logs"


def _get_default_vector_dir():
    return "/app/data/chromadb" if Path("/.dockerenv").exists() else "../data/chromadb"


class Config:
    # Business logic constants
    CHUNK_SIZE = 2048
    OVERLAP = 250

    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    LOG_DIR = PROJECT_ROOT / "logs"

    # From environment
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

