import os
from pathlib import Path


class Config:
    # Business logic constants
    CHUNK_SIZE = 2048
    OVERLAP = 250

    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = str(PROJECT_ROOT / "data")
    LOG_DIR = str(PROJECT_ROOT / "logs")

    VECTOR_STORE_DIR = str(PROJECT_ROOT / "data" / "chromadb")

    # From environment
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
