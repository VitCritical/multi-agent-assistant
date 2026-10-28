"""
config.py — centralised settings loaded from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ── LangSmith ─────────────────────────────────────────────────────
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "asterline-research-assistant")

# ── Embeddings ────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Storage ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "data" / "memory.db"))
CORPUS_PATH: str = os.getenv("CORPUS_PATH", str(BASE_DIR / "corpus.jsonl"))

# ── RAG ───────────────────────────────────────────────────────────
TOP_K_CHUNKS: int = 6          # chunks retrieved per query
CHROMA_COLLECTION: str = "asterline_kb"

# ── Memory ────────────────────────────────────────────────────────
MEMORY_TYPES = ["preference", "correction", "investigation", "context"]

# ensure data dirs exist
Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
