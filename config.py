"""
config.py — Central configuration loader.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API keys ──────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── Paths ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
UPLOAD_DIR: Path = BASE_DIR / "data"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── App metadata ──────────────────────────────────────────
APP_TITLE: str = os.getenv("APP_TITLE", "Company Knowledge Base Q&A")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "company_kb")

# ── Model settings ────────────────────────────────────────
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Groq free-tier models (pick one in .env):
#   llama-3.3-70b-versatile   — smartest, 6000 RPM free
#   llama-3.1-8b-instant      — fastest, 14400 RPM free
#   mixtral-8x7b-32768        — long context, 5000 RPM free
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Chunking / retrieval ──────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 64))
TOP_K: int = int(os.getenv("TOP_K", 4))

# ── Supported file types ──────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def validate_config() -> list[str]:
    errors = []
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        errors.append(
            "GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com/keys"
        )
    return errors
