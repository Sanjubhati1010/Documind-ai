
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def get_setting(name: str, default: str = "") -> str:
    """Read configuration from environment or Streamlit secrets."""

    value = os.getenv(name)

    if value and value.strip():
        return value.strip()

    try:
        import streamlit as st

        secret_value = st.secrets.get(name, default)
        return str(secret_value).strip()

    except Exception:
        return default


GOOGLE_API_KEY = get_setting("GOOGLE_API_KEY")

GEMINI_MODEL = get_setting(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

EMBEDDING_MODEL = get_setting(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_SIZE = int(get_setting("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(get_setting("CHUNK_OVERLAP", "150"))
RETRIEVAL_K = int(get_setting("RETRIEVAL_K", "4"))

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"


def validate_config() -> None:
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")

    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be non-negative "
            "and smaller than CHUNK_SIZE."
        )

    if RETRIEVAL_K <= 0:
        raise ValueError("RETRIEVAL_K must be greater than zero.")


def require_google_api_key() -> str:
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to your local .env "
            "or Streamlit Cloud app secrets."
        )

    return GOOGLE_API_KEY
