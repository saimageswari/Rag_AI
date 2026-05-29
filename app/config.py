from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_DIR = BASE_DIR / "data" / "vectorstores"
EXPORT_DIR = BASE_DIR / "exports"
HISTORY_DIR = BASE_DIR / "data" / "history"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls", ".json"}


@dataclass(frozen=True)
class AppConfig:
    llm_provider: str = os.getenv("LLM_PROVIDER", "local").strip().lower()
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "650"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    top_k: int = int(os.getenv("TOP_K", "4"))


def ensure_directories() -> None:
    for directory in (UPLOAD_DIR, VECTOR_DIR, EXPORT_DIR, HISTORY_DIR):
        directory.mkdir(parents=True, exist_ok=True)
