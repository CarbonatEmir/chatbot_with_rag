from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    database_url: str
    ollama_model: str
    embedding_model: str
    rag_top_k: int
    rag_min_score: float
    log_level: str
    admin_password: str


def load_settings() -> Settings:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        # Backward-compat for existing codebase; encourages .env usage.
        database_url = "postgresql://postgres:1357913@localhost:5432/lasersan_Ai"

    return Settings(
        database_url=database_url,
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5").strip(),
        # Default to a model that is already present in this repo's Ollama setup.
        embedding_model=os.getenv("EMBEDDING_MODEL", "mxbai-embed-large").strip(),
        rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
        rag_min_score=float(os.getenv("RAG_MIN_SCORE", "0.25")),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip(),
        admin_password=os.getenv("ADMIN_PASSWORD", "lasersan2026").strip(),
    )

