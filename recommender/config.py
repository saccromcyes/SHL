"""Runtime configuration from environment variables (or .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Any OpenAI-compatible endpoint (Groq by default).
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"  # reranks and explains
    llm_fast_model: str = "llama-3.1-8b-instant"  # parses the request
    llm_timeout_s: float = 30.0

    embedder: str = "fastembed"  # or "hash" for tests/offline
    embed_model: str = "BAAI/bge-small-en-v1.5"
    model_cache_dir: str = str(ROOT / ".cache" / "models")

    catalog_path: str = str(ROOT / "data" / "shl_assessments.csv")
    candidates: int = 30  # how many go to the reranker
    max_results: int = 10
    rrf_k: int = 60

    max_query_chars: int = 6000
    rate_limit_per_10min: int = 40
    fetch_timeout_s: float = 8.0
    fetch_max_bytes: int = 2_000_000

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
