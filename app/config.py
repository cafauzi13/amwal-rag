"""Central configuration: every tunable parameter is read from .env here."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "index"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
CLEANED_DIR = DATA_DIR / "cleaned"  # cleaned text per document, with [Hal. N] markers

# Files inside INDEX_DIR, written by scripts/build_index.py
INDEX_FILE = "faiss.index"
META_FILE = "meta.jsonl"
INFO_FILE = "build_info.json"

EMBED_DIM = 1024  # jina-embeddings-v5-text-small


def _floats(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated list like '5,15,30' into floats."""
    values = tuple(float(x) for x in raw.split(",") if x.strip())
    if not values:
        raise ValueError(f"Backoff schedule must not be empty: {raw!r}")
    return values


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # DeepSeek
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    llm_temperature: float
    llm_max_retries: int

    # Jina
    jina_api_key: str
    jina_embed_model: str
    jina_rerank_model: str
    jina_embed_batch_size: int
    jina_max_retries: int
    jina_rate_limit_backoff: tuple[float, ...]
    jina_server_backoff: tuple[float, ...]
    jina_target_tpm: int  # pacing between embed batches; 0 = off

    # /chat path: one short retry (build_index keeps the long Jina backoff above)
    chat_max_retries: int
    chat_retry_delay: float

    # Retrieval
    retrieve_top_k: int
    rerank_top_n: int

    # HTTP
    http_timeout: float

    # API server (app/main.py)
    api_key: str
    max_concurrent_requests: int
    queue_wait_seconds: float
    request_timeout_seconds: float
    enable_docs: bool


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from .env (real environment variables take precedence)."""
    load_dotenv(ROOT_DIR / ".env")
    env = os.getenv
    return Settings(
        deepseek_api_key=env("DEEPSEEK_API_KEY", "").strip(),
        deepseek_base_url=env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=env("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        llm_temperature=float(env("LLM_TEMPERATURE", "0")),
        llm_max_retries=int(env("LLM_MAX_RETRIES", "3")),
        jina_api_key=env("JINA_API_KEY", "").strip(),
        jina_embed_model=env("JINA_EMBED_MODEL", "jina-embeddings-v5-text-small"),
        jina_rerank_model=env("JINA_RERANK_MODEL", "jina-reranker-v3.5"),
        jina_embed_batch_size=int(env("JINA_EMBED_BATCH_SIZE", "32")),
        jina_max_retries=int(env("JINA_MAX_RETRIES", "3")),
        jina_rate_limit_backoff=_floats(env("JINA_RATE_LIMIT_BACKOFF", "5,15,30")),
        jina_server_backoff=_floats(env("JINA_SERVER_BACKOFF", "1,2,4")),
        jina_target_tpm=int(env("JINA_TARGET_TPM", "80000")),
        chat_max_retries=int(env("CHAT_MAX_RETRIES", "1")),
        chat_retry_delay=float(env("CHAT_RETRY_DELAY", "1.0")),
        retrieve_top_k=int(env("RETRIEVE_TOP_K", "10")),
        rerank_top_n=int(env("RERANK_TOP_N", "3")),
        http_timeout=float(env("HTTP_TIMEOUT", "30")),
        api_key=env("API_KEY", "").strip(),
        max_concurrent_requests=int(env("MAX_CONCURRENT_REQUESTS", "5")),
        queue_wait_seconds=float(env("QUEUE_WAIT_SECONDS", "5")),
        request_timeout_seconds=float(env("REQUEST_TIMEOUT_SECONDS", "60")),
        enable_docs=env("ENABLE_DOCS", "false").strip().lower() in ("1", "true", "yes"),
    )
