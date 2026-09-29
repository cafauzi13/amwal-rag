"""Jina AI client: embeddings and reranking over HTTP (httpx)."""
from __future__ import annotations

import logging
import time
from typing import Any, Literal, Sequence

import httpx
import numpy as np

from app.clients import APIKeyError, APIRequestError, ResponseFormatError
from app.config import EMBED_DIM, get_settings

EMBED_URL = "https://api.jina.ai/v1/embeddings"
RERANK_URL = "https://api.jina.ai/v1/rerank"

Task = Literal["retrieval.query", "retrieval.passage"]

logger = logging.getLogger(__name__)
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    """Return a shared HTTP client (created lazily so a missing key fails at call time)."""
    global _client
    if _client is None:
        s = get_settings()
        if not s.jina_api_key:
            raise APIKeyError("Jina: API key salah/kosong (isi JINA_API_KEY di .env).")
        _client = httpx.Client(
            timeout=s.http_timeout,
            headers={
                "Authorization": f"Bearer {s.jina_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
    return _client


def _retry_after(resp: httpx.Response) -> float | None:
    """Seconds from the Retry-After header, if present and numeric."""
    value = resp.headers.get("Retry-After")
    try:
        return max(float(value), 0.0) if value else None
    except ValueError:
        return None


def _backoff(schedule: tuple[float, ...], attempt: int) -> float:
    """Delay for a given retry attempt; reuses the last value if retries outnumber the schedule."""
    return schedule[min(attempt, len(schedule) - 1)]


def _post(url: str, payload: dict[str, Any]) -> Any:
    """POST with retries: 429 honours Retry-After, 5xx/network errors use a short backoff."""
    s = get_settings()
    client = _get_client()
    attempt = 0
    while True:
        try:
            resp = client.post(url, json=payload)
        except httpx.TransportError as exc:  # timeouts, connection resets, DNS
            if attempt >= s.jina_max_retries:
                raise APIRequestError(
                    f"Jina: gagal terhubung setelah {attempt + 1} percobaan ({type(exc).__name__}: {exc})"
                ) from exc
            delay = _backoff(s.jina_server_backoff, attempt)
            logger.warning("Jina network error (%s), retry in %.1fs", type(exc).__name__, delay)
        else:
            status = resp.status_code
            if status == 401:
                raise APIKeyError("Jina: API key salah/kosong (HTTP 401).")
            if status != 429 and status < 500:
                if resp.is_error:
                    raise APIRequestError(f"Jina HTTP {status}: {resp.text[:500]}")
                return resp.json()
            if attempt >= s.jina_max_retries:
                raise APIRequestError(
                    f"Jina HTTP {status} setelah {attempt + 1} percobaan: {resp.text[:500]}"
                )
            if status == 429:
                delay = _retry_after(resp)
                if delay is None:
                    delay = _backoff(s.jina_rate_limit_backoff, attempt)
            else:
                delay = _backoff(s.jina_server_backoff, attempt)
            logger.warning("Jina HTTP %d, retry in %.1fs", status, delay)
        time.sleep(delay)
        attempt += 1


def _pacing_delay(data: Any, batch: Sequence[str], elapsed: float, target_tpm: int) -> float:
    """Pause needed before the next batch to stay under target_tpm (0 disables pacing)."""
    if target_tpm <= 0:
        return 0.0
    usage = data.get("usage") if isinstance(data, dict) else None
    tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    if not isinstance(tokens, (int, float)):
        tokens = sum(len(t) for t in batch) / 4  # rough estimate: ~4 chars per token
    return max(tokens / target_tpm * 60 - elapsed, 0.0)


def embed(
    texts: str | Sequence[str], task: Task, batch_size: int | None = None
) -> np.ndarray:
    """Embed texts in sequential batches; returns L2-normalised float32 array of shape (n, dim)."""
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)

    s = get_settings()
    size = batch_size or s.jina_embed_batch_size
    vectors: list[list[float]] = []
    pause = 0.0
    for start in range(0, len(texts), size):
        if pause > 0:
            logger.info("Jina pacing: jeda %.1fs (target %d TPM)", pause, s.jina_target_tpm)
            time.sleep(pause)
        batch = list(texts[start : start + size])
        began = time.perf_counter()
        data = _post(
            EMBED_URL,
            {"model": s.jina_embed_model, "task": task, "input": batch, "truncate": True},
        )
        pause = _pacing_delay(data, batch, time.perf_counter() - began, s.jina_target_tpm)
        try:
            items = sorted(data["data"], key=lambda d: d["index"])
            vectors.extend(item["embedding"] for item in items)
        except (KeyError, TypeError) as exc:
            raise ResponseFormatError(f"Jina embed: format respons tak dikenal ({exc!r})", data) from exc
        if len(items) != len(batch):
            raise ResponseFormatError(
                f"Jina embed: {len(batch)} input tapi {len(items)} embedding kembali", data
            )

    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def rerank(query: str, docs: Sequence[str], top_n: int | None = None) -> list[tuple[int, float]]:
    """Rerank docs against query; returns [(original_doc_index, score), ...] best first."""
    if not docs:
        return []
    s = get_settings()
    n = min(top_n or s.rerank_top_n, len(docs))
    data = _post(
        RERANK_URL,
        {
            "model": s.jina_rerank_model,
            "query": query,
            "documents": list(docs),
            "top_n": n,
            "return_documents": False,
        },
    )
    return _parse_rerank(data)


def _parse_rerank(data: Any) -> list[tuple[int, float]]:
    """Parse the rerank response defensively (field names not fully verified)."""
    results = data.get("results", data.get("data")) if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise ResponseFormatError("Jina rerank: tidak ada list 'results'/'data' di respons", data)

    parsed: list[tuple[int, float]] = []
    for item in results:
        if not isinstance(item, dict):
            raise ResponseFormatError("Jina rerank: item hasil bukan object", data)
        idx = item.get("index", item.get("document_index"))
        score = item.get("relevance_score", item.get("score"))
        if idx is None or score is None:
            raise ResponseFormatError(
                f"Jina rerank: field index/skor tidak ditemukan (keys: {sorted(item)})", data
            )
        parsed.append((int(idx), float(score)))
    parsed.sort(key=lambda pair: pair[1], reverse=True)
    return parsed
