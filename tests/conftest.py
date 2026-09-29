"""Test fixtures: an app built from test settings, with the RAG pipeline mocked (no real API calls)."""
from __future__ import annotations

from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from app import rag
from app.config import get_settings

API_KEY = "test-key-123"
HEADERS = {"X-API-Key": API_KEY}
ANSWER = {
    "answer": "Jawaban uji.\n\nCatatan: jawaban ini bersifat informatif berdasarkan dokumen yang tersedia "
              "dan bukan pengganti fatwa resmi.",
    "sources": [{"sumber": "WKF-REG-02_UU-41-2004.pdf", "unit": "Pasal 10", "halaman": 5}],
}


@pytest.fixture(autouse=True)
def no_real_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test reaches the real pipeline."""

    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("real upstream call in a test")

    monkeypatch.setattr(rag, "load_index", boom)
    monkeypatch.setattr(rag, "answer", boom)
    monkeypatch.setattr(rag, "index_status", lambda: (True, None))


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., TestClient]]:
    """make_client(answer=fn, index_ready=bool, **env) -> TestClient with lifespan started."""
    clients: list[TestClient] = []

    def factory(answer: Callable[..., dict[str, Any]] | None = None, index_ready: bool = True,
                **env: str) -> TestClient:
        settings = {"API_KEY": API_KEY, "MAX_CONCURRENT_REQUESTS": "10", "QUEUE_WAIT_SECONDS": "5",
                    "REQUEST_TIMEOUT_SECONDS": "60", "ENABLE_DOCS": "false", **env}
        for key, value in settings.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()
        monkeypatch.setattr(rag, "answer", answer or (lambda question, deadline=None: ANSWER))
        monkeypatch.setattr(rag, "index_status",
                            lambda: (True, None) if index_ready else (False, "indeks perlu dibangun ulang"))
        from app.main import create_app

        client = TestClient(create_app(), raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)
    get_settings.cache_clear()
