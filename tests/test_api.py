"""API contract tests for app/main.py (RAG pipeline mocked)."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import pytest

from app import main
from app.clients import APIRequestError, DeadlineExceeded
from app.rag import IndexNotReadyError
from tests.conftest import ANSWER, HEADERS


def assert_error(resp: Any, status: int, code: str) -> dict[str, Any]:
    """Uniform error format: {request_id, error: {code, message}} plus X-Request-ID header."""
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"request_id", "error"}
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    assert resp.headers["X-Request-ID"] == body["request_id"]
    return body


def is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------- success


def test_chat_ok(make_client):
    resp = make_client().post("/chat", json={"question": "Apa syarat nazhir?"}, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == ANSWER["answer"] and body["sources"] == ANSWER["sources"]
    assert body["request_id"] == resp.headers["X-Request-ID"]


def test_question_is_stripped_before_answer(make_client):
    seen = {}

    def fake(question, deadline=None):
        seen["q"] = question
        return ANSWER

    make_client(answer=fake).post("/chat", json={"question": "   Apa itu wakaf?  "}, headers=HEADERS)
    assert seen["q"] == "Apa itu wakaf?"


# ---------------------------------------------------------------- 401


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "salah"}, {"X-API-Key": ""}])
def test_401_missing_or_wrong_key(make_client, headers):
    resp = make_client().post("/chat", json={"question": "Apa itu wakaf?"}, headers=headers)
    assert_error(resp, 401, "invalid_api_key")


def test_401_wins_over_invalid_body(make_client):
    resp = make_client().post("/chat", content=b"{bukan json", headers={"Content-Type": "application/json"})
    assert_error(resp, 401, "invalid_api_key")


@pytest.mark.parametrize("key", ["", "ganti-dengan-secret-acak"])
def test_500_when_server_key_not_configured(make_client, key):
    resp = make_client(API_KEY=key).post("/chat", json={"question": "Apa itu wakaf?"}, headers={"X-API-Key": key})
    body = assert_error(resp, 500, "internal_error")
    assert "dikonfigurasi" in body["error"]["message"]


# ---------------------------------------------------------------- 422


@pytest.mark.parametrize(
    "kwargs",
    [
        {"json": {}},
        {"json": {"question": ""}},
        {"json": {"question": "  hi  "}},
        {"json": {"question": "x" * 501}},
        {"json": {"question": 123}},
        {"json": {"question": None}},
        {"json": ["Apa itu wakaf?"]},
        {"content": b"{bukan json", "headers": {"Content-Type": "application/json"}},
    ],
)
def test_422_invalid_question(make_client, kwargs):
    headers = {**HEADERS, **kwargs.pop("headers", {})}
    resp = make_client().post("/chat", headers=headers, **kwargs)
    assert_error(resp, 422, "invalid_question")


def test_boundaries_3_and_500_chars_are_ok(make_client):
    client = make_client()
    assert client.post("/chat", json={"question": "abc"}, headers=HEADERS).status_code == 200
    assert client.post("/chat", json={"question": "x" * 500}, headers=HEADERS).status_code == 200


# ---------------------------------------------------------------- request_id


@pytest.mark.parametrize("rid", ["abc-123", "A" * 64, "0f1c107-ok"])
def test_valid_request_id_is_echoed(make_client, rid):
    client = make_client()
    ok = client.post("/chat", json={"question": "Apa itu wakaf?"}, headers={**HEADERS, "X-Request-ID": rid})
    err = client.post("/chat", json={"question": "hi"}, headers={**HEADERS, "X-Request-ID": rid})
    health = client.get("/health", headers={"X-Request-ID": rid})
    for resp in (ok, err, health):
        assert resp.headers["X-Request-ID"] == rid and resp.json()["request_id"] == rid


@pytest.mark.parametrize("rid", ["A" * 65, "abc_123", "abc 123", "abc/../x", "ümlaut", ""])
def test_invalid_request_id_is_replaced(make_client, rid):
    client = make_client()
    headers = {**HEADERS, "X-Request-ID": rid} if rid.isascii() else {**HEADERS, "X-Request-ID": rid.encode()}
    for resp in (client.post("/chat", json={"question": "Apa itu wakaf?"}, headers=headers),
                 client.post("/chat", json={}, headers=headers)):
        new = resp.headers["X-Request-ID"]
        assert new != rid and is_uuid(new) and resp.json()["request_id"] == new


def test_request_id_generated_when_absent(make_client):
    resp = make_client().post("/chat", json={"question": "Apa itu wakaf?"}, headers=HEADERS)
    assert is_uuid(resp.json()["request_id"])


# ---------------------------------------------------------------- /health


def test_health_ok(make_client):
    resp = make_client().get("/health")
    assert resp.status_code == 200 and resp.json()["status"] == "ok"
    assert resp.json()["request_id"] == resp.headers["X-Request-ID"]


def test_health_503_when_index_not_ready(make_client):
    called = []
    client = make_client(index_ready=False, answer=lambda *a, **k: called.append(1) or ANSWER)
    resp = client.get("/health")
    assert resp.status_code == 503 and resp.json()["status"] == "unavailable"
    assert "request_id" in resp.json() and not called


def test_health_needs_no_api_key(make_client):
    assert make_client(API_KEY="").get("/health").status_code == 200


# ---------------------------------------------------------------- 429


def test_429_when_all_slots_busy_then_recovers(make_client):
    entered, release = threading.Event(), threading.Event()

    def slow(question, deadline=None):
        entered.set()
        release.wait(5)
        return ANSWER

    client = make_client(answer=slow, MAX_CONCURRENT_REQUESTS="1", QUEUE_WAIT_SECONDS="0.2")
    first: dict[str, Any] = {}
    worker = threading.Thread(
        target=lambda: first.setdefault("resp", client.post("/chat", json={"question": "satu"}, headers=HEADERS)))
    worker.start()
    assert entered.wait(5)

    busy = client.post("/chat", json={"question": "dua"}, headers=HEADERS)
    assert_error(busy, 429, "rate_limited")

    release.set()
    worker.join(5)
    assert first["resp"].status_code == 200
    assert client.post("/chat", json={"question": "tiga"}, headers=HEADERS).status_code == 200


def test_queued_request_gets_slot_within_wait(make_client):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow(question, deadline=None):
        calls.append(question)
        if len(calls) == 1:
            entered.set()
            release.wait(5)
        return ANSWER

    client = make_client(answer=slow, MAX_CONCURRENT_REQUESTS="1", QUEUE_WAIT_SECONDS="3")
    worker = threading.Thread(target=lambda: client.post("/chat", json={"question": "satu"}, headers=HEADERS))
    worker.start()
    assert entered.wait(5)
    threading.Timer(0.3, release.set).start()  # frees the slot while the second request is queued
    assert client.post("/chat", json={"question": "dua"}, headers=HEADERS).status_code == 200
    worker.join(5)


# ---------------------------------------------------------------- 503 / 504 / 500


@pytest.mark.parametrize("exc", [APIRequestError("Jina HTTP 503", 503), IndexNotReadyError("rebuild")])
def test_503_upstream_or_index(make_client, exc):
    def fail(question, deadline=None):
        raise exc

    resp = make_client(answer=fail).post("/chat", json={"question": "Apa itu wakaf?"}, headers=HEADERS)
    body = assert_error(resp, 503, "upstream_unavailable")
    assert "rebuild" not in body["error"]["message"] and "Jina" not in body["error"]["message"]


def test_504_when_deadline_exceeded(make_client):
    def fail(question, deadline=None):
        raise DeadlineExceeded("habis")

    resp = make_client(answer=fail).post("/chat", json={"question": "Apa itu wakaf?"}, headers=HEADERS)
    assert_error(resp, 504, "timeout")


def test_deadline_passed_to_answer(make_client):
    seen = {}

    def fake(question, deadline=None):
        seen["remaining"] = deadline - time.monotonic()
        return ANSWER

    make_client(answer=fake, REQUEST_TIMEOUT_SECONDS="7").post("/chat", json={"question": "abc"}, headers=HEADERS)
    assert 6 < seen["remaining"] <= 7


def test_504_backstop_keeps_slot_until_thread_finishes(make_client, monkeypatch):
    monkeypatch.setattr(main, "TIMEOUT_GRACE", 0.0)
    release = threading.Event()

    def stuck(question, deadline=None):  # ignores the deadline on purpose
        release.wait(5)
        return ANSWER

    client = make_client(answer=stuck, MAX_CONCURRENT_REQUESTS="1", QUEUE_WAIT_SECONDS="0.1",
                         REQUEST_TIMEOUT_SECONDS="0.2")
    assert_error(client.post("/chat", json={"question": "abc"}, headers=HEADERS), 504, "timeout")
    # the worker thread is still running, so its slot must still be taken
    assert_error(client.post("/chat", json={"question": "abc"}, headers=HEADERS), 429, "rate_limited")
    release.set()
    time.sleep(0.2)
    monkeypatch.setattr(main.rag, "answer", lambda question, deadline=None: ANSWER)
    assert client.post("/chat", json={"question": "abc"}, headers=HEADERS).status_code == 200


def test_500_on_unexpected_error_hides_details(make_client):
    def fail(question, deadline=None):
        raise RuntimeError("rahasia internal")

    resp = make_client(answer=fail).post("/chat", json={"question": "Apa itu wakaf?"}, headers=HEADERS)
    body = assert_error(resp, 500, "internal_error")
    assert "rahasia" not in resp.text


# ---------------------------------------------------------------- 404 / 405 / docs / logs


def test_404_and_405_uniform(make_client):
    client = make_client()
    assert_error(client.get("/tidak-ada"), 404, "not_found")
    assert_error(client.get("/chat"), 405, "method_not_allowed")


def test_docs_disabled_by_default(make_client):
    client = make_client()
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_docs_enabled_with_env(make_client):
    client = make_client(ENABLE_DOCS="true")
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_logs_have_no_question_answer_or_key(make_client, caplog):
    secret_question = "Pertanyaan-rahasia-XYZ tentang wakaf"
    client = make_client()
    logger = main.logging.getLogger("app")
    logger.propagate = True  # let caplog see app.* records
    try:
        with caplog.at_level("INFO"):
            client.post("/chat", json={"question": secret_question}, headers={**HEADERS, "X-Request-ID": "log-1"})
            client.post("/chat", json={"question": secret_question}, headers={"X-API-Key": "salah"})
    finally:
        logger.propagate = False
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "request_id=log-1" in text and "status=200" in text and "status=401" in text
    for leaked in (secret_question, "Jawaban uji", HEADERS["X-API-Key"], "salah"):
        assert leaked not in text
