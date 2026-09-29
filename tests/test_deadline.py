"""Upstream calls honour the request deadline (fake HTTP clients, no network)."""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from app.clients import DeadlineExceeded, jina, llm
from app.clients.jina import chat_policy


class FakeHttp:
    """Stands in for httpx.Client: records timeouts, returns queued responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.timeouts: list[float] = []

    def post(self, url: str, json: Any, timeout: float) -> httpx.Response:
        self.timeouts.append(timeout)
        return self.responses.pop(0)


def ok_response() -> httpx.Response:
    return httpx.Response(200, json={"results": []}, request=httpx.Request("POST", "https://x"))


def error_response(status: int) -> httpx.Response:
    return httpx.Response(status, text="err", request=httpx.Request("POST", "https://x"))


@pytest.fixture
def fake_http(monkeypatch):
    def install(*responses):
        client = FakeHttp(*responses)
        monkeypatch.setattr(jina, "_client", client)
        return client
    return install


def test_jina_timeout_cut_to_remaining_budget(fake_http):
    client = fake_http(ok_response())
    jina._post("https://x", {}, chat_policy(time.monotonic() + 2))
    assert 0 < client.timeouts[0] <= 2


def test_jina_expired_deadline_makes_no_call(fake_http):
    client = fake_http(ok_response())
    with pytest.raises(DeadlineExceeded):
        jina._post("https://x", {}, chat_policy(time.monotonic() - 1))
    assert client.timeouts == []


def test_jina_no_retry_sleep_past_deadline(fake_http, monkeypatch):
    monkeypatch.setenv("CHAT_RETRY_DELAY", "1.0")
    jina.get_settings.cache_clear()
    client = fake_http(error_response(503), ok_response())
    started = time.monotonic()
    with pytest.raises(DeadlineExceeded):
        jina._post("https://x", {}, chat_policy(time.monotonic() + 0.3))  # 1 s retry delay does not fit
    assert time.monotonic() - started < 0.3 and len(client.timeouts) == 1
    jina.get_settings.cache_clear()


def test_build_policy_has_no_deadline(fake_http):
    client = fake_http(ok_response())
    jina._post("https://x", {}, jina.build_policy())
    assert client.timeouts == [jina.get_settings().http_timeout]


class FakeOpenAI:
    """Stands in for the OpenAI client: records with_options() and returns a canned completion."""

    def __init__(self) -> None:
        self.options: list[dict[str, Any]] = []
        message = SimpleNamespace(content="ok", reasoning_content=None)
        self.completion = SimpleNamespace(choices=[SimpleNamespace(message=message)],
                                          usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1), model="m")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: self.completion))

    def with_options(self, **kwargs: Any) -> "FakeOpenAI":
        self.options.append(kwargs)
        return self


def test_deepseek_timeout_cut_and_sdk_retries_off(monkeypatch):
    fake = FakeOpenAI()
    monkeypatch.setattr(llm, "_client", fake)
    llm.chat("s", "u", max_retries=1, deadline=time.monotonic() + 2)
    assert fake.options[0]["max_retries"] == 0
    assert 0 < fake.options[0]["timeout"] <= 2


def test_deepseek_expired_deadline_makes_no_call(monkeypatch):
    fake = FakeOpenAI()
    monkeypatch.setattr(llm, "_client", fake)
    with pytest.raises(DeadlineExceeded):
        llm.chat("s", "u", max_retries=1, deadline=time.monotonic() - 1)
    assert fake.options == []
