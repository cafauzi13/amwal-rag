"""DeepSeek chat client via the OpenAI SDK (thinking mode disabled)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import openai
from openai import OpenAI

from app.clients import APIKeyError, APIRequestError, DeadlineExceeded, ResponseFormatError, time_left
from app.config import get_settings

logger = logging.getLogger(__name__)
_client: OpenAI | None = None


@dataclass(frozen=True)
class ChatResult:
    """Result of a single chat completion."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_content: str | None  # should be None/"" when thinking is disabled
    model: str


def _get_client() -> OpenAI:
    """Return a shared client; the SDK retries 429/5xx with exponential backoff."""
    global _client
    if _client is None:
        s = get_settings()
        if not s.deepseek_api_key:
            raise APIKeyError("DeepSeek: API key salah/kosong (isi DEEPSEEK_API_KEY di .env).")
        _client = OpenAI(
            api_key=s.deepseek_api_key,
            base_url=s.deepseek_base_url,
            timeout=s.http_timeout,
            max_retries=s.llm_max_retries,
        )
    return _client


def _create(client: OpenAI, system_prompt: str, user_prompt: str, temperature: float) -> Any:
    """One completion call, with SDK errors mapped to our exceptions."""
    s = get_settings()
    try:
        return client.chat.completions.create(
            model=s.deepseek_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}},
        )
    except openai.AuthenticationError as exc:
        raise APIKeyError("DeepSeek: API key salah/kosong (HTTP 401).", 401) from exc
    except openai.APIStatusError as exc:
        raise APIRequestError(f"DeepSeek HTTP {exc.status_code}: {exc.message}", exc.status_code) from exc
    except openai.APIConnectionError as exc:  # includes timeouts
        raise APIRequestError(f"DeepSeek: gagal terhubung ({type(exc).__name__}: {exc})") from exc


def _retryable(exc: APIRequestError) -> bool:
    """Network errors, 429 and 5xx are worth one more try; 4xx (bad request, auth) are not."""
    return exc.status_code is None or exc.status_code == 429 or exc.status_code >= 500


def chat(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_retries: int | None = None,
    deadline: float | None = None,
) -> ChatResult:
    """Send one system+user turn to DeepSeek and return the answer with token usage.

    max_retries overrides LLM_MAX_RETRIES for this call (the /chat path uses CHAT_MAX_RETRIES).
    With a deadline (time.monotonic()), retries run here instead of in the SDK so that every
    attempt's timeout is cut to the remaining budget; raises DeadlineExceeded when it runs out.
    """
    s = get_settings()
    temp = s.llm_temperature if temperature is None else temperature
    client = _get_client()
    if deadline is None:
        if max_retries is not None:
            client = client.with_options(max_retries=max_retries)
        resp = _create(client, system_prompt, user_prompt, temp)
    else:
        attempts = 1 + (s.llm_max_retries if max_retries is None else max_retries)
        for attempt in range(attempts):
            timeout = time_left(deadline, s.http_timeout)
            try:
                resp = _create(client.with_options(timeout=timeout, max_retries=0), system_prompt, user_prompt, temp)
                break
            except APIRequestError as exc:
                time_left(deadline, s.http_timeout)  # a timeout cut short by the deadline -> DeadlineExceeded
                if attempt == attempts - 1 or not _retryable(exc):
                    raise
                if s.chat_retry_delay >= time_left(deadline, float("inf")):
                    raise DeadlineExceeded("anggaran waktu habis sebelum retry DeepSeek") from exc
                logger.warning("DeepSeek %s (HTTP %s), retry in %.1fs",
                               type(exc).__name__, exc.status_code, s.chat_retry_delay)
                time.sleep(s.chat_retry_delay)

    if not resp.choices:
        raise ResponseFormatError("DeepSeek: respons tanpa choices", resp.model_dump())
    message = resp.choices[0].message
    usage = resp.usage
    return ChatResult(
        text=(message.content or "").strip(),
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        reasoning_content=getattr(message, "reasoning_content", None),
        model=resp.model,
    )
