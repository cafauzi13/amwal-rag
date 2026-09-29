"""DeepSeek chat client via the OpenAI SDK (thinking mode disabled)."""
from __future__ import annotations

from dataclasses import dataclass

import openai
from openai import OpenAI

from app.clients import APIKeyError, APIRequestError, ResponseFormatError
from app.config import get_settings

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


def chat(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_retries: int | None = None,
) -> ChatResult:
    """Send one system+user turn to DeepSeek and return the answer with token usage.

    max_retries overrides LLM_MAX_RETRIES for this call (the /chat path uses CHAT_MAX_RETRIES).
    """
    s = get_settings()
    client = _get_client()
    if max_retries is not None:
        client = client.with_options(max_retries=max_retries)
    try:
        resp = client.chat.completions.create(
            model=s.deepseek_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=s.llm_temperature if temperature is None else temperature,
            extra_body={"thinking": {"type": "disabled"}},
        )
    except openai.AuthenticationError as exc:
        raise APIKeyError("DeepSeek: API key salah/kosong (HTTP 401).") from exc
    except openai.APIStatusError as exc:
        raise APIRequestError(f"DeepSeek HTTP {exc.status_code}: {exc.message}") from exc
    except openai.APIConnectionError as exc:  # includes timeouts
        raise APIRequestError(f"DeepSeek: gagal terhubung ({type(exc).__name__}: {exc})") from exc

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
