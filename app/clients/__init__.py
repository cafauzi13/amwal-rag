"""API clients (Jina, DeepSeek) and their shared exceptions."""
from __future__ import annotations

from typing import Any


class APIRequestError(RuntimeError):
    """An API call failed (non-retryable status, or retries exhausted)."""


class APIKeyError(APIRequestError):
    """API key is missing or rejected (HTTP 401)."""


class ResponseFormatError(APIRequestError):
    """The API responded, but not in the expected shape. `raw` holds the payload."""

    def __init__(self, message: str, raw: Any = None) -> None:
        super().__init__(message)
        self.raw = raw
