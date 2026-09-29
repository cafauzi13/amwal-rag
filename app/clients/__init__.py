"""API clients (Jina, DeepSeek) and their shared exceptions."""
from __future__ import annotations

import time
from typing import Any


class APIRequestError(RuntimeError):
    """An API call failed (non-retryable status, or retries exhausted).

    status_code is the upstream HTTP status when there was one, so callers can log it
    without logging the message (which may quote the upstream response body).
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class APIKeyError(APIRequestError):
    """API key is missing or rejected (HTTP 401)."""


class ResponseFormatError(APIRequestError):
    """The API responded, but not in the expected shape. `raw` holds the payload."""

    def __init__(self, message: str, raw: Any = None) -> None:
        super().__init__(message)
        self.raw = raw


class DeadlineExceeded(RuntimeError):
    """The request's time budget ran out before an upstream call could finish."""


def time_left(deadline: float | None, cap: float) -> float:
    """Timeout for the next upstream call: min(cap, seconds until deadline).

    deadline is a time.monotonic() value (None = no deadline). Raises DeadlineExceeded
    when the budget is already spent.
    """
    if deadline is None:
        return cap
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DeadlineExceeded("anggaran waktu request habis")
    return min(cap, remaining)
