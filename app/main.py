"""FastAPI app: POST /chat and GET /health, per API_CONTRACT.md and the rules in CLAUDE.md.

Run locally:  uvicorn app.main:app --no-access-log   (set ENABLE_DOCS=true for /docs)
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StrictStr, StringConstraints
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import rag
from app.clients import APIRequestError, DeadlineExceeded
from app.config import get_settings

logger = logging.getLogger("app.api")

REQUEST_ID_RE = re.compile(r"[A-Za-z0-9-]{1,64}")
API_KEY_PLACEHOLDER = "ganti-dengan-secret-acak"  # value shipped in .env.example; never accepted
TIMEOUT_GRACE = 2.0  # backstop on top of REQUEST_TIMEOUT_SECONDS; upstream calls already honour the deadline

MSG_UNAUTHORIZED = "API key tidak valid."
MSG_INVALID = f"Pertanyaan harus berupa teks {rag.MIN_QUESTION_CHARS}–{rag.MAX_QUESTION_CHARS} karakter."
MSG_BUSY = "Server sedang sibuk. Silakan coba lagi beberapa saat lagi."
MSG_INTERNAL = "Terjadi kesalahan pada server."
MSG_NOT_CONFIGURED = "Server belum dikonfigurasi."
MSG_UNAVAILABLE = "Layanan sedang tidak tersedia. Silakan coba lagi beberapa saat lagi."
MSG_TIMEOUT = "Waktu pemrosesan habis. Silakan coba lagi."
HTTP_CODES = {404: ("not_found", "Endpoint tidak ditemukan."), 405: ("method_not_allowed", "Metode tidak diizinkan.")}


class ApiError(Exception):
    """An error response in the contract's uniform format."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(code)
        self.status, self.code, self.message = status, code, message


Question = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=rag.MIN_QUESTION_CHARS, max_length=rag.MAX_QUESTION_CHARS),
]


class ChatRequest(BaseModel):
    question: Question


class Source(BaseModel):
    sumber: str
    unit: str
    halaman: int | None


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[Source]


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


def error_response(request_id: str, status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"request_id": request_id, "error": {"code": code, "message": message}},
        headers={"X-Request-ID": request_id},
    )


def api_key_configured() -> bool:
    key = get_settings().api_key
    return bool(key) and key != API_KEY_PLACEHOLDER


def check_api_key(request: Request) -> None:
    """Raise ApiError unless X-API-Key matches API_KEY (constant-time compare)."""
    if not api_key_configured():
        logger.error("API_KEY belum diisi atau masih placeholder; /chat ditolak")
        raise ApiError(500, "internal_error", MSG_NOT_CONFIGURED)
    given = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(given.encode(), get_settings().api_key.encode()):
        raise ApiError(401, "invalid_api_key", MSG_UNAUTHORIZED)


async def acquire_slot(sem: asyncio.Semaphore, timeout: float) -> bool:
    """Wait up to `timeout` for a slot; never leaks one if the wait races with a release."""
    task = asyncio.ensure_future(sem.acquire())
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if task in done:
        return True
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return False
    return True  # acquired just as we gave up: keep it


def _configure_logging() -> None:
    app_logger = logging.getLogger("app")
    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        app_logger.addHandler(handler)
        app_logger.setLevel(logging.INFO)
        app_logger.propagate = False
    logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    s = get_settings()
    if not api_key_configured():
        logger.error("API_KEY belum diisi atau masih placeholder di .env: semua POST /chat akan dibalas 500 "
                     "sampai diisi dengan secret acak.")
    ready, reason = await run_in_threadpool(rag.index_status)  # load now so the first request is fast
    if ready:
        logger.info("Indeks siap")
    else:
        logger.error("Indeks tidak siap, /health = 503: %s", reason)
    app.state.slots = asyncio.Semaphore(s.max_concurrent_requests)
    app.state.executor = ThreadPoolExecutor(max_workers=s.max_concurrent_requests, thread_name_prefix="rag")
    try:
        yield
    finally:
        app.state.executor.shutdown(wait=False, cancel_futures=True)


def create_app() -> FastAPI:
    """Build the app from the current settings (tests call this after changing env)."""
    _configure_logging()
    s = get_settings()
    app = FastAPI(
        title="amwal-rag",
        version="1",
        lifespan=lifespan,
        docs_url="/docs" if s.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if s.enable_docs else None,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        """request_id, API key check for /chat, uniform 500s, and one log line per request."""
        given = request.headers.get("x-request-id", "")
        request_id = given if REQUEST_ID_RE.fullmatch(given) else str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            if request.url.path == "/chat" and request.method == "POST":
                check_api_key(request)  # before the body is read, so 401 wins over 422
            response = await call_next(request)
        except ApiError as exc:
            response = error_response(request_id, exc.status, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 - never leak details; log the class only
            logger.error("request_id=%s unhandled %s", request_id, type(exc).__name__)
            response = error_response(request_id, 500, "internal_error", MSG_INTERNAL)
        response.headers["X-Request-ID"] = request_id
        logger.info("request_id=%s %s %s status=%d %.0fms", request_id, request.method, request.url.path,
                    response.status_code, (time.perf_counter() - started) * 1000)
        return response

    @app.exception_handler(ApiError)
    async def on_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(_request_id(request), exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(_request_id(request), 422, "invalid_question", MSG_INVALID)

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code, message = HTTP_CODES.get(exc.status_code, ("http_error", str(exc.detail)))
        return error_response(_request_id(request), exc.status_code, code, message)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest, request: Request) -> dict[str, Any]:
        """Answer one question (single-turn). Requires header X-API-Key."""
        request_id = _request_id(request)
        settings = get_settings()
        state = request.app.state
        if not await acquire_slot(state.slots, settings.queue_wait_seconds):
            raise ApiError(429, "rate_limited", MSG_BUSY)

        deadline = time.monotonic() + settings.request_timeout_seconds
        try:
            future = asyncio.get_running_loop().run_in_executor(state.executor, rag.answer, body.question, deadline)
        except BaseException:
            state.slots.release()
            raise

        def release(f: asyncio.Future) -> None:
            # the slot is freed when the worker thread really finishes, not when we stop waiting
            state.slots.release()
            if not f.cancelled():
                f.exception()  # mark retrieved so an abandoned failure is not reported as unhandled

        future.add_done_callback(release)
        try:
            result = await asyncio.wait_for(asyncio.shield(future), settings.request_timeout_seconds + TIMEOUT_GRACE)
        except (asyncio.TimeoutError, DeadlineExceeded):
            logger.warning("request_id=%s timeout", request_id)
            raise ApiError(504, "timeout", MSG_TIMEOUT)
        except rag.IndexNotReadyError:
            logger.error("request_id=%s indeks tidak siap", request_id)
            raise ApiError(503, "upstream_unavailable", MSG_UNAVAILABLE)
        except APIRequestError as exc:
            logger.warning("request_id=%s upstream %s (HTTP %s)", request_id, type(exc).__name__, exc.status_code)
            raise ApiError(503, "upstream_unavailable", MSG_UNAVAILABLE)
        except ValueError:
            raise ApiError(422, "invalid_question", MSG_INVALID)
        return {"request_id": request_id, **result}

    @app.get("/health")
    def health(request: Request) -> JSONResponse:
        """Index loaded and consistent with data/chunks.jsonl? Never calls external APIs."""
        ready, _ = rag.index_status()
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"request_id": _request_id(request), "status": "ok" if ready else "unavailable"},
        )

    return app


app = create_app()
