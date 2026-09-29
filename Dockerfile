# amwal-rag API image. Build/run with docker compose (see README / docker-compose.yml).
# The FAISS index is NOT baked in: it lives in the ./index volume and is built once with
#   docker compose run --rm api python -m scripts.build_index
FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Pinned dependencies first, so code changes do not reinstall them
COPY requirements.lock .
RUN pip install -r requirements.lock

# Default non-root user. docker-compose.yml overrides the uid/gid with the host's
# (APP_UID/APP_GID) so the bind-mounted ./index stays writable without chown.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent app

COPY app ./app
COPY scripts ./scripts
COPY data/chunks.jsonl ./data/chunks.jsonl
RUN mkdir -p /app/index && chown app:app /app/index

USER app
EXPOSE 8000

# /health only checks that the index is loaded (no external API calls); 503 -> unhealthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# One worker on purpose: MAX_CONCURRENT_REQUESTS is enforced per process (a thread pool
# handles concurrency). More workers would multiply the limit.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--workers", "1"]
