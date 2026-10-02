"""A broken index on disk means "not ready" (503), never 500; fixing the files recovers without restart.

Uses a tiny real index in a temp dir and the real load_index()/answer(); answer() fails at
load_index() before any upstream call, so no network is used.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import rag
from app.chunks import sha256
from app.config import EMBED_DIM, INDEX_FILE, INFO_FILE, META_FILE, get_settings
from tests.conftest import API_KEY, HEADERS

# captured at import, before the autouse fixture in conftest replaces them with stubs
REAL_LOAD_INDEX = rag.load_index
REAL_INDEX_STATUS = rag.index_status
REAL_ANSWER = rag.answer

QUESTION = "Pertanyaan-rahasia-XYZ tentang wakaf"


def write_index(tmp: Path) -> dict[str, Path]:
    """Three chunks, a matching FAISS index, meta.jsonl and build_info.json."""
    chunks = tmp / "chunks.jsonl"
    rows = [{"text": f"passage: teks {i}", "kategori": "wakaf", "jenis": "regulasi",
             "sumber": f"WKF-REG-0{i}_X.pdf", "unit": f"Pasal {i}", "halaman": i} for i in range(1, 4)]
    chunks.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    index_dir = tmp / "index"
    index_dir.mkdir()
    vectors = np.random.default_rng(0).random((len(rows), EMBED_DIM), dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    faiss.write_index(index, str(index_dir / INDEX_FILE))
    metas = [{k: v for k, v in r.items() if k != "text"} for r in rows]
    (index_dir / META_FILE).write_text("".join(json.dumps(m) + "\n" for m in metas), encoding="utf-8")
    (index_dir / INFO_FILE).write_text(json.dumps({
        "model": get_settings().jina_embed_model, "limit": None, "chunks_sha256": sha256(chunks),
    }), encoding="utf-8")
    return {"chunks": chunks, "dir": index_dir}


@pytest.fixture
def real_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setenv("API_KEY", API_KEY)
    get_settings.cache_clear()
    paths = write_index(tmp_path)
    monkeypatch.setattr(rag, "CHUNKS_PATH", paths["chunks"])
    monkeypatch.setattr(rag, "INDEX_DIR", paths["dir"])
    monkeypatch.setattr(rag, "load_index", REAL_LOAD_INDEX)
    monkeypatch.setattr(rag, "index_status", REAL_INDEX_STATUS)
    monkeypatch.setattr(rag, "answer", REAL_ANSWER)
    monkeypatch.setattr(rag, "_index", None)
    yield paths
    get_settings.cache_clear()


@pytest.fixture
def client(real_index: dict[str, Path]) -> Any:
    from app.main import create_app

    with TestClient(create_app(), raise_server_exceptions=False) as c:
        yield c


def chat(client: Any) -> Any:
    return client.post("/chat", json={"question": QUESTION}, headers=HEADERS)


def truncate_faiss(paths: dict[str, Path]) -> bytes:
    path = paths["dir"] / INDEX_FILE
    good = path.read_bytes()
    path.write_bytes(good[: len(good) // 2])
    return good


def corrupt_meta(paths: dict[str, Path]) -> bytes:
    path = paths["dir"] / META_FILE
    good = path.read_bytes()
    lines = good.decode().splitlines(keepends=True)
    path.write_text(lines[0] + '{"sumber": "terpotong\n' + "".join(lines[2:]), encoding="utf-8")
    return good


def test_good_index_is_ready(client):
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("breaker,filename", [(truncate_faiss, INDEX_FILE), (corrupt_meta, META_FILE)])
def test_broken_index_is_503_then_recovers_without_restart(real_index, breaker, filename, caplog):
    good = breaker(real_index)
    from app.main import create_app

    logger = logging.getLogger("app")
    logger.propagate = True  # let caplog see app.* records
    try:
        with caplog.at_level("INFO"), TestClient(create_app(), raise_server_exceptions=False) as client:
            health = client.get("/health")
            assert health.status_code == 503 and health.json()["status"] == "unavailable"

            resp = chat(client)
            assert resp.status_code == 503
            body = resp.json()
            assert body["error"]["code"] == "upstream_unavailable"
            # generic message for the client: no class names, file names or paths
            for detail in ("Error", "faiss", "meta", "json", ".index", "build_index"):
                assert detail not in body["error"]["message"]

            (real_index["dir"] / filename).write_bytes(good)  # fix the file, no restart
            assert client.get("/health").status_code == 200
            assert rag._index is not None
    finally:
        logger.propagate = False

    log = "\n".join(r.getMessage() for r in caplog.records)
    assert "Gagal memuat indeks (" in log  # error class goes to the log
    assert QUESTION not in log


def test_answer_path_errors_are_not_swallowed(real_index, monkeypatch):
    """Only loading is converted to "not ready": an error after loading still surfaces as-is."""
    def broken_retrieve(question, deadline=None):
        raise RuntimeError("bug di jalur jawaban")

    monkeypatch.setattr(rag, "retrieve", broken_retrieve)
    with pytest.raises(RuntimeError, match="jalur jawaban"):
        rag.answer("Apa itu wakaf?")
