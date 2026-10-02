"""RAG pipeline: embed query -> FAISS top-k -> Jina rerank top-n -> DeepSeek.

Rules (see CLAUDE.md): one short retry per external call, rerank failure falls back
to FAISS order, answers are plain text with a code-appended disclaimer, and refusals
carry no sources.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np

from app.chunks import load_chunks, sha256
from app.clients.jina import chat_policy, embed, rerank
from app.clients.llm import chat
from app.config import CHUNKS_PATH, EMBED_DIM, INDEX_DIR, INDEX_FILE, INFO_FILE, META_FILE, get_settings

logger = logging.getLogger(__name__)

MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 500
UNIT_MAX_CHARS = 60

REFUSAL_NOT_FOUND = (
    "Maaf, informasi tidak ditemukan dalam dokumen wakaf, zakat, infak, sedekah, dan kurban yang tersedia."
)
REFUSAL_OFF_TOPIC = "Maaf, saya hanya dapat menjawab pertanyaan seputar wakaf, zakat, infak, sedekah, dan kurban."
DISCLAIMER = (
    "Catatan: jawaban ini bersifat informatif berdasarkan dokumen yang tersedia dan bukan pengganti fatwa resmi."
)

# Adapted from the thesis prompt (rag-waqf-chatbot src/pipeline.py): rules moved to the
# system message; rule 3 cites naturally instead of file/page (sources[] carries those);
# added an off-topic rule, plain-text rule, and "no disclaimer" (appended by code).
SYSTEM_PROMPT = f"""Anda adalah asisten AI ahli dalam bidang wakaf, zakat, infak, sedekah, dan kurban di Indonesia.
Tugas Anda adalah menjawab pertanyaan pengguna secara akurat, objektif, dan hanya berdasarkan konteks dokumen yang disediakan.

Aturan Jawaban:
1. Jawablah pertanyaan hanya berdasarkan fakta-fakta yang tertulis di dalam Konteks Dokumen. Baca dan pertimbangkan SELURUH dokumen yang diberikan sebelum menyimpulkan informasi tidak ada, termasuk mensintesis informasi dari beberapa dokumen sekaligus jika diperlukan.
2. Jangan menambahkan informasi di luar dokumen, spekulasi, atau interpretasi pribadi.
3. Sebutkan dasar rujukan secara alami di dalam kalimat bila tersedia, misalnya "Menurut UU No. 41 Tahun 2004 Pasal 10, ...". Jangan menulis nama berkas atau nomor halaman; daftar sumber ditampilkan terpisah oleh sistem.
4. Jika pertanyaan pengguna memiliki beberapa bagian, jawab SETIAP bagian yang informasinya tersedia dalam Konteks Dokumen, dan hanya nyatakan tidak ditemukan untuk bagian yang benar-benar tidak tercakup.
5. Jika SELURUH informasi untuk menjawab pertanyaan tidak terdapat dalam Konteks Dokumen yang diberikan, jawab persis seperti ini:
   "{REFUSAL_NOT_FOUND}"
6. Jika pertanyaan sama sekali tidak berkaitan dengan wakaf, zakat, infak, sedekah, atau kurban, jawab persis seperti ini:
   "{REFUSAL_OFF_TOPIC}"
7. Tuliskan jawaban Anda secara ringkas, profesional, dan dalam Bahasa Indonesia yang baik dan benar.
8. Gunakan TEKS POLOS tanpa Markdown: jangan memakai tanda bintang, garis bawah, tanda #, tabel, atau blok kode. Untuk daftar, tulis setiap butir di baris baru diawali "- " atau "1. ".
9. Jangan menulis disclaimer atau catatan penutup; sistem akan menambahkannya."""


class IndexNotReadyError(RuntimeError):
    """The index is missing or does not match data/chunks.jsonl; message says how to fix it."""


@dataclass(frozen=True)
class RagIndex:
    """FAISS index plus chunk texts and metadata, aligned by position."""

    index: faiss.Index
    texts: list[str]
    metas: list[dict[str, Any]]

    def search(self, query_vec: np.ndarray, k: int) -> list[int]:
        _, ids = self.index.search(query_vec.astype(np.float32), min(k, self.index.ntotal))
        return [int(i) for i in ids[0] if i >= 0]


def _load() -> RagIndex:
    """Load index/ and chunks.jsonl, verifying they belong together."""
    rebuild = "Jalankan: python -m scripts.build_index"
    paths = {name: INDEX_DIR / name for name in (INDEX_FILE, META_FILE, INFO_FILE)}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise IndexNotReadyError(f"Indeks belum dibangun ({', '.join(missing)} tidak ada). {rebuild}")

    info = json.loads(paths[INFO_FILE].read_text(encoding="utf-8"))
    if info.get("limit") is not None:
        raise IndexNotReadyError(f"index/ berisi build sampel (--limit {info['limit']}). {rebuild}")
    model = get_settings().jina_embed_model
    if info.get("model") != model:
        raise IndexNotReadyError(
            f"Indeks dibangun dengan model {info.get('model')!r}, tetapi JINA_EMBED_MODEL={model!r}. {rebuild}"
        )
    if sha256(CHUNKS_PATH) != info.get("chunks_sha256"):
        raise IndexNotReadyError(f"{CHUNKS_PATH.name} berubah sejak indeks dibangun; indeks perlu dibangun ulang. {rebuild}")

    texts, chunk_metas, _ = load_chunks(CHUNKS_PATH)
    with paths[META_FILE].open(encoding="utf-8") as f:
        metas = [json.loads(line) for line in f if line.strip()]
    index = faiss.read_index(str(paths[INDEX_FILE]))
    counts = {"vektor": index.ntotal, "meta": len(metas), "chunks": len(texts)}
    if len(set(counts.values())) != 1:
        raise IndexNotReadyError(f"Jumlah tidak sejajar ({counts}). {rebuild}")
    if index.d != EMBED_DIM:
        raise IndexNotReadyError(f"Dimensi indeks {index.d}, seharusnya {EMBED_DIM}. {rebuild}")
    if metas != chunk_metas:
        raise IndexNotReadyError(f"meta.jsonl tidak sama dengan metadata {CHUNKS_PATH.name}. {rebuild}")
    logger.info("Indeks termuat: %d vektor, model %s", index.ntotal, model)
    return RagIndex(index, texts, metas)


_index: RagIndex | None = None
_lock = threading.Lock()


def load_index() -> RagIndex:
    """Return the cached index, loading it on first use. Failures are not cached, so a rebuild is picked up.

    Any failure while loading (truncated faiss.index, bad JSON, unreadable file, ...) becomes
    IndexNotReadyError, so /health and /chat answer 503 instead of 500. Only the error class is
    logged; this path never sees a question.
    """
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                try:
                    _index = _load()
                except IndexNotReadyError:
                    raise
                except Exception as exc:  # noqa: BLE001 - index loading only, not the answer path
                    logger.error("Gagal memuat indeks (%s)", type(exc).__name__)
                    raise IndexNotReadyError(
                        f"Indeks rusak atau tidak terbaca ({type(exc).__name__}). "
                        "Jalankan: python -m scripts.build_index --fresh"
                    ) from exc
    return _index


def index_status() -> tuple[bool, str | None]:
    """(ready, reason) for /health; never calls external APIs."""
    try:
        load_index()
        return True, None
    except IndexNotReadyError as exc:
        return False, str(exc)


@dataclass(frozen=True)
class Retrieval:
    """Final chunk ids (best first), FAISS candidates before rerank, and whether rerank fell back."""

    ids: list[int]
    candidates: list[int]
    rerank_fallback: bool


def retrieve(question: str, deadline: float | None = None) -> Retrieval:
    """Embed -> FAISS top-k -> rerank top-n (fallback: FAISS top-n). No LLM call.

    deadline: time.monotonic() by which all upstream calls must finish (None = no limit).
    """
    s = get_settings()
    idx = load_index()
    policy = chat_policy(deadline)

    t0 = time.perf_counter()
    candidates = idx.search(embed(question, "retrieval.query", retry=policy), s.retrieve_top_k)
    t1 = time.perf_counter()
    n = min(s.rerank_top_n, len(candidates))
    try:
        ranked = rerank(question, [idx.texts[i] for i in candidates], top_n=n, retry=policy)
        ids = [candidates[j] for j, _ in ranked[:n]]
        if len(ids) != n or len(set(ids)) != n:
            raise ValueError(f"rerank mengembalikan {len(ids)} hasil unik, diharapkan {n}")
        fallback = False
    except Exception as exc:  # noqa: BLE001 - rerank must never fail /chat
        # class and HTTP status only: the message can quote the upstream body (and thus the question)
        logger.warning("Rerank gagal (%s, HTTP %s); memakai top-%d FAISS",
                       type(exc).__name__, getattr(exc, "status_code", None), n)
        ids, fallback = candidates[:n], True
    t2 = time.perf_counter()
    logger.info("retrieve: embed+faiss %.0f ms, rerank %.0f ms%s",
                (t1 - t0) * 1000, (t2 - t1) * 1000, " (fallback)" if fallback else "")
    return Retrieval(ids, candidates, fallback)


def _context(idx: RagIndex, ids: list[int]) -> str:
    """Numbered context blocks in the thesis format: Dokumen n [Sumber: ..., unit, Hal: ...]."""
    blocks = []
    for n, i in enumerate(ids, start=1):
        meta = idx.metas[i]
        unit = f", {meta['unit']}" if meta.get("unit") else ""
        header = f"[Sumber: {meta.get('sumber') or 'Tidak Diketahui'}{unit}, Hal: {meta.get('halaman') or '-'}]"
        blocks.append(f"Dokumen {n} {header}:\n{idx.texts[i].strip()}")
    return "\n\n".join(blocks)


_MARKDOWN = [
    (re.compile(r"\*\*(.+?)\*\*", re.S), r"\1"),
    (re.compile(r"__(.+?)__", re.S), r"\1"),
    (re.compile(r"`+"), ""),
    (re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*"), ""),
    (re.compile(r"(?m)^([ \t]*)[*•][ \t]+"), r"\1- "),
]


def _plain_text(text: str) -> str:
    """Strip Markdown the model may still emit, and any disclaimer line it wrote itself."""
    for pattern, repl in _MARKDOWN:
        text = pattern.sub(repl, text)
    lines = [ln for ln in text.splitlines() if "bukan pengganti fatwa" not in ln.lower()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("\"'“”").lower())


def _refusal(text: str) -> str | None:
    """The canonical refusal sentence if the answer starts with one, else None."""
    norm = _normalise(text)
    for sentence in (REFUSAL_NOT_FOUND, REFUSAL_OFF_TOPIC):
        if norm.startswith(_normalise(sentence).rstrip(".")):
            return sentence
    return None


def _shorten(text: str, limit: int = UNIT_MAX_CHARS) -> str:
    """Cut at a word boundary to about `limit` chars, adding "…" when cut."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit + 1]
    cut = cut.rsplit(" ", 1)[0] if " " in cut else text[:limit]
    return cut.rstrip(" ,;:-–") + "…"


def _sources(idx: RagIndex, ids: list[int], limit: int) -> list[dict[str, Any]]:
    """Up to `limit` sources, best first, deduplicated on sumber+halaman."""
    seen: set[tuple[Any, Any]] = set()
    out: list[dict[str, Any]] = []
    for i in ids:
        meta = idx.metas[i]
        halaman = meta.get("halaman") if isinstance(meta.get("halaman"), int) else None
        key = (meta.get("sumber"), halaman)
        if key in seen:
            continue
        seen.add(key)
        out.append({"sumber": meta.get("sumber"), "unit": _shorten(meta.get("unit") or ""), "halaman": halaman})
        if len(out) == limit:
            break
    return out


def answer(question: str, deadline: float | None = None) -> dict[str, Any]:
    """Answer one question (single-turn). Returns {"answer": str, "sources": [...]}.

    Raises ValueError for an invalid question, IndexNotReadyError if the index is not
    usable, app.clients.APIRequestError when Jina embed or DeepSeek is unavailable, and
    app.clients.DeadlineExceeded when the deadline (time.monotonic()) runs out.
    """
    q = question.strip()
    if not MIN_QUESTION_CHARS <= len(q) <= MAX_QUESTION_CHARS:
        raise ValueError(f"Pertanyaan harus {MIN_QUESTION_CHARS}–{MAX_QUESTION_CHARS} karakter.")

    s = get_settings()
    idx = load_index()
    found = retrieve(q, deadline)
    t0 = time.perf_counter()
    result = chat(
        SYSTEM_PROMPT,
        f"Konteks Dokumen:\n{_context(idx, found.ids)}\n\nPertanyaan: {q}",
        max_retries=s.chat_max_retries,
        deadline=deadline,
    )
    logger.info("llm: %.0f ms, token prompt=%d completion=%d",
                (time.perf_counter() - t0) * 1000, result.prompt_tokens, result.completion_tokens)

    text = _plain_text(result.text)
    refusal = _refusal(text)
    if refusal or not text:
        return {"answer": refusal or REFUSAL_NOT_FOUND, "sources": []}
    return {"answer": f"{text}\n\n{DISCLAIMER}", "sources": _sources(idx, found.ids, s.rerank_top_n)}
