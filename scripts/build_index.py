"""Build the FAISS index from data/chunks.jsonl.

Embeds every chunk with Jina (task retrieval.passage), checkpointing each batch to disk
so an interrupted run resumes without re-spending tokens, then writes to index/:
  faiss.index      IndexFlatIP over L2-normalised vectors (inner product = cosine)
  meta.jsonl       one line per vector, same order: every chunk field except "text"
  build_info.json  model, dim, count, data hash, tokens, build time

Run from the repo root:
  python -m scripts.build_index             # full build -> index/
  python -m scripts.build_index --limit 50  # sample     -> index/sample/ (index/ untouched)
  python -m scripts.build_index --fresh     # discard checkpoint and start over
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.chunks import PREFIX, load_chunks, sha256
from app.clients.jina import iter_embed_batches
from app.config import CHUNKS_PATH, EMBED_DIM, INDEX_DIR, INDEX_FILE, INFO_FILE, META_FILE, get_settings

TASK = "retrieval.passage"


class BuildError(RuntimeError):
    """Build cannot continue; message is shown to the user."""


def write_atomic(path: Path, write) -> None:
    """Write via a temp file then rename, so a crash never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.replace(path)


def write_json(path: Path, data: Any) -> None:
    write_atomic(path, lambda p: p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"))


class Checkpoint:
    """Per-batch .npy files plus a manifest that pins the data/model/batching they belong to."""

    def __init__(self, directory: Path, identity: dict[str, Any], fresh: bool) -> None:
        self.dir = directory
        self.manifest_path = directory / "manifest.json"
        if fresh and directory.exists():
            shutil.rmtree(directory)
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            stale = {k: (self.manifest.get(k), v) for k, v in identity.items() if self.manifest.get(k) != v}
            if stale:
                detail = ", ".join(f"{k}: {old!r} -> {new!r}" for k, (old, new) in stale.items())
                raise BuildError(
                    f"Checkpoint di {directory} dibuat untuk data/setting lain ({detail}). "
                    "Jalankan ulang dengan --fresh untuk membuangnya."
                )
        else:
            directory.mkdir(parents=True, exist_ok=True)
            self.manifest = {**identity, "batches": {}}
            write_json(self.manifest_path, self.manifest)

    def _file(self, i: int) -> Path:
        return self.dir / f"batch_{i:05d}.npy"

    def has(self, i: int) -> bool:
        return str(i) in self.manifest["batches"] and self._file(i).exists()

    def load(self, i: int) -> np.ndarray:
        return np.load(self._file(i))

    def save(self, i: int, vectors: np.ndarray, tokens: int, exact: bool) -> None:
        def write(p: Path) -> None:
            with p.open("wb") as f:  # file handle: np.save(path) would append ".npy" to the .tmp name
                np.save(f, vectors)

        write_atomic(self._file(i), write)
        self.manifest["batches"][str(i)] = {"rows": len(vectors), "tokens": tokens, "exact": exact}
        write_json(self.manifest_path, self.manifest)

    def tokens(self) -> tuple[int, bool]:
        entries = self.manifest["batches"].values()
        return sum(e["tokens"] for e in entries), all(e["exact"] for e in entries)

    def remove(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def build(limit: int | None, fresh: bool) -> int:
    started = time.perf_counter()
    s = get_settings()
    out_dir = INDEX_DIR / "sample" if limit is not None else INDEX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        texts, metas, prefixed = load_chunks(CHUNKS_PATH, limit)
    except ValueError as exc:
        raise BuildError(str(exc)) from exc
    n = len(texts)
    if n == 0:
        raise BuildError(f"Tidak ada chunk di {CHUNKS_PATH}")
    size = s.jina_embed_batch_size
    ranges = [(start, min(start + size, n)) for start in range(0, n, size)]
    print(f"Chunks   : {n} dari {CHUNKS_PATH.name}" + (f" (--limit {limit})" if limit is not None else ""))
    print(f"Prefiks  : {prefixed} baris berawalan {PREFIX!r} (dibuang sebelum embed)")
    print(f"Batch    : {len(ranges)} x {size}, model {s.jina_embed_model}, "
          f"pacing {s.jina_target_tpm or 'mati'} TPM")
    print(f"Output   : {out_dir}")

    ckpt = Checkpoint(
        out_dir / ".checkpoint",
        {
            "chunks_sha256": sha256(CHUNKS_PATH),
            "n_chunks": n,
            "model": s.jina_embed_model,
            "batch_size": size,
            "task": TASK,
        },
        fresh,
    )
    todo = [i for i in range(len(ranges)) if not ckpt.has(i)]
    resumed = len(ranges) - len(todo)
    if resumed:
        print(f"Resume   : {resumed} batch diambil dari checkpoint, {len(todo)} tersisa")

    try:
        batches = (texts[ranges[i][0] : ranges[i][1]] for i in todo)
        for i, (vectors, tokens, exact) in zip(todo, iter_embed_batches(batches, TASK)):
            ckpt.save(i, vectors, tokens, exact)
            print(f"  batch {i + 1}/{len(ranges)}: {len(vectors)} vektor, {tokens} token")
    except KeyboardInterrupt:
        print(f"\nDihentikan. {len(ckpt.manifest['batches'])}/{len(ranges)} batch tersimpan di checkpoint; "
              "jalankan perintah yang sama untuk melanjutkan.")
        return 130

    vectors = np.vstack([ckpt.load(i) for i in range(len(ranges))]).astype(np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Validate before writing anything final
    norms = np.linalg.norm(vectors, axis=1)
    checks = {
        f"jumlah vektor == jumlah chunk ({index.ntotal} == {n})": index.ntotal == n,
        f"jumlah metadata == jumlah vektor ({len(metas)} == {index.ntotal})": len(metas) == index.ntotal,
        f"dimensi == {EMBED_DIM} ({index.d})": index.d == EMBED_DIM,
        f"norm vektor ~ 1 (min {norms.min():.4f}, max {norms.max():.4f})": bool(np.allclose(norms, 1.0, atol=1e-3)),
    }
    print("\nValidasi:")
    for label, ok in checks.items():
        print(f"  [{'OK' if ok else 'GAGAL'}] {label}")
    if not all(checks.values()):
        print("Build dibatalkan; index lama (jika ada) tidak diubah. Checkpoint dipertahankan.")
        return 1

    tokens, exact = ckpt.tokens()
    duration = time.perf_counter() - started
    write_atomic(out_dir / INDEX_FILE, lambda p: faiss.write_index(index, str(p)))
    write_atomic(
        out_dir / META_FILE,
        lambda p: p.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metas), encoding="utf-8"),
    )
    write_json(out_dir / INFO_FILE, {
        "model": s.jina_embed_model,
        "task": TASK,
        "dim": index.d,
        "count": index.ntotal,
        "limit": limit,
        "chunks_sha256": ckpt.manifest["chunks_sha256"],
        "prefix_stripped": prefixed,
        "total_tokens": tokens,
        "tokens_exact": exact,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    ckpt.remove()

    print("\nRingkasan:")
    print(f"  Vektor : {index.ntotal}")
    print(f"  Dimensi: {index.d}")
    print(f"  Durasi : {duration:.1f} s" + (f" ({resumed} batch dari checkpoint)" if resumed else ""))
    print(f"  Token  : {tokens}" + ("" if exact else " (estimasi, sebagian dari jumlah karakter / 4)"))
    print(f"  Disimpan ke {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FAISS index dari data/chunks.jsonl")
    parser.add_argument("--limit", type=int, help="hanya N chunk pertama; output ke index/sample/")
    parser.add_argument("--fresh", action="store_true", help="buang checkpoint dan mulai dari awal")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit harus > 0")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        return build(args.limit, args.fresh)
    except BuildError as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - keep checkpoint, tell the user how to resume
        print(f"\nERROR: {type(exc).__name__}: {exc}\n"
              "Batch yang sudah selesai tersimpan di checkpoint; jalankan perintah yang sama untuk melanjutkan.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
