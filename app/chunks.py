"""Reading data/chunks.jsonl, shared by build_index and the RAG pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PREFIX = "passage: "  # leftover from the thesis' E5 pipeline; Jina uses the task param


def load_chunks(path: Path, limit: int | None = None) -> tuple[list[str], list[dict[str, Any]], int]:
    """Read chunks; returns (texts without prefix, metadata without text, prefixed count).

    Raises ValueError on a row without a usable "text" field.
    """
    texts: list[str] = []
    metas: list[dict[str, Any]] = []
    prefixed = 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if limit is not None and len(texts) >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.pop("text", None)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path.name} baris {lineno}: field 'text' kosong/tidak ada")
            if text.startswith(PREFIX):
                text = text[len(PREFIX):]
                prefixed += 1
            texts.append(text)
            metas.append(row)
    return texts, metas, prefixed


def sha256(path: Path) -> str:
    """Hex digest of a file, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
