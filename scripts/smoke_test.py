"""Smoke test for Jina embed, Jina rerank and DeepSeek chat.

Run from the repo root:  python -m scripts.smoke_test
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable

import numpy as np

from app.clients import ResponseFormatError
from app.clients.jina import embed, rerank
from app.clients.llm import chat
from app.config import EMBED_DIM

QUERY = "Apa syarat nazhir wakaf uang?"
PASSAGES = {
    "a": "Nazhir wakaf uang wajib terdaftar di Badan Wakaf Indonesia dan dikelola oleh LKS-PWU yang ditunjuk Menteri.",
    "b": "Zakat fitrah dibayarkan sebelum salat Idulfitri sebesar satu sha' makanan pokok.",
    "c": "Hewan kurban harus memenuhi syarat umur dan tidak cacat.",
}
IRRELEVANT = ("x", "Resep rendang daging sapi khas Padang.")

SYSTEM_PROMPT = "Kamu asisten ZISWAF. Jawab singkat dalam Bahasa Indonesia."
USER_PROMPT = "Jelaskan apa itu wakaf produktif dalam dua kalimat."

TestFn = Callable[[], tuple[bool, str]]


def test_embedding() -> tuple[bool, str]:
    """Check dimension, norm, and that passage (a) is closest to the query."""
    q = embed(QUERY, "retrieval.query")
    p = embed(list(PASSAGES.values()), "retrieval.passage")
    print(f"  shape query={q.shape}, passages={p.shape}")

    dims_ok = q.shape == (1, EMBED_DIM) and p.shape == (len(PASSAGES), EMBED_DIM)
    norms = np.linalg.norm(np.vstack([q, p]), axis=1)
    norms_ok = bool(np.allclose(norms, 1.0, atol=1e-3))
    print(f"  norms min={norms.min():.4f} max={norms.max():.4f}")

    sims = p @ q[0]
    for label, sim in zip(PASSAGES, sims):
        print(f"  cos(query, {label}) = {sim:.4f}")
    best = list(PASSAGES)[int(np.argmax(sims))]

    ok = dims_ok and norms_ok and best == "a"
    note = f"dim_ok={dims_ok}, norm_ok={norms_ok}, terdekat=({best})"
    return ok, note


def test_rerank() -> tuple[bool, str]:
    """Check that passage (a) is ranked first among 4 documents."""
    labels = list(PASSAGES) + [IRRELEVANT[0]]
    docs = list(PASSAGES.values()) + [IRRELEVANT[1]]
    results = rerank(QUERY, docs, top_n=len(docs))
    for rank, (idx, score) in enumerate(results, start=1):
        print(f"  #{rank} ({labels[idx]}) score={score:.4f}  {docs[idx][:60]}")
    top = labels[results[0][0]] if results else None
    return top == "a", f"peringkat 1 = ({top})"


def test_deepseek() -> tuple[bool, str]:
    """Check that the answer is non-empty and thinking mode is off."""
    result = chat(SYSTEM_PROMPT, USER_PROMPT)
    print(f"  model: {result.model}")
    print(f"  jawaban: {result.text}")
    print(f"  token: prompt={result.prompt_tokens}, completion={result.completion_tokens}")
    has_reasoning = bool(result.reasoning_content)
    if has_reasoning:
        print(f"  reasoning_content (seharusnya kosong): {result.reasoning_content[:200]}...")
    ok = bool(result.text) and not has_reasoning
    return ok, f"jawaban_kosong={not result.text}, thinking_aktif={has_reasoning}"


def _describe_raw(raw: Any) -> None:
    """Print the shape of an unexpected API payload to help fix the parser."""
    if isinstance(raw, dict):
        print(f"  keys respons mentah: {sorted(raw)}")
        for key in ("results", "data"):
            if isinstance(raw.get(key), list) and raw[key] and isinstance(raw[key][0], dict):
                print(f"  keys item pertama '{key}': {sorted(raw[key][0])}")
    print(f"  respons mentah (potongan): {json.dumps(raw, ensure_ascii=False, default=str)[:800]}")


def run(name: str, fn: TestFn) -> tuple[str, bool, float, str]:
    """Run one test, never letting its failure stop the others."""
    print(f"\n=== {name} ===")
    start = time.perf_counter()
    try:
        ok, note = fn()
    except ResponseFormatError as exc:
        ok, note = False, f"{type(exc).__name__}: {exc}"
        _describe_raw(exc.raw)
    except Exception as exc:  # noqa: BLE001 - smoke test must report, not crash
        ok, note = False, f"{type(exc).__name__}: {exc}"
    ms = (time.perf_counter() - start) * 1000
    print(f"[{'PASS' if ok else 'FAIL'}] {name} | {ms:.0f} ms | {note}")
    return name, ok, ms, note


def main() -> int:
    """Run all tests, print a summary, return a process exit code."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # avoid crashes on non-UTF-8 consoles

    tests: list[tuple[str, TestFn]] = [
        ("Tes 1 - Embedding (Jina)", test_embedding),
        ("Tes 2 - Rerank (Jina)", test_rerank),
        ("Tes 3 - Chat (DeepSeek)", test_deepseek),
    ]
    results = [run(name, fn) for name, fn in tests]

    print("\n=== RINGKASAN ===")
    for name, ok, ms, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {ms:>7.0f} ms  {name}")
    passed = sum(ok for _, ok, _, _ in results)
    print(f"\n{passed}/{len(results)} tes lulus.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
