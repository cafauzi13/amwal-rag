"""Retrieval regression test against the ground-truth draft (no LLM), plus an off-topic check.

Run from the repo root:
  python -m scripts.regression_test                  # retrieve() on status=utama rows -> data/eval/regression_deploy.json
  python -m scripts.regression_test --compare [FILE] # also score FILE (default: data/eval/regression_skripsi.json)
  python -m scripts.regression_test --offtopic       # answer() on 5 off-topic questions

Result format and scoring rules: data/eval/SCHEMA.md
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.chunks import sha256
from app.config import DATA_DIR, INDEX_DIR, INFO_FILE, get_settings
from app.rag import REFUSAL_NOT_FOUND, REFUSAL_OFF_TOPIC, answer, load_index, retrieve

EVAL_DIR = DATA_DIR / "eval"
GT_PATH = EVAL_DIR / "ground_truth_draf_v2.csv"
OUT_PATH = EVAL_DIR / "regression_deploy.json"
SKRIPSI_PATH = EVAL_DIR / "regression_skripsi.json"
SCHEMA_VERSION = 1

OFFTOPIC_QUESTIONS = [
    "Bagaimana resep rendang daging sapi yang empuk?",
    "Siapa juara Liga Champions tahun lalu?",
    "Bagaimana cara menginstal Python di Windows?",
    "Berapa harga saham BBCA hari ini?",
    "Rekomendasikan film horor Indonesia yang seru.",
]

CODE_RE = re.compile(r"^[A-Z]{3}-(?:REG|LIT)-\d+$")
PAGE_RE = re.compile(r"\bhalaman\s+(\d+)(?:\s*-\s*(\d+))?", re.I)
PASAL_RE = re.compile(r"\bpasal\s+(\d+[a-z]?)\b", re.I)
LEVELS = ("src", "hal", "psl")
LEVEL_NAMES = {"src": "sumber", "hal": "halaman (±1)", "psl": "Pasal"}


def doc_code(sumber: str) -> str:
    """Document code from a chunk's file name: text before the first underscore."""
    return sumber.split("_", 1)[0]


@dataclass(frozen=True)
class Loc:
    """One parsed part of lokasi_sumber."""

    code: str
    page_from: int | None
    page_to: int | None
    pasal: str | None  # lower-case, e.g. "10" or "49a"; REG documents only


def parse_loc(part: str) -> Loc:
    code, _, rest = part.partition(",")
    code = code.strip()
    page = PAGE_RE.search(rest)
    page_from = int(page.group(1)) if page else None
    page_to = int(page.group(2)) if page and page.group(2) else page_from
    pasal = PASAL_RE.search(rest) if "-REG-" in code else None
    return Loc(code, page_from, page_to, pasal.group(1).lower() if pasal else None)


def split_lokasi(lokasi: str) -> list[str]:
    return [p.strip() for p in lokasi.split("||") if p.strip()]


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def load_ground_truth(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """(utama rows, cadangan rows) in CSV order."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["status"] == "utama"], [r for r in rows if r["status"] == "cadangan"]


def corpus_codes() -> set[str]:
    return {doc_code(m.get("sumber") or "") for m in load_index().metas}


# ---------------------------------------------------------------- scoring


def _pasal_match(unit: str, pasal: str) -> bool:
    norm = " ".join((unit or "").lower().split())
    return re.match(rf"pasal {re.escape(pasal)}(?![0-9a-z])", norm) is not None


def score_item(item: dict[str, Any], codes_in_corpus: set[str]) -> dict[str, Any]:
    """Metrics for one item; None means "tidak dinilai" at that level."""
    locs = [parse_loc(p) for p in item["gt_lokasi"]]
    gt = [c for c in unique(item["gt_sumber"]) if c in codes_in_corpus]
    page_locs = [l for l in locs if l.code in codes_in_corpus and l.page_from is not None]
    pasal_locs = [l for l in locs if l.code in codes_in_corpus and l.pasal is not None]

    result: dict[str, Any] = {}
    for n, key in ((3, "top3"), (10, "top10")):
        got = [(doc_code(x.get("sumber") or ""), x.get("halaman"), x.get("unit") or "") for x in item[key]]
        got_codes = {c for c, _, _ in got}
        result[f"src@{n}"] = any(c in got_codes for c in gt) if gt else None
        result[f"hal@{n}"] = (
            any(
                c == l.code and isinstance(h, int) and l.page_from - 1 <= h <= l.page_to + 1
                for l in page_locs
                for c, h, _ in got
            )
            if page_locs
            else None
        )
        result[f"psl@{n}"] = (
            any(c == l.code and _pasal_match(u, l.pasal) for l in pasal_locs for c, _, u in got)
            if pasal_locs
            else None
        )
    top3_codes = {doc_code(x.get("sumber") or "") for x in item["top3"]}
    result["rec@3"] = len(set(gt) & top3_codes) / len(set(gt)) if gt else None
    return result


def summarise(items: list[dict[str, Any]], scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate means over assessed items, overall and per domain/kesulitan."""

    def mean(values: list[Any]) -> tuple[float | None, int]:
        assessed = [float(v) for v in values if v is not None]
        return (sum(assessed) / len(assessed) if assessed else None), len(assessed)

    metrics = ["src@3", "rec@3", "src@10", "hal@3", "hal@10", "psl@3", "psl@10"]
    out: dict[str, Any] = {"overall": {m: mean([s[m] for s in scores]) for m in metrics}}
    for field in ("domain", "kesulitan"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item, s in zip(items, scores):
            groups[item[field]].append(s)
        out[field] = {g: {m: mean([s[m] for s in ss]) for m in metrics} for g, ss in groups.items()}
    return out


# ---------------------------------------------------------------- printing


def fmt_flag(v: Any) -> str:
    return "·" if v is None else ("✓" if v else "✗")


def fmt_pct(pair: tuple[float | None, int]) -> str:
    value, n = pair
    return "   –    " if value is None else f"{value * 100:5.1f}% ({n})"


def print_table(items: list[dict[str, Any]], scores: list[dict[str, Any]]) -> None:
    print(f"\n{'id':7} {'domain':13} {'kes':6} {'src@3':>5} {'rec@3':>5} {'src@10':>6} "
          f"{'hal@3':>5} {'hal@10':>6} {'psl@3':>5} {'psl@10':>6} {'fb':>3}")
    for item, s in zip(items, scores):
        rec = "  ·" if s["rec@3"] is None else f"{s['rec@3']:.2f}"
        fb = "err" if item.get("error") else ("ya" if item.get("rerank_fallback") else "")
        print(f"{item['id']:7} {item['domain']:13} {item['kesulitan']:6} {fmt_flag(s['src@3']):>5} {rec:>5} "
              f"{fmt_flag(s['src@10']):>6} {fmt_flag(s['hal@3']):>5} {fmt_flag(s['hal@10']):>6} "
              f"{fmt_flag(s['psl@3']):>5} {fmt_flag(s['psl@10']):>6} {fb:>3}")
    print("  ✓ cocok  ✗ tidak cocok  · tidak dinilai   fb = rerank fallback")


def print_summary(items: list[dict[str, Any]], scores: list[dict[str, Any]]) -> None:
    summary = summarise(items, scores)
    cols = ["src@3", "rec@3", "src@10", "hal@3", "hal@10", "psl@3", "psl@10"]
    print("\nRingkasan (rata-rata atas pertanyaan yang dinilai; angka dalam kurung = jumlah dinilai)")
    print(f"  {'':22}" + "".join(f"{c:>13}" for c in cols))
    print(f"  {'SEMUA':22}" + "".join(f"{fmt_pct(summary['overall'][c]):>13}" for c in cols))
    for field in ("domain", "kesulitan"):
        for group, values in summary[field].items():
            print(f"  {field[:3]}={group:18}" + "".join(f"{fmt_pct(values[c]):>13}" for c in cols))

    print("\nTidak dinilai per tingkat:")
    for level in LEVELS:
        skipped = [i["id"] for i, s in zip(items, scores) if s[f"{level}@3"] is None]
        print(f"  {LEVEL_NAMES[level]:13}: {len(skipped):2d}  {', '.join(skipped) or '-'}")

    print("\nKontribusi rerank (benar di top-10 FAISS, hilang di top-3):")
    for level in LEVELS:
        lost = [i["id"] for i, s in zip(items, scores) if s[f"{level}@10"] and s[f"{level}@3"] is False]
        print(f"  {LEVEL_NAMES[level]:13}: {len(lost):2d}  {', '.join(lost) or '-'}")
    fallbacks = [i["id"] for i in items if i.get("rerank_fallback")]
    errors = [i["id"] for i in items if i.get("error")]
    print(f"\nRerank fallback: {len(fallbacks)}  {', '.join(fallbacks) or '-'}")
    print(f"Error          : {len(errors)}  {', '.join(errors) or '-'}")


def check_ground_truth(utama: list[dict[str, str]], cadangan: list[dict[str, str]], codes: set[str]) -> None:
    """Report problematic utama rows and same-domain/difficulty cadangan that could replace them."""
    problems: list[tuple[dict[str, str], str]] = []
    for row in utama:
        parts = split_lokasi(row["lokasi_sumber"])
        if not parts:
            problems.append((row, "lokasi_sumber kosong"))
            continue
        bad_format = [p for p in parts if not CODE_RE.match(p.partition(",")[0].strip())]
        missing = unique([parse_loc(p).code for p in parts if parse_loc(p).code not in codes])
        if bad_format:
            problems.append((row, f"kode tidak terbaca: {bad_format}"))
        if missing:
            problems.append((row, f"kode tidak ada di korpus: {missing}"))
    print("\nPemeriksaan ground truth (baris utama):")
    if not problems:
        print("  semua kode ada di korpus, tidak ada lokasi kosong")
        return
    for row, reason in problems:
        alt = [c["id"] for c in cadangan if c["domain"] == row["domain"] and c["kesulitan"] == row["kesulitan"]]
        print(f"  {row['id']}: {reason}; kandidat cadangan: {', '.join(alt) or 'tidak ada'}")


# ---------------------------------------------------------------- modes


def run_regression() -> list[dict[str, Any]]:
    utama, cadangan = load_ground_truth(GT_PATH)
    idx = load_index()
    codes = corpus_codes()
    s = get_settings()
    print(f"Ground truth: {GT_PATH.name}, {len(utama)} baris utama ({len(cadangan)} cadangan tidak dihitung)")

    def slim(i: int) -> dict[str, Any]:
        m = idx.metas[i]
        return {"sumber": m.get("sumber"), "unit": m.get("unit"), "halaman": m.get("halaman")}

    items: list[dict[str, Any]] = []
    for row in utama:
        parts = split_lokasi(row["lokasi_sumber"])
        item: dict[str, Any] = {
            "id": row["id"],
            "domain": row["domain"],
            "kesulitan": row["kesulitan"],
            "pertanyaan": row["pertanyaan"],
            "gt_sumber": unique([parse_loc(p).code for p in parts]),
            "gt_lokasi": parts,
            "top10": [],
            "top3": [],
            "rerank_fallback": None,
            "error": None,
        }
        try:
            found = retrieve(row["pertanyaan"].strip())
            item["top10"] = [slim(i) for i in found.candidates]
            item["top3"] = [slim(i) for i in found.ids]
            item["rerank_fallback"] = found.rerank_fallback
        except Exception as exc:  # noqa: BLE001 - record and keep going
            item["error"] = f"{type(exc).__name__}: {exc}"
        items.append(item)
        print(f"  {row['id']}: {'ERROR ' + item['error'] if item['error'] else 'ok'}", flush=True)

    build_info = json.loads((INDEX_DIR / INFO_FILE).read_text(encoding="utf-8"))
    result = {
        "schema_version": SCHEMA_VERSION,
        "repo": "amwal-rag",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "embed_model": s.jina_embed_model,
            "reranker": s.jina_rerank_model,
            "top_k": s.retrieve_top_k,
            "top_n": s.rerank_top_n,
            "ground_truth_file": GT_PATH.name,
            "ground_truth_sha256": sha256(GT_PATH),
            "filter": "status=utama",
            "chunks_sha256": build_info.get("chunks_sha256"),
        },
        "items": items,
    }
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    scores = [score_item(i, codes) for i in items]
    print_table(items, scores)
    print_summary(items, scores)
    check_ground_truth(utama, cadangan, codes)
    print(f"\nDisimpan ke {OUT_PATH}")
    return items


def run_compare(deploy_items: list[dict[str, Any]], path: Path) -> None:
    print(f"\n=== Perbandingan dengan {path.name} ===")
    if not path.exists():
        print(f"  {path} belum ada; perbandingan dilewati.")
        return
    other = json.loads(path.read_text(encoding="utf-8"))
    if other.get("schema_version") != SCHEMA_VERSION:
        print(f"  schema_version {other.get('schema_version')!r} tidak didukung (harus {SCHEMA_VERSION}).")
        return
    codes = corpus_codes()
    other_by_id = {i["id"]: i for i in other.get("items", [])}
    common = [i for i in deploy_items if i["id"] in other_by_id]
    only = sorted(set(other_by_id) ^ {i["id"] for i in deploy_items})
    if only:
        print(f"  id yang hanya ada di salah satu file (diabaikan): {', '.join(only)}")
    ours = [score_item(i, codes) for i in common]
    theirs = [score_item(other_by_id[i["id"]], codes) for i in common]
    a, b = summarise(common, ours)["overall"], summarise(common, theirs)["overall"]
    print(f"  {'metrik':8} {'skripsi':>13} {'deploy':>13}")
    for m in ("src@3", "rec@3", "src@10", "hal@3", "hal@10", "psl@3", "psl@10"):
        print(f"  {m:8} {fmt_pct(b[m]):>13} {fmt_pct(a[m]):>13}")
    for level in LEVELS:
        worse = [i["id"] for i, x, y in zip(common, ours, theirs) if y[f"{level}@3"] and x[f"{level}@3"] is False]
        better = [i["id"] for i, x, y in zip(common, ours, theirs) if x[f"{level}@3"] and y[f"{level}@3"] is False]
        print(f"  {LEVEL_NAMES[level]:13}@3 lulus di skripsi, gagal di deploy: {', '.join(worse) or '-'}")
        print(f"  {LEVEL_NAMES[level]:13}@3 gagal di skripsi, lulus di deploy: {', '.join(better) or '-'}")


def run_offtopic() -> int:
    print(f"Off-topic check: {len(OFFTOPIC_QUESTIONS)} pertanyaan (memanggil DeepSeek)\n")
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for q in OFFTOPIC_QUESTIONS:
        try:
            result = answer(q)
            ans, sources = result["answer"], result["sources"]
            if sources:
                verdict, note = "FAIL", f"sources tidak kosong ({len(sources)})"
            elif ans == REFUSAL_OFF_TOPIC:
                verdict, note = "PASS", "penolakan luar topik"
            elif ans == REFUSAL_NOT_FOUND:
                verdict, note = "WARN", "ditolak dengan kalimat 'informasi tidak ditemukan'"
            else:
                verdict, note = "FAIL", "pertanyaan dijawab"
        except Exception as exc:  # noqa: BLE001
            verdict, note, ans = "FAIL", f"{type(exc).__name__}: {exc}", ""
        counts[verdict] += 1
        print(f"[{verdict}] {q}\n       {note}\n       jawaban: {ans[:160]!r}")
    print(f"\nPASS {counts['PASS']}  WARN {counts['WARN']}  FAIL {counts['FAIL']}")
    return 1 if counts["FAIL"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Regression test retrieval terhadap ground truth")
    parser.add_argument("--compare", nargs="?", const=str(SKRIPSI_PATH), metavar="FILE",
                        help=f"bandingkan dengan FILE (default {SKRIPSI_PATH.name})")
    parser.add_argument("--offtopic", action="store_true", help="uji 5 pertanyaan luar topik lewat answer()")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(name)s: %(message)s")

    if args.offtopic:
        return run_offtopic()
    items = run_regression()
    if args.compare:
        run_compare(items, Path(args.compare))
    return 0


if __name__ == "__main__":
    sys.exit(main())
