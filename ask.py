"""CLI for the RAG pipeline.

Run from the repo root:  python ask.py "Apa syarat menjadi nazhir perseorangan?" [--json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from app.clients import APIRequestError
from app.rag import IndexNotReadyError, answer


def main() -> int:
    parser = argparse.ArgumentParser(description="Tanya chatbot ZISWAF (single-turn)")
    parser.add_argument("question", help="pertanyaan, dalam tanda kutip")
    parser.add_argument("--json", action="store_true", help="cetak hasil mentah sebagai JSON")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        result = answer(args.question)
    except (ValueError, IndexNotReadyError, APIRequestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print(result["answer"])
    if result["sources"]:
        print("\nSumber:")
        for n, src in enumerate(result["sources"], start=1):
            page = f", hal. {src['halaman']}" if src["halaman"] is not None else ""
            print(f"  {n}. {src['sumber']} — {src['unit']}{page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
