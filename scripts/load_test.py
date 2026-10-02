"""Concurrent /chat load test: fire N requests at once and summarise status codes and latency.

Run from the repo root (or inside the image):
  python -m scripts.load_test --n 8 --url http://127.0.0.1:8000

The API key comes from the API_KEY environment variable (falling back to .env via app.config)
and is never printed. Each request sends X-Request-ID load-<i> so it can be matched in the logs.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from collections import Counter
from typing import Any

import httpx

QUESTIONS = [
    "Apa syarat menjadi nazhir perseorangan?",
    "Apa yang dimaksud dengan wakaf uang?",
    "Siapa saja yang berhak menerima zakat?",
    "Berapa nisab zakat penghasilan?",
    "Apa perbedaan infak dan sedekah?",
    "Bagaimana pengelolaan dana infak dan sedekah oleh BAZNAS?",
    "Apa syarat hewan kurban yang sah?",
    "Bagaimana hukum menjual kulit hewan kurban?",
]


def api_key() -> str:
    key = os.environ.get("API_KEY", "").strip()
    if not key:
        from app.config import get_settings  # reads .env when run from the repo

        key = get_settings().api_key
    return key


async def one(client: httpx.AsyncClient, i: int, url: str, key: str, start: asyncio.Event) -> dict[str, Any]:
    question = QUESTIONS[i % len(QUESTIONS)]
    sent_id = f"load-{i + 1}"
    await start.wait()
    began = time.perf_counter()
    result: dict[str, Any] = {"no": i + 1, "sent_id": sent_id, "question": question}
    try:
        resp = await client.post(
            f"{url}/chat",
            json={"question": question},
            headers={"X-API-Key": key, "X-Request-ID": sent_id},
        )
    except httpx.HTTPError as exc:
        result.update(status=None, latency=time.perf_counter() - began, code=type(exc).__name__,
                      rid_ok=False, sources=None)
        return result
    result["latency"] = time.perf_counter() - began
    result["status"] = resp.status_code
    try:
        body = resp.json()
    except ValueError:
        body = {}
    result["code"] = (body.get("error") or {}).get("code", "")
    result["sources"] = len(body["sources"]) if isinstance(body.get("sources"), list) else None
    result["rid_ok"] = body.get("request_id") == resp.headers.get("X-Request-ID") == sent_id
    return result


async def run(n: int, url: str, timeout: float) -> list[dict[str, Any]]:
    key = api_key()
    if not key:
        sys.exit("ERROR: API_KEY kosong (set env API_KEY atau isi .env).")
    start = asyncio.Event()
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [asyncio.create_task(one(client, i, url, key, start)) for i in range(n)]
        await asyncio.sleep(0.2)  # let every task reach start.wait()
        start.set()  # release them together
        return list(await asyncio.gather(*tasks))


def fmt_latency(values: list[float]) -> str:
    if not values:
        return "-"
    return f"min {min(values):.2f}s  median {statistics.median(values):.2f}s  maks {max(values):.2f}s"


def report(results: list[dict[str, Any]]) -> None:
    print(f"\n{'no':>3} {'status':>6} {'latensi':>8} {'kode error':22} {'src':>3} {'req_id':>6}  pertanyaan")
    for r in sorted(results, key=lambda x: x["no"]):
        status = "-" if r["status"] is None else str(r["status"])
        src = "" if r["sources"] is None else str(r["sources"])
        print(f"{r['no']:>3} {status:>6} {r['latency']:7.2f}s {r['code'] or '':22} {src:>3} "
              f"{'ok' if r['rid_ok'] else 'TIDAK':>6}  {r['question'][:45]}")

    counts = Counter("tanpa respons" if r["status"] is None else r["status"] for r in results)
    main = {s: counts.get(s, 0) for s in (200, 429, 503, 504)}
    other = {s: c for s, c in counts.items() if s not in main}
    print(f"\nRingkasan ({len(results)} request bersamaan)")
    print("  status : " + "  ".join(f"{s}={c}" for s, c in main.items())
          + ("  lainnya: " + ", ".join(f"{s}={c}" for s, c in other.items()) if other else ""))
    print(f"  latensi semua : {fmt_latency([r['latency'] for r in results])}")
    print(f"  latensi 200   : {fmt_latency([r['latency'] for r in results if r['status'] == 200])}")
    rid_ok = sum(1 for r in results if r["rid_ok"])
    print(f"  request_id lengkap (body = header = yang dikirim): {rid_ok}/{len(results)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Uji paralel POST /chat")
    parser.add_argument("--n", type=int, default=8, help="jumlah request bersamaan (default 8)")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="base URL layanan")
    parser.add_argument("--timeout", type=float, default=75.0, help="timeout klien per request (detik)")
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n harus > 0")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Mengirim {args.n} request /chat bersamaan ke {args.url}")
    report(asyncio.run(run(args.n, args.url.rstrip("/"), args.timeout)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
