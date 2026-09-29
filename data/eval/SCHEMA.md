# Skema hasil regression test retrieval

Dipakai oleh dua file dengan struktur yang sama:

- `regression_deploy.json`: dibuat oleh `amwal-rag` (`python -m scripts.regression_test`)
- `regression_skripsi.json`: dibuat oleh repo skripsi `rag-waqf-chatbot` (Sesi A)

File hanya berisi **hasil retrieval mentah**. Metrik tidak disimpan; keduanya dinilai ulang oleh skrip yang sama (`scripts/regression_test.py --compare`) supaya adil.

## Struktur

```json
{
  "schema_version": 1,
  "repo": "amwal-rag",
  "generated_at": "2026-09-29T09:00:00+00:00",
  "config": {
    "embed_model": "jina-embeddings-v5-text-small",
    "reranker": "jina-reranker-v3.5",
    "top_k": 10,
    "top_n": 3,
    "ground_truth_file": "ground_truth_draf_v2.csv",
    "ground_truth_sha256": "…",
    "filter": "status=utama"
  },
  "items": [
    {
      "id": "W-R02",
      "domain": "wakaf",
      "kesulitan": "rendah",
      "pertanyaan": "Apa yang dimaksud dengan nazhir?",
      "gt_sumber": ["WKF-REG-02", "WKF-LIT-18"],
      "gt_lokasi": ["WKF-REG-02, Pasal 1 angka 4", "WKF-LIT-18, halaman 1"],
      "top10": [{"sumber": "WKF-REG-02_UU-41-2004.pdf", "unit": "Pasal 1", "halaman": 2}],
      "top3":  [{"sumber": "WKF-LIT-18_Syarat-dan-Ketentuan-Nazhir-BWI.pdf", "unit": "Pembukaan", "halaman": 1}],
      "rerank_fallback": false,
      "error": null
    }
  ]
}
```

## Field

| Field | Tipe | Isi |
|---|---|---|
| `schema_version` | int | `1` |
| `repo` | string | `"amwal-rag"` atau `"rag-waqf-chatbot"` |
| `generated_at` | string | Waktu ISO 8601 (UTC) |
| `config.*` | object | Model embedding/reranker, `top_k`, `top_n`, nama dan SHA-256 file ground truth, filter baris. Boleh ditambah field lain (mis. `chunks_sha256`) |
| `items[].id`, `domain`, `kesulitan`, `pertanyaan` | string | Disalin apa adanya dari CSV |
| `items[].gt_sumber` | list[string] | Kode dokumen unik dari `lokasi_sumber` (teks sebelum koma pertama tiap bagian), urut kemunculan |
| `items[].gt_lokasi` | list[string] | `lokasi_sumber` dipecah di `\|\|`, tiap bagian di-trim, **apa adanya** (duplikat dipertahankan) |
| `items[].top10` | list[obj] | Hasil FAISS **sebelum** rerank, urut skor, tanpa dedup |
| `items[].top3` | list[obj] | Hasil akhir **setelah** rerank (atau fallback), urut peringkat, tanpa dedup |
| `…{sumber}` | string | Nama file persis seperti di metadata chunk, mis. `WKF-REG-02_UU-41-2004.pdf` |
| `…{unit}` | string | Field `unit` chunk apa adanya (tanpa dipotong) |
| `…{halaman}` | int / null | Halaman chunk; `null` bila tidak ada |
| `items[].rerank_fallback` | bool / null | `true` bila rerank gagal dan top-3 diambil dari FAISS; `null` bila tidak berlaku |
| `items[].error` | string / null | Pesan error bila pertanyaan gagal diproses; saat itu `top10`/`top3` = `[]` |

## Aturan untuk pengisi

- Hanya baris `status = utama` dari `ground_truth_draf_v2.csv`, urutan sama dengan CSV.
- Bila metadata repo memakai key lain (`source`, `page`), petakan ke `sumber`, `halaman`.
- Jangan simpan id chunk atau indeks vektor: kedua repo memakai model embedding berbeda, jadi perbandingan hanya lewat `(sumber, halaman)`.

## Cara penilaian (untuk referensi)

Kode dokumen chunk = `sumber` dipotong sebelum `_` pertama, dicocokkan **sama persis** dengan `gt_sumber`.

| Tingkat | Cocok bila | Dinilai bila |
|---|---|---|
| Sumber (utama) | minimal satu kode GT ada di top-N; **recall@3** = proporsi kode GT unik yang muncul di top-3 | ada kode GT yang terdapat di korpus |
| Halaman | kode sama dan halaman chunk dalam `[awal−1, akhir+1]` dari `halaman N` / `halaman N-M` | ada bagian `gt_lokasi` dengan halaman |
| Pasal | dokumen `-REG-`, kode sama, dan `unit` (huruf kecil, spasi dirapikan) diawali `pasal N` tanpa angka/huruf lanjutan | ada bagian `gt_lokasi` dokumen REG yang menyebut Pasal |

Pertanyaan yang tidak memenuhi syarat "dinilai" pada suatu tingkat dilaporkan sebagai **tidak dinilai**, bukan gagal. Kode GT yang tidak ada di korpus diabaikan untuk penilaian.
