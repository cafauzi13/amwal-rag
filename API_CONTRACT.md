# Kontrak API — Chatbot ZISWAF (amwal-rag)

Layanan tanya-jawab ZISWAF berbasis dokumen regulasi dan literasi. Dipanggil **dari backend AMWAL** (server-to-server), bukan dari browser.

- **Base URL:** `https://<subdomain>.amwal.site` (menyusul setelah subdomain ditetapkan)
- **Format:** JSON UTF-8 (`Content-Type: application/json`)

## Batas tanggung jawab

- Layanan ini **stateless** dan **single-turn**: setiap request berdiri sendiri, tanpa session dan tanpa riwayat percakapan. Jangan kirim pertanyaan sebelumnya.
- Layanan ini **tidak menyimpan** pertanyaan maupun jawaban sebagai riwayat. Jika riwayat percakapan dibutuhkan (tampilan, audit, analitik), penyimpanannya ada di sisi backend AMWAL.

## Autentikasi

`POST /chat` wajib menyertakan header:

```
X-API-Key: <shared secret>
```

Nilai key dibagikan terpisah (tidak lewat repo) dan disimpan di konfigurasi backend AMWAL. `GET /health` tidak memerlukan key.

## Request ID

Setiap response (sukses maupun error) berisi `request_id`, juga di header `X-Request-ID`.

- Klien API boleh mengirim header `X-Request-ID`. Nilai itu dipakai bila **maks. 64 karakter** dan hanya berisi **huruf, angka, dan tanda hubung** (`-`).
- Bila tidak dikirim atau tidak valid, server membuat ID baru (tanpa error).

Cantumkan `request_id` saat melaporkan masalah agar mudah dilacak di log.

## `POST /chat`

### Request

```json
{ "question": "Apa syarat menjadi nazhir perseorangan?" }
```

| Field | Tipe | Aturan |
|---|---|---|
| `question` | string | Wajib. Spasi di awal/akhir dipangkas, lalu panjangnya harus **3–500 karakter**. |

### Response `200 OK`

```json
{
  "request_id": "b3f1c2a4-9d7e-4f0a-8c21-5e6d7a8b9c0d",
  "answer": "Menurut UU No. 41 Tahun 2004 Pasal 10 ayat (1), perseorangan dapat menjadi nazhir apabila memenuhi syarat: warga negara Indonesia, beragama Islam, dewasa, amanah, mampu secara jasmani dan rohani, serta tidak terhalang melakukan perbuatan hukum.\n\nJawaban ini bersifat informatif; untuk kepastian hukum, rujuk dokumen resmi atau lembaga terkait.",
  "sources": [
    { "sumber": "WKF-REG-02_UU-41-2004.pdf", "unit": "Pasal 10", "halaman": 5 }
  ]
}
```

- `answer`: **teks polos tanpa Markdown**, bisa mengandung baris baru (`\n`). Sudah termasuk disclaimer. Tampilkan dengan escape HTML; ubah `\n` menjadi `<br>` bila perlu.
- `sources`: 0–3 entri, tanpa duplikat `sumber`+`halaman`.
  - `sumber` (string): nama dokumen asal.
  - `unit` (string): Pasal atau judul bagian, mis. `"Pasal 10"`, `"Pembukaan"`.
  - `halaman` (integer atau `null`): `null` bila halaman tidak diketahui.
- Pertanyaan di luar topik ZISWAF tetap **200**: `answer` berisi penolakan sopan dan `sources` berupa `[]`.

## `GET /health`

Hanya memeriksa apakah indeks sudah termuat; tidak memanggil API eksternal. Cocok untuk healthcheck Docker/nginx.

```json
{ "request_id": "...", "status": "ok" }
```

- `200` → `"status": "ok"`
- `503` → `"status": "unavailable"` (indeks belum termuat)

## Format error

Semua error memakai bentuk yang sama (termasuk error validasi):

```json
{
  "request_id": "b3f1c2a4-9d7e-4f0a-8c21-5e6d7a8b9c0d",
  "error": { "code": "invalid_question", "message": "Pertanyaan harus 3–500 karakter." }
}
```

`message` berbahasa Indonesia dan aman ditampilkan ke pengguna.

| HTTP | `code` | Kapan | Retry otomatis? |
|---|---|---|---|
| 401 | `invalid_api_key` | Header `X-API-Key` tidak ada atau salah | Tidak |
| 422 | `invalid_question` | `question` tidak ada, bukan string, atau di luar 3–500 karakter | Tidak |
| 429 | `rate_limited` | Server sedang penuh (lihat *Batas request bersamaan*) | Ya, sekali |
| 404 / 405 | `not_found` / `method_not_allowed` | Path tidak ada / metode HTTP salah (mis. `GET /chat`) | Tidak |
| 500 | `internal_error` | Error tak terduga di server, atau server belum dikonfigurasi (API key server kosong) | Tidak |
| 503 | `upstream_unavailable` | Layanan AI (embedding/LLM) sedang tidak tersedia atau kuotanya habis | Ya, sekali |
| 504 | `timeout` | Pemrosesan melebihi 60 detik | Tidak |

## Batas request bersamaan

Server memproses paling banyak `MAX_CONCURRENT_REQUESTS` request `/chat` sekaligus (default **10**, diatur lewat `.env` server). Batas ini **global**, bukan per IP. Request yang melebihi batas menunggu di antrean singkat (beberapa detik); bila tetap belum mendapat giliran, server membalas `429 rate_limited`.

## Catatan integrasi

- **Timeout HTTP klien: ≥ 70 detik** (server memotong di 60 detik).
- **Retry:** hanya untuk `429` dan `503`, maksimal sekali, beri jeda ±2 detik. Jangan retry `504` secara otomatis.
- Waktu respons normal ±3–10 detik; tampilkan indikator loading.

## Contoh curl

```bash
# Sukses
curl -X POST https://<subdomain>.amwal.site/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $AMWAL_RAG_KEY" \
  -d '{"question": "Apa syarat menjadi nazhir perseorangan?"}'

# 422 — pertanyaan terlalu pendek
curl -X POST https://<subdomain>.amwal.site/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $AMWAL_RAG_KEY" \
  -d '{"question": "hi"}'

# Health
curl https://<subdomain>.amwal.site/health
```
