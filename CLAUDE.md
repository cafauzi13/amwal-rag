# amwal-rag

RAG chatbot ZISWAF untuk amwal.site: embed query (Jina) → FAISS top-10 → rerank top-3 (Jina) → jawaban (DeepSeek). Data sumber di `data/chunks.jsonl`; indeks FAISS di `index/` (tidak di-commit, dibangun ulang dari data). Kontrak API untuk backend AMWAL ada di `API_CONTRACT.md` — jaga agar kode dan kontrak tetap sinkron.

## Aturan internal (wajib dipatuhi)

1. **Retry di jalur `/chat`:** panggilan Jina dan DeepSeek maksimal **1 retry singkat**. Jangan pakai backoff panjang milik build index (`JINA_RATE_LIMIT_BACKOFF` 5/15/30 dan `JINA_TARGET_TPM` pacing hanya untuk `build_index`). Server memotong total pemrosesan di 60 detik (→ 504): `answer()` menerima `deadline` (`time.monotonic()`), dan setiap timeout HTTP serta jeda retry ke Jina/DeepSeek dipangkas ke sisa anggaran (`app/clients.time_left`), supaya thread tidak menggantung. Slot `MAX_CONCURRENT_REQUESTS` baru dilepas saat thread benar-benar selesai.
2. **Rerank gagal → fallback top-3 FAISS.** Kegagalan rerank tidak boleh membuat `/chat` error.
3. **Prompt sistem melarang Markdown:** `answer` harus teks polos (boleh baris baru), sudah berisi disclaimer, dan pertanyaan di luar ZISWAF ditolak dengan sopan (200, `sources: []`).
4. **`/health` tidak memanggil API eksternal;** hanya cek indeks termuat (200 / 503).
5. **`X-API-Key` wajib untuk `POST /chat`** (401 `invalid_api_key`); `/health` tanpa key. Tidak ada CORS — layanan dipanggil server-to-server.
6. **Batas request bersamaan global:** `MAX_CONCURRENT_REQUESTS` di `.env` (default 10), dengan antrean singkat beberapa detik sebelum menolak dengan 429 `rate_limited`. Bukan pembatasan per IP (semua request datang dari satu backend).
7. **`X-Request-ID` dari klien** dipakai hanya bila ≤ 64 karakter dan hanya `[A-Za-z0-9-]`; selain itu buat ID baru. `request_id` ada di setiap response dan error.
8. **Stateless, single-turn:** tidak menyimpan riwayat percakapan.

## Konvensi

- Semua parameter dibaca dari `.env` lewat `app/config.py`; `.env.example` harus ikut diperbarui. Jangan pernah membaca, menampilkan, atau meng-commit `.env`.
- Pesan error untuk pengguna berbahasa Indonesia.
- Teks chunk diawali `"passage: "` (sisa pipeline E5 skripsi); buang saat membaca `chunks.jsonl`, karena Jina memakai parameter `task`.
- Smoke test: `python -m scripts.smoke_test` (Windows: aktifkan `.venv` dulu).

## Aturan kerja

1. **Jangan commit/push kecuali diminta eksplisit.** Jangan menjalankan `git add`, `git commit`, atau `git push`. Setelah tugas selesai, tampilkan `git status --short` dan `git diff --stat`, lalu usulkan pengelompokan file dan pesan commit sebagai teks. Tunggu konfirmasi bahwa commit sudah dilakukan sebelum lanjut ke langkah berikutnya.
2. **Jangan menghentikan proses berdasarkan nama image** (mis. `taskkill /IM python.exe`). Hanya hentikan proses yang kamu jalankan sendiri, berdasarkan PID-nya.
