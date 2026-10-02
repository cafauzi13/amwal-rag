# amwal-rag

Chatbot ZISWAF (RAG) untuk amwal.site: `POST /chat` dan `GET /health`. Kontrak API ada di [API_CONTRACT.md](API_CONTRACT.md).

**Asumsi:** chatbot berjalan sebagai container di jaringan Docker `dokploy-network` bersama backend AMWAL dan tidak membuka port ke host; backend memanggil `http://amwal-rag:8000`. File `docker-compose.yml` (port `127.0.0.1:8000`) hanya untuk Docker biasa atau uji lokal.

## Deploy dengan Dokploy

File compose: [docker-compose.dokploy.yml](docker-compose.dokploy.yml). Indeks FAISS disimpan di named volume `amwal-rag-index`, karena Dokploy menghapus isi folder repo setiap deploy.

1. **Buat service tipe Compose** dari repo Git private ini, dengan *compose path* `./docker-compose.dokploy.yml`. Pastikan **Randomize Compose** (dan isolated deployment) **mati**, supaya nama service tetap `amwal-rag`.
2. **Isi variabel di tab Environment**, mengikuti [.env.example](.env.example). Wajib: `API_KEY`, `JINA_API_KEY`, `DEEPSEEK_API_KEY`. Buat `API_KEY` acak dengan:
   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   `API_KEY` yang kosong atau masih placeholder membuat `/chat` membalas 500 "Server belum dikonfigurasi."
3. **Deploy.** Sebelum indeks dibangun, status container `unhealthy` dan `/health` membalas 503. Itu normal.
4. **Bangun indeks sekali** lewat terminal container di Dokploy:
   ```bash
   python -m scripts.build_index
   ```
   Butuh ±5 menit dan ±370 ribu token Jina. Isi `JINA_TARGET_TPM=0` bila key Jina tier berbayar, supaya lebih cepat. Untuk build pertama tidak perlu restart: request berikutnya langsung memuat indeks dan status menjadi `healthy`. Bila indeks dibangun ulang saat sudah termuat (`--fresh`), **restart service** sesudahnya.
5. **Panggil dari backend AMWAL:** `POST http://amwal-rag:8000/chat` dengan header `X-API-Key: <API_KEY>` dan body `{"question": "..."}`. Container backend **harus ikut di jaringan `dokploy-network`**. Timeout klien ≥ 70 detik; retry sekali hanya untuk 429/503 (detail di [API_CONTRACT.md](API_CONTRACT.md)).
6. **Jangan hapus volume `amwal-rag-index`.** Redeploy tidak menyentuhnya; tanpa volume itu indeks harus dibangun ulang.

`API_KEY` hanya dipakai dari server backend. Jangan pernah menanamnya di JavaScript/HTML yang dikirim ke browser.

## Belum diverifikasi tanpa Dokploy

`docker-compose.dokploy.yml` sudah diuji dengan Docker Desktop (Docker 29.8.1, Compose v5.5.1) memakai jaringan bridge `dokploy-network` yang dibuat manual: build image, build indeks di named volume oleh user non-root (tanpa `chown`), status `healthy`, `/chat` dari container lain lewat `http://amwal-rag:8000` (200, dan 401 untuk key salah), serta indeks tetap ada setelah `down` lalu `up`. Hal-hal berikut hanya bisa dipastikan di server Dokploy:

- **`env_file: .env` dari tab Environment.** Dokploy menulis variabel tab Environment ke file `.env` di samping file compose; bila tidak, container berjalan tanpa key (`/chat` 500, log startup menyebut `API_KEY belum diisi`).
- **Randomize Compose / isolated deployment harus mati.** Bila menyala, Dokploy menambahkan akhiran pada nama service, volume, dan jaringan, sehingga alamat `amwal-rag` dan volume `amwal-rag-index` berubah.
- **Persistensi volume saat redeploy.** Named volume seharusnya bertahan (perilaku standar Docker); cek bahwa setelah redeploy `/health` langsung 200 tanpa build ulang.
- **Status deploy saat container `unhealthy`** sebelum indeks ada: Dokploy mungkin menandai deploy gagal atau terus menunggu. Lanjutkan ke langkah 4 selama container berjalan.
- **Backend harus ada di `dokploy-network`** dan bisa me-resolve nama `amwal-rag`.
- **Jenis jaringan di server bisa berbeda** dari uji lokal (mis. jaringan overlay Swarm buatan Dokploy vs. bridge lokal). Bila container tidak bisa bergabung ke `dokploy-network`, sesuaikan definisi jaringan di `docker-compose.dokploy.yml`.

## Docker biasa / uji lokal

```bash
mkdir -p index
docker compose build
docker compose run --rm api python -m scripts.build_index   # sekali
docker compose up -d
curl -s http://127.0.0.1:8000/health
```

Container berjalan sebagai user host (`APP_UID`/`APP_GID`, default 1000) agar bind mount `./index` bisa ditulis; isi keduanya di `.env` bila `id -u` / `id -g` bukan 1000.
