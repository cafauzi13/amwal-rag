# amwal-rag

Chatbot ZISWAF (wakaf, zakat, infak, sedekah, kurban) untuk amwal.site, berbasis RAG atas dokumen regulasi dan literasi. Alurnya: pertanyaan → embedding (Jina) → pencarian FAISS top-10 → rerank top-3 (Jina) → jawaban (DeepSeek). Layanan ini **stateless dan single-turn**, dipanggil server-to-server oleh backend AMWAL lewat `POST /chat`, dengan `GET /health` untuk pemeriksaan. Format request, response, dan error ada di [API_CONTRACT.md](API_CONTRACT.md).

**Asumsi:** chatbot berjalan sebagai container di jaringan Docker `dokploy-network` bersama backend AMWAL dan tidak membuka port ke host; backend memanggil `http://amwal-rag:8000`. File `docker-compose.yml` (port `127.0.0.1:8000`) hanya untuk Docker biasa atau uji lokal.

## Prasyarat

- Server dengan Dokploy (Docker dan Compose sudah termasuk) dan jaringan `dokploy-network` (dibuat oleh Dokploy).
- Akses ke repo Git private ini dari Dokploy.
- Key API **Jina** dan **DeepSeek** dari akun HETI, dengan saldo/kuota aktif.
- Container backend AMWAL ikut di jaringan `dokploy-network`.

## Konfigurasi

Semua pengaturan dibaca dari variabel lingkungan. Di Dokploy, isi di **tab Environment**; untuk Docker biasa, salin [.env.example](.env.example) menjadi `.env`. File `.env` tidak boleh di-commit.

**Wajib diisi**

| Variabel | Isi |
|---|---|
| `API_KEY` | Secret bersama yang dikirim backend sebagai header `X-API-Key`. Buat acak (lihat di bawah). Kosong atau masih `ganti-dengan-secret-acak` → `/chat` membalas 500 "Server belum dikonfigurasi." |
| `JINA_API_KEY` | Key Jina (embedding dan rerank). |
| `DEEPSEEK_API_KEY` | Key DeepSeek (pembuat jawaban). |

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Sering disesuaikan**

| Variabel | Default | Arti |
|---|---|---|
| `MAX_CONCURRENT_REQUESTS` | `5` | Jumlah `/chat` yang diproses bersamaan. Lihat [Kapasitas dan batas](#kapasitas-dan-batas). |
| `QUEUE_WAIT_SECONDS` | `5` | Lama request menunggu slot sebelum dibalas 429. |
| `REQUEST_TIMEOUT_SECONDS` | `60` | Batas waktu total satu `/chat` sebelum dibalas 504. |
| `JINA_TARGET_TPM` | `80000` | Jeda antarbatch saat build indeks agar di bawah batas key gratis (100 ribu token/menit). Isi `0` untuk key berbayar. |
| `ENABLE_DOCS` | `false` | `true` membuka `/docs` dan `/openapi.json`. Hanya untuk uji lokal, jangan di produksi. |

**Biarkan default kecuali tahu alasannya:** model (`DEEPSEEK_MODEL`, `JINA_EMBED_MODEL`, `JINA_RERANK_MODEL`), `DEEPSEEK_BASE_URL`, `LLM_TEMPERATURE`, `RETRIEVE_TOP_K`, `RERANK_TOP_N`, `HTTP_TIMEOUT`, `CHAT_MAX_RETRIES`, `CHAT_RETRY_DELAY`, serta retry/backoff khusus build indeks (`LLM_MAX_RETRIES`, `JINA_MAX_RETRIES`, `JINA_RATE_LIMIT_BACKOFF`, `JINA_SERVER_BACKOFF`, `JINA_EMBED_BATCH_SIZE`). Mengganti `JINA_EMBED_MODEL` mengharuskan indeks dibangun ulang.

## Deploy dengan Dokploy

File compose: [docker-compose.dokploy.yml](docker-compose.dokploy.yml). Indeks FAISS disimpan di named volume `amwal-rag-index`, karena Dokploy menghapus isi folder repo setiap deploy.

1. **Buat service tipe Compose** dari repo Git private ini, dengan *compose path* `./docker-compose.dokploy.yml`. Pastikan **Randomize Compose** (dan isolated deployment) **mati**, supaya nama service tetap `amwal-rag`.
2. **Isi variabel di tab Environment** (lihat [Konfigurasi](#konfigurasi)), minimal `API_KEY`, `JINA_API_KEY`, `DEEPSEEK_API_KEY`.
3. **Deploy.** Sebelum indeks dibangun, status container `unhealthy`, `/health` membalas 503, dan `/chat` membalas 503 `upstream_unavailable` tanpa memanggil Jina/DeepSeek. Itu normal.
4. **Bangun indeks sekali** dari dalam container, lewat terminal container di Dokploy atau lewat SSH ke server:
   ```bash
   docker ps --filter name=amwal-rag            # cari nama container, mis. <app>-amwal-rag-1
   docker exec -it <container> python -m scripts.build_index
   ```
   Butuh ±5 menit dan ±370 ribu token Jina (lebih cepat dengan `JINA_TARGET_TPM=0` bila key berbayar). Tidak perlu restart: request berikutnya langsung memuat indeks dan status menjadi `healthy` (terbukti dalam uji lokal).
5. **Periksa:**
   ```bash
   docker inspect --format '{{.State.Health.Status}}' <container>          # harus: healthy
   docker exec <container> python -m scripts.load_test --n 1 --url http://amwal-rag:8000
   ```
   Perintah kedua mengirim satu `/chat` lewat jaringan `dokploy-network` memakai `API_KEY` milik container itu sendiri (tidak dicetak), dan harus menampilkan `200=1`.
6. **Panggil dari backend AMWAL:** `POST http://amwal-rag:8000/chat`, header `Content-Type: application/json` dan `X-API-Key: <API_KEY>`, body `{"question": "..."}`. Contoh dari container backend:
   ```bash
   curl -s -X POST http://amwal-rag:8000/chat \
     -H "Content-Type: application/json" -H "X-API-Key: <API_KEY>" \
     -d '{"question": "Apa syarat menjadi nazhir perseorangan?"}'
   ```
   Timeout HTTP klien ≥ 70 detik; retry sekali hanya untuk 429/503 (detail di [API_CONTRACT.md](API_CONTRACT.md)).
7. **Jangan hapus volume `amwal-rag-index`.** Redeploy tidak menyentuhnya; tanpa volume itu indeks harus dibangun ulang.

`API_KEY` hanya dipakai dari server backend. Jangan pernah menanamnya di JavaScript/HTML yang dikirim ke browser.

## Operasional

- **Log:** tab Logs di Dokploy, atau `docker logs <container>`. Satu baris per request berisi `request_id`, method, path, status, dan durasi. Isi pertanyaan, jawaban, dan key tidak pernah dicatat. Cocokkan masalah dengan `request_id` yang diterima backend.
- **Update kode:** redeploy di Dokploy. Indeks di volume tetap dipakai selama `data/chunks.jsonl` dan `JINA_EMBED_MODEL` tidak berubah.
- **Kapan indeks harus dibangun ulang:** bila `data/chunks.jsonl` berubah atau `JINA_EMBED_MODEL` diganti. Tandanya `/health` 503 dan log startup berisi "chunks.jsonl berubah sejak indeks dibangun" atau "Indeks dibangun dengan model …, tetapi JINA_EMBED_MODEL=…". Jalankan:
  ```bash
  docker exec -it <container> python -m scripts.build_index --fresh
  ```
  lalu **restart service** (indeks lama sudah dimuat di memori dan tidak diganti otomatis).
- **Indeks rusak atau tidak terbaca** (file terpotong, JSON rusak): `/health` dan `/chat` membalas 503, log berisi `Gagal memuat indeks (<jenis error>)`. Bangun ulang seperti di atas.
- **Cek ketiga API (opsional, memakai sedikit kuota):** `docker exec <container> python -m scripts.smoke_test`.
- **Ganti `API_KEY`:** ubah di tab Environment, redeploy, lalu perbarui nilai yang sama di backend.

## Kapasitas dan batas

Hasil uji paralel (`scripts/load_test.py`) di Docker lokal dengan **key Jina gratis**:

| Uji | Hasil |
|---|---|
| 8 request bersamaan, batas 10 | 8× 200, latensi 4,1–4,8 detik |
| 12 request bersamaan, batas 10 | 9× 200 dan **3× 503**: Jina membalas HTTP 429 saat ±10 embedding dikirim serentak, dan satu retry singkat juga ditolak |
| 6 request bersamaan, batas 2, antrean 1 detik | 2× 200 dan 4× 429 `rate_limited`, sesuai desain |

Karena itu default `MAX_CONCURRENT_REQUESTS` diturunkan menjadi **5**: request berlebih menunggu di antrean layanan ini, bukan ditolak Jina.

> **Belum terkonfirmasi:** apakah key Jina berbayar (akun HETI) menghilangkan 429 serentak. Sebelum go-live, jalankan dengan key produksi:
> ```bash
> docker exec <container> python -m scripts.load_test --n 12 --url http://amwal-rag:8000
> ```
> Bila hasilnya 12× 200 dan log tidak berisi `Jina HTTP 429`, `MAX_CONCURRENT_REQUESTS` boleh dinaikkan (mis. ke 10), lalu uji ulang.

**Arti status untuk klien:**

- **429 `rate_limited`:** layanan ini sedang penuh (semua slot terpakai dan antrean habis). Retry sekali setelah ±2 detik.
- **503 `upstream_unavailable`:** Jina/DeepSeek menolak atau tidak tersedia (rate limit, saldo habis, gangguan), atau indeks belum siap. Retry sekali; bila berulang, periksa log dan saldo.
- **504 `timeout`:** pemrosesan melewati 60 detik. Jangan retry otomatis.

**Pantau saldo:** DeepSeek (platform.deepseek.com) dan kuota Jina (dashboard jina.ai). Saldo atau kuota habis muncul sebagai 503 bagi klien; log mencatat kode HTTP dari penyedia, mis. `upstream APIRequestError (HTTP 402)` (DeepSeek menandai saldo kurang dengan 402) atau `(HTTP 429)`. Gambaran biaya: ±Rp 6–7 per pertanyaan untuk DeepSeek ditambah sedikit token Jina, dan ±370 ribu token Jina setiap kali indeks dibangun.

## Belum diverifikasi tanpa Dokploy

`docker-compose.dokploy.yml` sudah diuji dengan Docker Desktop (Docker 29.8.1, Compose v5.5.1) memakai jaringan bridge `dokploy-network` yang dibuat manual: build image, build indeks di named volume oleh user non-root (tanpa `chown`), status `healthy`, `/chat` dari container lain lewat `http://amwal-rag:8000` (200, dan 401 untuk key salah), indeks tetap ada setelah `down` lalu `up`, serta container yang berjalan tanpa indeks (`/health` 503, status `unhealthy`, `/chat` 503 tanpa panggilan API) berubah menjadi `healthy` dan melayani `/chat` begitu file indeks muncul di volume, tanpa restart. Hal-hal berikut hanya bisa dipastikan di server:

- **`env_file: .env` dari tab Environment.** Dokploy menulis variabel tab Environment ke file `.env` di samping file compose; bila tidak, container berjalan tanpa key (`/chat` 500, log startup menyebut `API_KEY belum diisi`).
- **Randomize Compose / isolated deployment harus mati.** Bila menyala, Dokploy menambahkan akhiran pada nama service, volume, dan jaringan, sehingga alamat `amwal-rag` dan volume `amwal-rag-index` berubah.
- **Persistensi volume saat redeploy.** Named volume seharusnya bertahan (perilaku standar Docker); cek bahwa setelah redeploy `/health` langsung 200 tanpa build ulang.
- **Status deploy saat container `unhealthy`** sebelum indeks ada: Dokploy mungkin menandai deploy gagal atau terus menunggu. Lanjutkan ke langkah 4 selama container berjalan.
- **Backend harus ada di `dokploy-network`** dan bisa me-resolve nama `amwal-rag`.
- **Jenis jaringan di server bisa berbeda** dari uji lokal (mis. jaringan overlay Swarm buatan Dokploy vs. bridge lokal). Bila container tidak bisa bergabung ke `dokploy-network`, sesuaikan definisi jaringan di `docker-compose.dokploy.yml`.
- **Key Jina berbayar (HETI)** belum diuji dengan beban serentak; lihat [Kapasitas dan batas](#kapasitas-dan-batas).

## Docker biasa / uji lokal

Untuk server tanpa Dokploy atau uji di laptop. Port hanya dibuka ke `127.0.0.1:8000`; bila backend ada di server lain, tambahkan proxy HTTPS (nginx, dengan `proxy_read_timeout` ≥ 70 detik).

```bash
cp .env.example .env        # lalu isi key
mkdir -p index
docker compose build
docker compose run --rm api python -m scripts.build_index   # sekali
docker compose up -d
curl -s http://127.0.0.1:8000/health
```

Container berjalan sebagai user host (`APP_UID`/`APP_GID`, default 1000) agar bind mount `./index` bisa ditulis; isi keduanya di `.env` bila `id -u` / `id -g` bukan 1000.

## Tanpa Docker (pengembangan)

Python 3.11:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
cp .env.example .env                                  # lalu isi key
python -m scripts.build_index                         # sekali
uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

Jalankan dengan satu worker: batas `MAX_CONCURRENT_REQUESTS` berlaku per proses. Alat bantu lain:

- `pytest`: seluruh tes (tanpa panggilan API sungguhan).
- `python ask.py "pertanyaan"`: tanya langsung lewat CLI.
- `python -m scripts.smoke_test`: cek Jina embed, Jina rerank, dan DeepSeek.
- `python -m scripts.load_test --n 8`: uji paralel ke server yang sedang berjalan.
- `python -m scripts.regression_test`: uji retrieval terhadap ground truth (butuh `data/eval/`, tidak ada di repo; format hasil di [data/eval/SCHEMA.md](data/eval/SCHEMA.md)).
