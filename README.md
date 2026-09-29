# amwal-rag

Chatbot ZISWAF (RAG) untuk amwal.site: `POST /chat` dan `GET /health`. Kontrak API ada di [API_CONTRACT.md](API_CONTRACT.md).

**Asumsi:** chatbot dan backend AMWAL berjalan di server yang sama, sehingga chatbot hanya dibuka ke `127.0.0.1:8000`. Jika berbeda server, ubah pengaturan port di `docker-compose.yml` dan tambahkan proxy HTTPS (nginx).
