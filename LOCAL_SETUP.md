# Running this locally — setup guide

This project was built and verified inside a cloud sandbox (see
`README.md` and `claude/project_status.md` for the full phase-by-phase
history). This guide is for running the exact same stack on your own
machine.

## Cloning via git? Read this first.

Everything below was originally written for someone receiving this
project as a **full folder/archive copy** (data included, see below). If
you instead got this via `git clone`, you have the application code only
— `.gitignore` deliberately excludes `data/postgres_data/`,
`data/qdrant_storage/`, `data/raw/`, `data/models/`, `.venv/`, and `.env`,
none of which belong in git (binary DB volumes, a 2.5GB model file, and
secrets, respectively). Practically, that means:

- **Path A ("fast start") below will NOT work for you as written** —
  there's no pre-built data to restore, because it was never in the repo.
  Either ask whoever gave you the `git clone` link to separately send you
  a zip of their `data/postgres_data/`, `data/qdrant_storage/`, and
  `data/raw/` folders (fastest), or go straight to **Path B** and rebuild
  the pipeline from scratch (slower, but self-contained — re-fetches and
  re-indexes all 47 sources).
- You still need to `cp .env.example .env` and fill it in yourself —
  nobody's real `.env` is ever in git.
- You still need to separately run `python scripts/download_generation_model.py`
  (the 2.5GB Qwen3 GGUF isn't in git either) — this step applies whether
  you got the code via archive or git clone.

## What's in the original archive (if you received one instead of git-cloning)

- All application code: `app/` (fetcher, extraction, embedding, review,
  retrieval, generation, API), `db/migrations/`, `scripts/`, `frontend/`
  (the React chat widget).
- `data/postgres_data/`, `data/qdrant_storage/`, `data/raw/` — the
  **already-fetched, already-chunked, already-embedded** state: all 47
  real sources, 410 chunks, indexed in both Postgres and Qdrant. This
  means you do **not** need to re-run the fetcher/extractor/embedder to
  get a working system — just bring the containers up and the data is
  already there (see Path A below). None of this is in git (see above) —
  it only exists if you received a full archive/folder copy directly.
- **NOT included either way** (regenerate these locally — see steps below):
  `.venv/` (Python virtualenv), `frontend/node_modules/`, `frontend/dist/`,
  `data/models/` (the 2.5GB Qwen3 GGUF file — re-download it, don't
  transfer it), and this machine's `.env` (copy `.env.example` instead).

## Prerequisites

- **Docker Desktop** (Mac/Windows) or **Docker Engine + Compose** (Linux)
- **Python 3.11** (a different 3.x will likely work too, but this was
  built and tested on 3.11)
- **Node.js 18+** and npm, for the frontend
- **cmake + a C/C++ compiler** (Xcode Command Line Tools on Mac, `build-
  essential` on Linux, or MSVC/MinGW on Windows) — only needed if the
  plain `pip install llama-cpp-python` step below doesn't find a prebuilt
  wheel for your platform and falls back to building from source
- Optional, only if you want to re-run the fetcher or extractor from
  scratch rather than using the shipped data: `playwright install
  chromium` and a system `tesseract-ocr` install (for OCR fallback on
  scanned PDFs)

## Path A — fast start (recommended): restore the shipped data

```bash
# 1. Bring up Postgres + Qdrant, using the data already in this archive
docker compose up -d
# wait ~10s for both healthchecks to pass, then confirm:
docker compose ps

# 2. Python environment
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
cp .env.example .env
pip install -r requirements.txt
# On Apple Silicon Macs, drop the --extra-index-url CPU-wheel line's
# effect by installing torch normally first if the pinned CPU wheel
# doesn't have an arm64 build — check https://pytorch.org for the
# right install command for your platform, then re-run the pip install
# above; everything else in requirements.txt is platform-independent.

# 3. Download the self-hosted generation model (~2.5GB, one-time)
python scripts/download_generation_model.py

# 4. Confirm the shipped data is intact
python -c "
import psycopg2
from app.core.config import get_settings
conn = psycopg2.connect(get_settings().postgres_dsn)
cur = conn.cursor()
cur.execute(\"SELECT review_status, count(*) FROM chunks GROUP BY review_status\")
print(cur.fetchall())
"
# Expect: [('pending_review', 410)] — nothing pre-approved, matching how
# this project was left at the end of Phase 8.

# 5. Start the backend
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
# in a separate terminal, confirm:
curl http://127.0.0.1:8000/health

# 6. Frontend, in another terminal
cd frontend
npm install
npm run dev
# open the URL it prints (usually http://localhost:5173)
```

At this point the widget is running, but every answer will say "I don't
have approved information covering that" — because, correctly, nothing
is approved yet (see step 4). To see it actually answer something, first
approve some real content through the review API:

```bash
# List a few pending chunks
curl "http://127.0.0.1:8000/review/pending?limit=5"

# Approve one by its id (from the response above)
curl -X POST http://127.0.0.1:8000/review/chunks/<chunk_id>/approve \
  -H "Content-Type: application/json" \
  -d '{"actor": "you", "reason": "reviewing for local testing"}'
```

Or approve everything at once for local exploration (**not** how this
would work in a real deployment — a real editorial review would go
chunk-by-chunk):

```bash
curl -s "http://127.0.0.1:8000/review/pending?limit=500" \
  | python3 -c "import json,sys; [print(c['id']) for c in json.load(sys.stdin)['chunks']]" \
  | while read id; do
      curl -s -X POST "http://127.0.0.1:8000/review/chunks/$id/approve" \
        -H "Content-Type: application/json" \
        -d '{"actor": "local-dev", "reason": "bulk-approve for local testing"}' -o /dev/null
    done
```

## Path B — rebuild everything from scratch

If Path A's shipped `data/postgres_data`/`data/qdrant_storage` don't
mount cleanly on your machine (this can happen with Docker bind-mount
permission differences across hosts — if Postgres fails to start with a
permissions error, this is why), delete those two folders and rebuild
the whole pipeline instead:

```bash
rm -rf data/postgres_data data/qdrant_storage
docker compose up -d
python scripts/migrate.py
python scripts/load_source_registry.py     # loads the 47-source registry into Postgres
playwright install chromium
python scripts/fetch_all_sources.py        # re-fetches all 47 real sources
python scripts/embed_and_index_all.py      # extracts, chunks, embeds, and indexes every source
                                            # (extraction/chunking now lives inside the shared
                                            # pipeline this script calls — see app/ingestion/pipeline.py)
```

Script names confirmed against the actual `scripts/` directory in this
archive — see `README.md`'s per-phase "How to verify" sections for more
detail on each one if needed.

## Known things worth knowing before you start

- **Generation is slow on CPU.** Expect 30 seconds to ~2.5 minutes per
  answer, depending on question complexity, on modest hardware. This is
  a real, load-bearing property of the "self-hosted, no GPU required"
  design goal — not a bug. It'll be faster on a machine with more CPU
  cores (`generation_n_threads` in `app/core/config.py` defaults to 2;
  raise it to match your machine's core count) or a GPU build of
  llama-cpp-python (out of scope for this build, but llama.cpp supports
  it).
- **If `llama-cpp-python` crashes with `Illegal instruction` the moment
  it tries to load the model**, that's a CPU-instruction-set mismatch
  (this happened once in the cloud sandbox this was built in, due to
  that environment's virtualization). Rebuild from source with newer
  instruction sets disabled:
  ```bash
  CMAKE_ARGS="-DGGML_AVX512=off -DGGML_AVX512_VBMI=off -DGGML_AVX512_VNNI=off -DGGML_AVX_VNNI=off" \
    pip install llama-cpp-python==0.3.34 --no-cache-dir --force-reinstall --no-binary llama-cpp-python
  ```
  This is unlikely to be needed on a normal personal machine — it was
  specific to the sandbox's virtualized CPU.
- The frontend's `VITE_API_BASE_URL` (in `frontend/.env`) defaults to
  `http://127.0.0.1:8000` — correct for this local setup as-is.
- **Optional faster generation backend: Gemini's API.** If the CPU-only
  Qwen latency above is too slow for your use, set `GENERATION_BACKEND=gemini`
  in `.env` (default is `qwen`) plus a free API key from
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey) as
  `GEMINI_API_KEY`. This is a real trade-off, not a free upgrade: every
  question and retrieved source excerpt for that turn is sent to Google
  instead of staying fully local, which was the original design goal.
  Free-tier quota is low (a handful of requests/minute) — expect 429s
  during heavy testing; `app/generation/gemini_backend.py` retries these
  automatically with backoff, but if you're hammering it in quick
  succession you can still exhaust the daily quota. Switching back to
  `GENERATION_BACKEND=qwen` reverts to fully local at any time.

## Full-stack Docker deployment (demo / AWS Lightsail)

The whole app runs as four containers (Postgres, Qdrant, the FastAPI backend,
and an nginx-served frontend). The frontend reverse-proxies the API, so the
browser talks to ONE origin -- which keeps the login cookie and CORS simple.

```bash
cp .env.example .env          # then edit it (see below)
docker compose build          # backend build is slow: it compiles whisper.cpp
                              # and bakes the embedding + STT models into the image
docker compose up -d          # brings up all four services
# open http://localhost:${FRONTEND_PORT:-80}
```

Before a real deployment, set in `.env`:

- `LOGIN_PASSWORD` — the shared access password. Leave it empty and the whole
  app is open (fine for local dev only). When set, visitors see a login screen
  and every API route except `/health` requires the session cookie.
- `SESSION_SECRET` — a long random string that signs the session cookie.
- `POSTGRES_PASSWORD` — not the dev default.
- `CORS_ALLOWED_ORIGINS` — the real site origin (not `*`).
- `SESSION_COOKIE_SECURE=true` once served over HTTPS.
- `APP_ENV=production` — makes the backend refuse to start if any of the above
  are still unsafe.

Voice input (the mic button) uses self-hosted whisper.cpp via `/transcribe`;
nothing is sent to a cloud speech service. `WHISPER_MODEL_SIZE` picks the model
(`base.en` default, `tiny.en` lighter).

To seed demo content without the separate kb_admin review service:

```bash
docker compose exec backend python scripts/bulk_approve_demo.py
```

That approves ALL pending chunks (a demo shortcut, not real editorial review).

On the Lightsail instance, open only ports 80/443 in Lightsail's firewall. The
Postgres/Qdrant host-port bindings in `docker-compose.yml` are bound to
`127.0.0.1` for local-dev convenience and must not be internet-reachable.

## Running behind a reverse proxy / load balancer

Rate limiting is per client IP. Behind nginx, a cloud load balancer or a CDN,
the server only sees the proxy's address, so **every visitor shares one limit**
(for example 20 chat messages per minute for the whole site) unless uvicorn is
told which proxies to trust:

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 \
  --proxy-headers --forwarded-allow-ips="10.0.0.5,10.0.0.6"
```

- List only your proxy's real address(es) in `--forwarded-allow-ips` (or set
  the `FORWARDED_ALLOW_IPS` environment variable). **Never use `"*"` on a
  public service** — it lets any visitor forge `X-Forwarded-For` and dodge the
  limit.
- The proxy must set/append `X-Forwarded-For` itself.
- With more than one worker, point `RATE_LIMIT_STORAGE_URI` at a shared store
  (for example `redis://redis:6379`, which also needs `pip install redis`).
  The default `memory://` counts per worker, so N workers allow N times the
  limit.

## Backups

Reviewer decisions and the audit log live only in Postgres, and the search
index lives only in Qdrant. Nothing else can recreate them, so back up both
(the commands below assume the default container names from `docker-compose.yml`).

```bash
# Postgres: a plain SQL dump
docker exec rag_postgres pg_dump -U rag_admin -d rag_chatbot > backup_postgres_$(date +%F).sql

# Restore into an empty database
docker exec -i rag_postgres psql -U rag_admin -d rag_chatbot < backup_postgres_YYYY-MM-DD.sql

# Qdrant: stop it so files are consistent, copy the storage folder, start again
docker compose stop qdrant
cp -r data/qdrant_storage backup_qdrant_$(date +%F)
docker compose start qdrant
```

Run the Postgres dump on a schedule (cron or Task Scheduler) and keep copies
off this machine. Postgres is the source of truth for review status, so if
Qdrant is lost but Postgres is intact, re-run the indexing step to rebuild it.

## Running the tests

```bash
# Backend (from the repo root)
pip install -r requirements-dev.txt
python -m pytest
```

`tests/test_reindex_review_status.py` needs a reachable Postgres (the one from
`docker compose up -d` works). It creates and drops its own throwaway database,
and is skipped automatically if Postgres isn't reachable, so it never touches
your real data. Everything else runs without any services.

```bash
# Frontend
cd frontend
npm install
npm test        # unit tests
npm run build   # type-check + production build
npm run lint
```
