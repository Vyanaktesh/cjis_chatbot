# Consulate RAG Chatbot

Self-hostable RAG chatbot over an Indian consulate's public web/PDF content
(passport, OCI, visa, surrender/renunciation, birth/death certificates,
police clearance, status tracking, community pages). Built in phases —
see the phase notes below for what exists so far.

No third-party SaaS APIs are required for the core pipeline (fetch → extract
→ chunk → embed → index → review → retrieve → generate), so this can
eventually run on restricted/offline-ish government infrastructure.

## Status: Phase 8 — Chat API + frontend

What's here:

**Phase 1 — Infrastructure**
- `docker-compose.yml` — Qdrant (vector DB) + Postgres (operational DB),
  self-hosted, with healthchecks and persistent volumes under `./data/`.
- `db/migrations/0001_init_schema.sql` — Postgres schema: `sources`,
  `source_versions`, `chunks`, `audit_log`, plus the enums
  (`service_category`, `review_status`, `source_type`, `fetch_result`) that
  back the metadata model described in the project brief.
- `app/core/config.py` — single source of truth for config, loaded from
  `.env` (see `.env.example`). Every later phase imports `get_settings()`
  from here rather than reading `os.environ` directly.
- `scripts/migrate.py` — applies pending SQL migrations, tracked in a
  `schema_migrations` table (idempotent — safe to re-run).
- `scripts/health_check.py` — connects to Postgres and Qdrant using the
  same config, and prints OK/FAIL per service.

**Phase 2 — Fetcher + source registry**
- `db/migrations/0002_source_registry_enrichment.sql` — widens
  `service_category` to the real ~14-value taxonomy used by the office's
  actual 47-source registry, and adds `sources.title` / `sources.source_group`.
- `db/seed/source_registry.json` — the real 47-source registry (VFS
  checklists + gov.in/mea.gov.in pages), plus the `not_yet_located` and
  `pending_from_office` gap lists, mirrored from the Claude project doc.
- `app/db/sources_repo.py`, `source_versions_repo.py`, `audit_repo.py` —
  CRUD for the source registry, version history, and append-only audit log.
- `app/fetcher/playwright_fetcher.py` — Playwright-based fetcher used
  uniformly for HTML and PDF: full page navigation for HTML (captures
  JS-rendered content), Playwright's request-context API for PDFs (raw
  bytes). Retries with exponential backoff, SHA-256 content hashing,
  custom User-Agent with contact info, structured JSON logging.
- `app/fetcher/robots.py` — robots.txt check (plain HTTP first, falls back
  to fetching robots.txt through Playwright if the domain blocks plain
  requests), defaults to "allow" if robots.txt is unreachable by either path.
- `app/fetcher/retry.py` — small hand-rolled retry/backoff helper.
- `scripts/load_source_registry.py` — upserts all 47 real sources into
  Postgres (idempotent, keyed on URL).
- `scripts/run_fetch_smoke_test.py` — live end-to-end smoke test: loads
  the registry, fetches a representative subset (1 VFS PDF, 1 VFS HTML
  page, 2 different gov.in HTML domains) plus one deliberately broken URL.
- `scripts/fetch_all_sources.py` — fetches every active source (all 47),
  not just the smoke-test subset. Run this before starting Phase 3, since
  chunking needs real fetched content for every source, not a handful.
- `scripts/phase2_report.py` — independent verification report (queries
  Postgres + the local filesystem directly).
- `app/fetcher/orchestrate.py` — shared "fetch one URL, persist the
  outcome" logic used by both the smoke test and the full-registry fetch,
  so the two never drift out of sync on how versions/files/audit rows get
  written.

**Phase 3 — Extraction + chunking**
- `app/extraction/blocks.py` — shared structured-text model (`Block`:
  heading/paragraph/list_item/table_row) produced by both extractors; the
  chunker only ever sees this, not raw markup/PDF bytes.
- `app/extraction/html_extractor.py` — BeautifulSoup-based: strips
  nav/footer/script/style/aside/form (site chrome), then walks the DOM in
  document order preserving heading hierarchy (h1-h6), nested lists
  (a `<li>`'s nested `<ul>/<ol>` is detached before reading its own text,
  so nested content is never duplicated into the parent item), and basic
  tables (one block per row).
- `app/extraction/pdf_extractor.py` — pdfplumber-based: groups words into
  visual lines, classifies each line as heading/list-item/paragraph using
  a font-size heuristic (vs. the document's most common body size) plus
  bullet/number-pattern matching, then merges wrapped continuation lines
  back into their parent list item/paragraph so a multi-line requirement
  stays one block. Pages with near-zero extractable text (scanned/
  image-only) are rasterized (pdf2image/poppler) and run through
  pytesseract OCR instead, tagged `source="ocr"`.
- `app/chunking/chunker.py` — section-aware: splits on heading boundaries
  first, and only falls back to a ~450-token cap within an overly long
  section — a split NEVER happens inside a single block's text, only
  between blocks, which is what guarantees a checklist requirement can't
  be cut in half. Each chunk is prefixed with its heading trail (e.g.
  "Document Checklist > Required Documents") so it's self-contained for
  retrieval.
- `app/chunking/tag_metadata.py` — the "tag metadata" pipeline step: turns
  chunker output + the Source/SourceVersion it came from into dicts shaped
  exactly like the `chunks` table's columns, ready for Phase 4 to embed
  and insert.
- `scripts/run_extraction_test.py` — runs extraction+chunking against
  every source's latest fetched version (all 47), writes full per-source
  chunk dumps to `data/chunks_preview/<source_id>.json`, and prints one
  real checklist's full chunk boundaries for manual "no split
  mid-requirement" inspection.
- `scripts/test_ocr_fallback.py` — none of the 47 real sources are scanned
  PDFs, so this builds a synthetic image-only PDF (text rendered onto a
  blank image, no real text layer) to prove the OCR fallback path actually
  triggers and actually recovers text, rather than leaving it untested.

**Phase 4 — Embedding + indexing**
- `app/embedding/bge_m3.py` — BAAI/bge-m3 wrapper (dense 1024-dim +
  sparse/lexical vectors), hardcoded to CPU (`use_fp16=False,
  devices=["cpu"]`) per the brief's modest/no-GPU requirement. Model
  downloads once from Hugging Face Hub on first use (~2.2GB, cached after)
  — no third-party API calls at inference time.
- `app/vectorstore/qdrant_store.py` — creates the `chunks` collection with
  named dense (cosine) + sparse vectors, upsert helper, filter builder
  (service_category / canonical / review_status), dense search helper.
- `app/db/chunks_repo.py` — Postgres mirror CRUD for the `chunks` table.
  Chunk IDs are derived deterministically from
  `(source_version_id, chunk_index)` via `uuid5`, so re-running the embed
  step is idempotent and the same ID is used as both the Postgres PK and
  the Qdrant point ID (as planned back in Phase 1).
- `scripts/embed_and_index_all.py` — the full Phase 4 pipeline: extract +
  chunk every source's latest version, embed every chunk, upsert to
  Qdrant, mirror-write to Postgres. Every chunk lands with
  `review_status='pending_review'` — nothing is retrievable until Phase
  5's admin review approves it.
- `scripts/query_qdrant.py` — raw similarity search directly against
  Qdrant (no retrieval-layer code yet — that's Phase 6), proving both that
  the vectors are semantically meaningful and that metadata filtering
  actually constrains results.

**Phase 5 — Admin review API**
- `app/ingestion/pipeline.py` — the extract/chunk/embed/index logic pulled
  out of Phase 4's script into a shared module, so the bulk CLI script and
  the new manual-upload API endpoint run through identical steps and can't
  drift apart (same pattern as Phase 2's `orchestrate.py`). Also where the
  brief's "content changes: don't overwrite, mark old chunks superseded,
  insert new version, re-review" behavior actually lives now — Phase 4
  hadn't implemented this yet since nothing had triggered a real version
  bump until Phase 5's manual upload made it testable. `index_records()`
  does every Postgres write first and every Qdrant write last — see the
  "bug found and fixed" note below for why that ordering matters.
- `app/review/service.py` — the approve/reject operation: updates
  `review_status` in both Postgres and Qdrant, writes an `audit_log` entry
  with an actor (and optional reason).
- `app/review/diff.py` — "view diff against previous version": compares a
  source's two most recent versions by content_hash-set membership (chunk
  boundaries can shift between versions, so hash comparison — not index
  alignment — is what tells genuinely new/removed text from same-text-
  renumbered), plus a human-readable unified diff of the full document.
- `app/api/main.py` — FastAPI app: `GET /health`, `GET /sources` (per-
  source chunk-count overview), `GET /review/pending` (filterable by
  `service_category`/`source_id`, paginated), `GET /review/chunks/{id}`,
  `GET /review/sources/{id}/diff`, `POST /review/chunks/{id}/approve`,
  `POST /review/chunks/{id}/reject`.
- `app/api/upload.py` — `POST /upload`: manual PDF upload through the
  exact same pipeline as the bulk fetcher. Re-uploading the same `url`
  with changed content creates a new version and supersedes the old one,
  identical to a re-fetch; re-uploading identical content is a no-op
  (`unchanged: true`), matching the fetcher's content-hash dedup.
- `db/migrations/0001` already had `review_status` (`pending_review` /
  `approved` / `rejected` / `superseded`) and `superseded_by` — Phase 5 is
  the first phase to actually exercise all four states.

**Update, kb_admin build:** `GET /sources`, `GET/POST /review/*`, and
`POST /upload` have since moved out of this app's `app/api/main.py` into
the separate `consulate-kb-admin` service (port 8100), behind HTTP Basic
auth on every route. The reasoning above (identical pipeline, no
Postgres/Qdrant drift) still holds — kb_admin reuses these exact same
`app/ingestion/pipeline.py` / `app/review/*` modules in-process via its own
sys.path bridge, it just doesn't expose them unauthenticated on this
process's port anymore. `app/api/upload.py` and `app/api/schemas.py` were
deleted; see `consulate-kb-admin/kb_admin/api/documents.py` and
`review.py` for where this logic lives now.

**Phase 6 — Hybrid retrieval**
- `app/vectorstore/qdrant_store.py` — `hybrid_search()`: Qdrant's native
  Query API, running the dense leg and the sparse leg as independent ANN
  searches (each pre-narrowed by the same filter — filtering happens
  *inside* each leg's candidate search, not as a post-filter after
  fusion), fused with Reciprocal Rank Fusion (`FusionQuery(fusion=RRF)`).
  RRF combines by rank position rather than raw score, since dense cosine
  similarity and sparse lexical scores aren't on comparable scales.
  `build_filter()` extended with `source_id`, `jurisdiction`, and
  `applicant_variant` (array-payload fields — a `MatchValue` condition
  against them matches if the array *contains* that value, e.g. a chunk
  tagged `["adult", "minor"]` matches a filter for either).
- `app/retrieval/retriever.py` — `search()`: embeds the query (BGE-M3),
  runs `hybrid_search`, and hard-enforces `review_status='approved'` on
  every call — this is **not** a caller-adjustable filter. Nothing
  pending/rejected/superseded is meant to reach an end user, and this is
  the first phase where that rule is actually load-bearing rather than
  just a schema value nothing reads yet.
- `app/api/main.py` — `GET /search?q=...&limit=...&service_category=...`
  (plus `canonical`, `source_id`, `jurisdiction`, `applicant_variant`): a
  thin HTTP wrapper over `retriever.search()`, for manual testing now and
  for Phase 7/8 to call once generation/chat exist.

**Phase 7 — Generation**
- `app/generation/qwen_backend.py` — `QwenGenerator`: self-hosted Qwen3
  via `llama-cpp-python` (llama.cpp), running a quantized GGUF file
  entirely on CPU — no network call, no third-party API. **Model size
  deviates from the brief on purpose:** the brief specified Qwen3-8B, but
  this build sandbox's ~2 CPU cores / ~8GB RAM made an 8B GGUF (~5GB file)
  too tight to reliably verify alongside Postgres + Qdrant already
  running — confirmed live (see "known issue" note below), not assumed.
  User explicitly chose **Qwen3-4B** (~2.5GB, `Q4_K_M`) as a fully
  self-hosted, CPU-only substitute over the alternative of using a
  third-party API (Claude) as a placeholder. Swapping to 8B (or anything
  larger) on real deployment hardware is a one-line config change
  (`GENERATION_MODEL_PATH`) plus a re-download — nothing in this module
  is 4B-specific.
- `scripts/download_generation_model.py` — one-time setup: downloads the
  GGUF from Hugging Face Hub into `data/models/` (gitignored — large
  binary, not repo content).
- `app/generation/prompt.py` — builds a strict grounded-RAG chat prompt:
  numbered source excerpts + the question, with a system prompt that
  requires every factual claim to carry a bracket citation, forbids
  outside knowledge, and requires an explicit "I don't have information
  covering that" when the sources don't answer the question. Also strips
  Qwen3's `<think>...</think>` reasoning block (suppressed via the
  `/no_think` suffix Qwen3's chat template recognizes) and extracts which
  `[n]` citations the answer actually used.
- `app/generation/service.py` — `answer_question()`: runs Phase 6
  retrieval (approved-only), and — this is deliberate — **skips calling
  the model entirely when retrieval finds zero chunks**, returning a
  fixed honest "nothing approved covers that" response instead of asking
  the model to correctly decline on an empty prompt.
- `POST /generate` added to the FastAPI app: `{query, limit,
  service_category, canonical, source_id, jurisdiction,
  applicant_variant}` → `{answer, grounded, citations, retrieved_count}`.

**Phase 8 — Chat API + frontend**
- `POST /chat` added to the FastAPI app (`app/api/main.py`) — the
  frontend-facing endpoint. It's a thin wrapper over the same
  `answer_question()` pipeline as `/generate`, shaped for a chat widget:
  `{message, session_id?, history?}` → `{session_id, answer, grounded,
  citations, retrieved_count}`. `CORSMiddleware` was added so the React
  dev server (a different origin) can call it directly.
- **Conversation memory — a real design change made mid-phase.** The
  original plan (matching Phase 7's single-turn pattern) was to answer
  every message independently, with no memory at all, purely because of
  this build's slow CPU-only generation (30-150s/turn already). That
  landed, then the user asked for genuine memory of a few prior turns —
  standard practice for a usable chatbot — so the design changed to a
  **bounded sliding window**: `app/generation/prompt.py`'s
  `_trim_history()` keeps at most the last 5 turns, further trimmed by an
  approximate character budget (~2400 chars, roughly 600 tokens) dropping
  the *oldest* turns first, so a couple of long messages can't silently
  evict everything else. Those turns are inserted into the model's chat
  messages as real alternating user/assistant turns (not mashed into one
  text blob), and the system prompt gained an explicit rule: prior turns
  are for resolving what the user means, not a source of facts — every
  claim in a new answer must still cite the *current* turn's numbered
  sources, since citation numbers reset every turn.
- **Retrieval also needed a cheap follow-up fix**, not just generation:
  a literal follow-up like "does it need to be notarized?" has no
  retrievable keywords on its own. Calling the LLM to rewrite the query
  first would double every turn's already-long latency, so
  `app/generation/service.py`'s `_retrieval_query()` instead prepends
  just the single most recent prior user message to the current query
  before embedding — a cheap heuristic, not a rewrite, but enough to pull
  "OCI" back in for a bare "does it need to be notarized?" follow-up (see
  verification below — this was tested for real, not assumed to work).
- **Frontend**: `frontend/` — a Vite + React + TypeScript app.
  - `frontend/src/chat/backendAdapter.ts` — bridges
    [`@assistant-ui/react`](https://github.com/assistant-ui/assistant-ui)
    (chosen after actually comparing options — see below) to `POST
    /chat`: extracts the latest message and a same-sized sliding window
    of prior turns from assistant-ui's own message state, and turns the
    backend's `citations` into assistant-ui `source` message parts
    (clickable source chips) rather than stuffing them into plain text.
  - `frontend/src/chat/ChatWidget.tsx` — the floating widget shell: a
    circular launcher button, an expandable panel with a blue header
    (title/subtitle + close button), a scrollable message area (built on
    assistant-ui's `ThreadPrimitive`/`MessagePrimitive` components,
    styled with Tailwind to match the reference blue/white consulate chat
    widget look), and a rounded composer input with a send button.
    Includes a "still thinking" indicator (`ThreadPrimitive.If running`)
    since a single turn can take up to ~2 minutes on this hardware —
    without it the widget would look frozen, which matters a lot more
    here than it would with a hosted, fast API.
  - Nothing is persisted client-side either: assistant-ui's default
    in-memory runtime holds the transcript only for the life of the
    browser tab (no localStorage/sessionStorage/cookies), which is what
    makes "session-scoped conversation, no PII storage beyond the
    session" true on both ends without a retention policy to trust.
- **UI library choice — actually compared, not assumed:** checked real
  npm download counts for two options before building anything:
  `@chatscope/chat-ui-kit-react` (~246k/month, a general messaging-app
  component kit) vs `@assistant-ui/react` (~6.1M/month, purpose-built for
  LLM chat UIs, actively maintained). Went with assistant-ui for its
  much larger community and closer fit, using its low-level primitives
  (`ThreadPrimitive`, `MessagePrimitive`, `ComposerPrimitive`) rather than
  a prebuilt theme, since the brief was to match a specific existing
  blue/white consulate widget design, not assistant-ui's own look.

**Known environment quirk (documented, not a code defect):** in this build
sandbox, Playwright's full-page navigation (`page.goto`) gets
`net::ERR_CONNECTION_RESET` on every domain — including totally unrelated
ones like `example.com` — because the sandbox's outbound egress proxy
doesn't work with a real Chromium browser process the way it does with
plain HTTP clients (confirmed via a minimal reproduction, with and without
explicit proxy configuration). Playwright's request-context API (used for
PDFs) is unaffected. The fetcher handles this by falling back to the
request-context API for HTML too when `page.goto` fails for a network
reason (not a genuine HTTP error) — logged clearly as a fallback, since it
means no JS execution for that fetch. **You should re-verify `page.goto`
in your actual target environment**, where this restriction is very
unlikely to apply — full-page navigation is what real deployments need
for any source that renders content client-side.

## How to verify Phase 1

From the repo root:

```bash
# 1. Configure environment (defaults are fine for local dev)
cp .env.example .env

# 2. Bring up Qdrant + Postgres
docker compose up -d
docker compose ps        # both should report "healthy" / "Up" within ~10s

# 3. Python env + deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Apply the Postgres schema
python scripts/migrate.py
# Expect: "apply  0001_init_schema.sql ..." then "OK  0001_init_schema.sql"
# Re-running should print "skip ... (already applied)" — confirms idempotency.

# 5. Run the health check
python scripts/health_check.py
```

Expected output from the health check:

```
Environment: development

[Postgres] OK - connected to 'rag_chatbot' at localhost:5432
[Postgres] tables present: audit_log, chunks, schema_migrations, source_versions, sources
[Qdrant]   OK - connected at localhost:6333
[Qdrant]   collections present: (none yet)

ALL OK
```

Optional deeper check — inspect the schema directly:

```bash
docker exec rag_postgres psql -U rag_admin -d rag_chatbot -c "\dt"
docker exec rag_postgres psql -U rag_admin -d rag_chatbot -c "\d chunks"
```

This exact sequence was run in the build environment (fresh containers,
fresh venv) and passed: both containers came up healthy, the migration
applied cleanly, re-running it correctly skipped, and the health check
printed `ALL OK` with all five tables (`sources`, `source_versions`,
`chunks`, `audit_log`, `schema_migrations`) present.

## How to verify Phase 2

Assumes Phase 1 is already up (`docker compose up -d`, venv active,
`scripts/migrate.py` run through `0002_source_registry_enrichment.sql`).

```bash
pip install -r requirements.txt   # picks up playwright==1.56.0
python -m playwright install chromium   # only if Chromium isn't already installed

# 1. Load the full 47-source registry into Postgres (safe to re-run)
python scripts/load_source_registry.py

# 2. Live fetch smoke test: registers the registry, then really fetches a
#    VFS PDF, a VFS HTML page, two different gov.in HTML domains, and one
#    deliberately broken URL that isn't a registered source at all.
python scripts/run_fetch_smoke_test.py

# 3. Independent verification report (queries Postgres + disk directly,
#    doesn't just trust step 2's own printout)
python scripts/phase2_report.py
```

Expected from the smoke test: 4 successes, 1 failure (the broken URL),
ending with `4 succeeded, 1 failed (1 failure expected — the broken URL)`.

Expected from the report: `sources registered: 47 (expected 47)`,
`source_versions created: 4` (one per real fetch — re-running the smoke
test again should NOT add duplicates, since an unchanged content_hash
against the latest version logs a `fetch_no_change` audit event instead of
creating a new version), an `audit_log` breakdown showing `fetch_attempt`,
`fetch_success`, and `fetch_failure` counts, the broken URL's
`fetch_failure` entry with its actual HTTP 404 error message attached, and
every raw file present on disk with `on_disk=True`. Ends with `ALL OK`.

This exact sequence was run in the build environment against the real
target domains (`services.vfsglobal.com`, `indiainatlanta.gov.in`,
`ociservices.gov.in`) and passed: 47 sources registered spanning 14
service categories, 4 real fetches succeeded (247KB PDF + three HTML
pages up to 509KB) with SHA-256 hashes and files verified on disk, and the
broken URL retried 3 times with exponential backoff before logging a
clean `fetch_failure` — the run did not crash.

## Fetching all 47 sources (needed before Phase 3)

The smoke test above only exercises 4 of the 47 registered sources.
Phase 3's chunker needs to run against real fetched content for every
source, so fetch everything before moving on:

```bash
python scripts/fetch_all_sources.py
python scripts/phase2_report.py
```

Expected: `total sources: 47`, `succeeded: 47 (47 new/changed, 0
unchanged)` on a first run, `failed: 0`, and the report showing 47
`source_versions` rows each with `on_disk=True`.

This was run against the real registry in the build environment: all 47
sources fetched successfully (no failures), ~14MB of raw content saved
under `data/raw/<source_id>/v<version>/`.

**Bug found and fixed during this run:** the first version of the fetcher
stored each source's raw file at a path keyed only by source ID (no
version number), so when a source's content changed on a re-fetch, the
new version's file silently overwrote the old one on disk — even though
Postgres still had a `source_versions` row for the old version pointing
at that now-clobbered path. Since the brief is explicit that versioning
must preserve "what changed and when," this was a real correctness bug,
not cosmetic. Fixed by nesting raw storage under
`data/raw/<source_id>/v<version>/<filename>` so every version's bytes are
independently preserved. Verified after the fix: re-fetching a source
whose HTML changed (two of the gov.in pages embed a volatile per-request
token, so their raw HTML hash changes on every fetch even though the
meaningful content doesn't) correctly created a new version file
alongside the old one, both intact on disk.

**Also worth knowing:** because Phase 2 hashes raw content (not yet
extracted text — see the note on this in "Notes for later phases" below),
a couple of the gov.in pages will look like they have "new" versions on
every re-fetch even when nothing meaningful changed, purely because of
embedded session tokens/nonces in the raw HTML. This is expected and is
exactly the reason the brief calls for hashing the extracted region once
Phase 3 exists — extraction should strip that volatility out.

## How to verify Phase 3

Assumes all 47 sources have been fetched (`scripts/fetch_all_sources.py`).

```bash
pip install -r requirements.txt   # picks up beautifulsoup4, pdfplumber, pytesseract, pdf2image
# tesseract-ocr and poppler-utils (pdftoppm) must be installed as system
# packages — both were already present in the build sandbox; on Debian/
# Ubuntu: apt-get install -y tesseract-ocr poppler-utils

# 1. Extract + chunk every source's latest fetched version
python scripts/run_extraction_test.py

# 2. Prove the OCR fallback path works (no real source is a scanned PDF,
#    so this builds a synthetic image-only PDF to test against)
python scripts/test_ocr_fallback.py
```

Expected from step 1: `sources processed: 47 / 47`, a per-source line for
each showing block/chunk counts, a "Spotlight" section printing every
chunk boundary for one real OCI checklist PDF (the most common applicant
profile), and full JSON chunk dumps written to
`data/chunks_preview/<source_id>.json` for manual inspection of any other
source. **Manually verify the spotlight output yourself**: read through
the numbered requirements and confirm none of them is cut off partway
through at a `--- chunk N ---` boundary — a boundary should only ever fall
between two requirements, never inside one.

Expected from step 2: 4 checks all `PASS`, ending `ALL OK`.

This exact sequence was run in the build environment and passed: all 47
sources produced 410 chunks total, and the spotlight OCI checklist's 19
numbered requirements (plus a 22-item lettered sub-list) were manually
verified to each land whole inside one chunk — including one item (a
name-change documentation requirement with five worked examples) long
enough to exceed the ~450-token cap on its own, which was still kept
intact rather than truncated, exactly as designed. The OCR fallback test
correctly recovered "Document Checklist" and all three list items from a
synthetic scanned PDF, with one single-word OCR misread ("Iwo" for "Two")
— an inherent OCR accuracy limitation, not a code defect.

**One source produced zero chunks, and that's expected, not a bug:**
"Provider application tracking, United States"
(`services.vfsglobal.com/usa/en/ind/track-application`) turned out to be
a JS-only Nuxt.js single-page-app shell — its raw HTML is just a loading
spinner and `<script>` tags, with no server-rendered text at all. This is
the direct, visible consequence of Phase 2's known sandbox limitation
(`page.goto` doesn't execute JS here — see the environment quirk note
above): in a real deployment where `page.goto` works normally, this page
would render its actual tracking-status content instead of the empty app
shell, and would then chunk normally. Re-fetching this specific source
once `page.goto` is confirmed working in the target environment should be
the first thing checked before Phase 4 embeds it.

**Follow-up investigation after Phase 4 (post-410-chunks review):** dug
into this specific source further to see whether it was genuinely
unfixable in this sandbox, rather than assuming so. Confirmed three things
directly: (1) `page.goto` fails with `net::ERR_CONNECTION_RESET` against
this URL even with the same explicit proxy config the fetcher already
passes at both launch and context level — so this isn't a missed proxy
setting; (2) the same failure reproduces against a totally unrelated
domain (`example.com`), confirming this sandbox's egress proxy breaks
Chromium browser-level navigation generally, not just for this one
domain; (3) the raw HTML has zero embedded SSR/state JSON of any kind
(no `__NUXT__`, no `__NUXT_DATA__`, no inline `application/json` script) —
this app is pure client-side-rendered with nothing statically embedded to
mine with a smarter parser, unlike some SPA frameworks that inline a
state blob even before hydration. So there is no way to recover this
page's real content from within this sandbox short of an environment
where `page.goto` actually executes JS. Deliberately did **not** paper
over the gap by hand-writing placeholder chunk text attributed to this
URL — inventing "what the page probably says" and feeding it into a
government-facing accuracy pipeline is exactly the kind of unverified
content this project's review/versioning workflow exists to keep out, even
hedged behind `review_status='pending_review'`. The two legitimate paths
forward are (a) re-fetch this one source once running somewhere
`page.goto` works, or (b) have an actual admin add real supplementary
content (e.g. "how to check your application status") through Phase 5's
planned manual-entry pathway once it exists.

Also fixed a smaller, related gap found during this investigation:
`used_request_fallback` was already tracked on `FetchResult` but was
never written into `audit_log`'s `fetch_success`/`fetch_no_change` detail
payloads (`app/fetcher/orchestrate.py`), so diagnosing *which* sources
were captured via the no-JS fallback required manually inspecting raw
HTML byte-by-byte, exactly as done here. Both event types now log
`used_request_fallback`, so a query like `SELECT entity_id, details FROM
audit_log WHERE details->>'used_request_fallback' = 'true'` will surface
every fallback-captured source going forward without needing to eyeball
each one.

**A real bug was found and fixed during the spotlight inspection:** the
PDF extractor's line-merging logic only merged consecutive PARAGRAPH
lines, so when a checklist requirement wrapped onto a second visual line
with no bullet/number marker, that continuation became a floating,
separately-classified PARAGRAPH block instead of part of its parent list
item — risking the continuation landing in a *different* chunk than the
requirement it belonged to, which is precisely the "split mid-requirement"
failure this phase is supposed to prevent. Fixed by also merging an
unmarked PARAGRAPH into an immediately preceding LIST_ITEM on the same
page. Verified after the fix by re-running the spotlight check: multi-line
requirements (e.g. item 14, item 19's sub-bullets) now read as one
complete, correctly-joined block.

## How to verify Phase 4

Assumes Qdrant + Postgres are up and all 47 sources have been fetched.

```bash
pip install -r requirements.txt   # picks up torch (CPU build) + FlagEmbedding
# First run downloads BAAI/bge-m3 from Hugging Face (~2.2GB, one-time)

# 1. Embed every source's chunks and index them (Qdrant + Postgres)
python scripts/embed_and_index_all.py

# 2. Raw similarity search directly against Qdrant, with metadata filters
python scripts/query_qdrant.py
```

Expected from step 1: model loads in well under a minute (after the
one-time download), then `Embedding 410 chunks...` followed by a summary
ending `all chunks inserted with review_status='pending_review'`.

Expected from step 2: an unfiltered query about OCI application documents
returns OCI checklist chunks; the same query re-run with a
`service_category=oci` filter returns only `oci` chunks; re-run again with
`service_category=visa` (deliberately mismatched with the query's actual
topic) returns only `visa` chunks anyway, proving the filter genuinely
constrains results rather than just correlating with what the query
already favors; a `canonical=true` filter returns only gov.in/mea.gov.in
chunks; and a combined filter (`service_category=status_tracking AND
canonical=true`) returns only chunks matching both. 5/5 checks `PASS`,
ending `ALL OK`.

This exact sequence was run in the build environment: 410 chunks (from 46
of the 47 sources — see Phase 3's note on the one JS-shell source that
produced zero chunks) were embedded in ~5 minutes on this sandbox's 2 CPU
cores (no GPU), then indexed into both Qdrant (`points_count: 410`) and
Postgres (`chunk_count: 410`, confirmed matching). The verification query
("What documents do I need to submit an OCI application as an adult?")
returned genuinely relevant OCI checklist content unfiltered (top score
0.72), and every one of the 5 filter-correctness checks passed — including
the deliberately-adversarial visa-filter-on-an-OCI-query test, which is
the one that actually proves filtering works rather than just looking
plausible.

**Worth noting for Phase 5's review workflow:** the `canonical=true`
results for the OCI query directly surfaced both sides of the item
46/47 conflict flagged back in Phase 2 (the "OCI Miscellaneous FAQs"
government portal page and the separate "OCI Reissuance Clarification"
page, both canonical, both about the same re-issue rule) — a live
demonstration of exactly the retrieval-time ambiguity that conflict flag
was warning about, and a good real test case to use once the review UI
exists.

**Bug found and fixed after the initial Phase 4 run:** `to_qdrant_payload()`
in `scripts/embed_and_index_all.py` writes a `used_ocr` field into every
Qdrant point's payload (whether the chunk's text came from OCR rather than
a native PDF text layer — lower-confidence content a reviewer should be
able to see), but the Postgres `chunks` table had no matching column at
all, so the "system of record" mirror was silently missing metadata the
vector store had — a real violation of this project's Postgres-as-source-
of-truth design. Fixed by adding migration
`db/migrations/0003_chunks_used_ocr.sql` (`used_ocr BOOLEAN NOT NULL
DEFAULT false`), wiring it through `chunks_repo.upsert_chunk()`'s INSERT/
ON CONFLICT clause, and setting it in `embed_and_index_all.py`'s
`db_record`. Re-ran the embed/index pipeline (idempotent thanks to
deterministic chunk IDs — it updated all 410 existing rows in place rather
than duplicating them) and confirmed both stores now agree exactly:
Postgres shows `(False, 409), (True, 1)` for `used_ocr` with zero NULLs,
and the single `true` row (the OCR-fallback page of the employment visa
checklist PDF, chunk_index 14) matches the one Qdrant point with
`used_ocr=true` by ID. Re-ran `query_qdrant.py` afterward as a regression
check — all 5/5 checks still `PASS`.

## How to verify Phase 5

*(Historical record: these commands were run against `app/api/main.py` on
port 8000 when `/sources`, `/review/*`, and `/upload` still lived there.
That surface has since moved to `consulate-kb-admin` on port 8100 — see the
"Update, kb_admin build" note above. To re-run an equivalent check today,
swap the base URL to `http://127.0.0.1:8100` and add
`-u admin:<ADMIN_PASSWORD>` (HTTP Basic auth) to each `curl` call; the
request/response shapes are unchanged.)*

```bash
pip install -r requirements.txt   # picks up fastapi/uvicorn/python-multipart

# 1. Start the API
uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# 2. Liveness + overview
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/sources

# 3. List pending chunks (optionally ?service_category=oci&source_id=...)
curl "http://127.0.0.1:8000/review/pending?limit=5"

# 4. Approve / reject a chunk
curl -X POST http://127.0.0.1:8000/review/chunks/<chunk_id>/approve \
  -H "Content-Type: application/json" -d '{"actor": "you@example.com", "reason": "looks correct"}'

# 5. Manual PDF upload (creates a source if `url` is new, or a new version
#    + supersedes the old chunks if `url` already exists with different content)
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@some.pdf" -F "url=manual://some-identifier" \
  -F "service_category=oci" -F "canonical=false" -F "actor=you@example.com"

# 6. Diff a source's latest version against its previous one (409 if it
#    only has one fetched version so far — true for 46 of the 47 today)
curl http://127.0.0.1:8000/review/sources/<source_id>/diff
```

This exact sequence was run in the build environment against the real 410
approved-pending chunks, not a toy dataset. `/sources` correctly showed
all 47 real sources including the JS-shell source at `total_chunks: 0`
(not `1` — see the bug note below). Approve/reject were exercised on two
real OCI-checklist chunks; both correctly changed `review_status` in
Postgres AND in Qdrant's payload, and both decisions were written to
`audit_log` with the actor and reason. Since none of the 47 registered
sources have a second fetched version yet, the versioning/supersede/diff
path was verified with a genuine end-to-end test instead of a mocked one:
a real two-page PDF was built (via `wkhtmltopdf`, so it has an actual
native text layer, not an image), uploaded through `POST /upload` as v1,
then re-uploaded with one requirement's wording changed and a new item
added as v2. Result: v2 created 2 new `pending_review` chunks, the 2 v1
chunks were correctly moved to `review_status='superseded'` in both
Postgres and Qdrant, and `GET /review/sources/{id}/diff` correctly
reported `added_chunk_indices: [1]`, `removed_chunk_indices: [1]`
(same position, different content_hash), 1 chunk unchanged, and a unified
diff that showed exactly the real wording change and the added item —
nothing fabricated, all derived from the actual re-embedded text. The
test source was deleted afterward (Postgres cascade + explicit Qdrant
point cleanup) so it doesn't linger alongside the real 47-source dataset.

**Three real bugs found and fixed while building/testing this phase:**

1. `mark_superseded()`'s `UPDATE ... WHERE id = ANY(%s)` passed a plain
   Python list of strings, which Postgres infers as `text[]` — but `id`
   is `uuid`, so every real attempt to supersede an old version's chunks
   failed with `operator does not exist: uuid = text`. Only surfaced once
   a genuine second version existed to trigger the supersede path (i.e.
   during the versioning test above, not before). Fixed with an explicit
   `::uuid[]` cast.
2. `counts_by_source()`'s `LEFT JOIN chunks` used `count(*)`, which counts
   the joined row even when every `chunks.*` column is NULL (no matching
   chunk) — so the one source with zero chunks would have shown
   `total_chunks: 1` instead of `0` through the `/sources` endpoint. Fixed
   to `count(c.id)`, which correctly returns 0 for a NULL join.
3. **Cross-store atomicity gap (the more interesting one):** Qdrant isn't
   part of the caller's Postgres transaction, so when bug #1 caused an
   exception in the *Postgres-only* supersede step — reached *after*
   `index_records()` had already upserted the new version's points into
   Qdrant — the outer transaction rolled back Postgres cleanly but Qdrant
   kept the writes it had already made, since nothing tells Qdrant to
   undo them. That left 2 real orphaned Qdrant points with no matching
   Postgres row (`qdrant.count()` read 412 against Postgres's 410).
   Reordered `index_records()` in `app/ingestion/pipeline.py` to do every
   Postgres write first and the Qdrant upsert last, so a Postgres-side
   failure now happens before Qdrant is touched at all. This closes the
   specific failure mode that actually occurred; it isn't full
   cross-store atomicity (a Qdrant-side failure after Postgres writes are
   staged, but before the transaction commits, is still a theoretical
   gap) — true 2PC/outbox handling across two different stores is out of
   scope for a "minimal" phase, and is noted as a known limitation rather
   than silently assumed away. Re-ran the exact versioning test that
   originally triggered this (fresh source, v1 then v2 upload) after the
   fix: Qdrant and Postgres both ended at 414/414, and the 2 stray orphan
   points from the original failure were separately identified (by
   diffing Qdrant point IDs against Postgres chunk IDs) and deleted.
4. A performance regression, not a correctness bug, introduced by an
   earlier draft of this same refactor: pulling shared logic into
   `embed_and_index_version()` made the bulk script call `embedder.embed()`
   once per source (47 separate model calls) instead of once for all 410
   chunks together, and a full re-run didn't finish inside a 10-minute
   timeout (versus ~5 minutes before). Fixed by splitting the pipeline
   into `extract_and_chunk()` (per source, cheap) and `index_records()`
   (Postgres/Qdrant writes, given already-embedded records), with the
   bulk script batching every chunk from every source through one
   `embed()` call and only calling `index_records()` per source — restored
   the original ~5-minute, single-batch performance while keeping the
   shared-pipeline architecture. The single-document upload endpoint still
   uses `embed_and_index_version()` (embed-then-index in one call), which
   is the right shape there since an upload is always exactly one document.

## How to verify Phase 6

```bash
# 1. Start the API (if not already running)
uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# 2. Hybrid search, restricted to approved content only
curl "http://127.0.0.1:8000/search?q=OCI+application+documents&limit=8"

# optional filters — all AND'd with the always-on review_status=approved
curl "http://127.0.0.1:8000/search?q=OCI+application+documents&canonical=true"
curl "http://127.0.0.1:8000/search?q=OCI+application+documents&applicant_variant=adult"
```

All 410 chunks were still `pending_review` going into this phase (Phase 5
never approved anything for real — those test decisions were reverted).
Since there was no approved content yet to search over, real-world
verification meant creating some: approved 15 real chunks through the
actual `/review/chunks/{id}/approve` endpoint (all 12 chunks of one real
OCI checklist, "New OCI, adult naturalised non-US national," plus 3
chunks from a real canonical gov.in OCI FAQ page) and rejected 2 real
chunks, leaving the other 393 untouched at `pending_review`. This wasn't
mocked data — it's the actual registry content, actually re-embedded in
Phase 4, run through the actual review API built in Phase 5.

With that real approved set in place: the same OCI query used back in
Phase 4 correctly returned only chunks from the 15 approved ones, ranked
sensibly (the two most specific "current passport / legal status"
requirement chunks scored highest). The `canonical=true` filter correctly
returned exactly the 3 approved gov.in FAQ chunks and nothing else;
`applicant_variant=adult` correctly returned exactly the 12 approved
adult-checklist chunks (proving the array-containment match works, not
just an accidental equality match on a single-element array); a
deliberately mismatched `service_category=visa` filter correctly returned
**zero** results, since none of the approved chunks are visa-category —
proof the filter genuinely excludes rather than just correlating with
what the query already favors.

**The adversarial test that actually matters for this phase:** three
chunks from the *pending* "OCI, minor foreign national" checklist —
about parental authorization forms and naturalization certificates for
minor applicants — were deliberately used to word a query
("Parental Authorization Form for minor OCI applicant naturalization
certificate") that is about as close a semantic match to *pending*
content as this dataset has. All 10 results still came back from the
*approved* set only; none of the 3 pending minor-checklist chunks
leaked in, even though they were plausibly the single best semantic
match available. This is what actually proves `review_status='approved'`
is enforced inside the ANN search itself (via the Prefetch-level filter),
not just applied as an afterthought that a strong enough semantic match
could route around.

Also compared plain `dense_search` against `hybrid_search` on a query
built around one exact, distinctive phrase from a real chunk ("FEDEX
PREPAID LABELS ARE NOT ACCEPTABLE courier requirements") — both correctly
surfaced that exact chunk first, and the hybrid result's RRF-fused scores
(`1.0, 0.5, 0.5, 0.35, 0.26...`) show the classic reciprocal-rank shape,
confirming fusion is actually running rather than silently falling back
to one leg.

The 15 approve / 2 reject decisions used for this test were made by me
(as a placeholder actor) purely to have real approved content to search
against — not a genuine editorial review by an actual person. All 17
were reverted back to `pending_review` in both Postgres and Qdrant after
verification (confirmed via a fresh count: 410/410 `pending_review`, 0
`approved`), so the real dataset is untouched and ready for genuine
review whenever that actually happens.

## How to verify Phase 7

```bash
# 1. One-time: download the generation model (~2.5GB, gitignored)
python scripts/download_generation_model.py

# 2. Start the API (if not already running)
uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# 3. Ask a real question against the real (small) approved set from Phase 6
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What documents does an adult need for a new OCI application?"}'
```

Two real bugs/version issues were found and fixed getting this far, not
assumed away:

- `llama-cpp-python==0.3.4` (the version pinned at Phase 4's install time,
  before Qwen3 existed) failed with `ValueError: Failed to load model from
  file` against the downloaded GGUF, even though the file's size matched
  the expected 2,497,280,256 bytes exactly (ruling out a corrupt
  download). Root cause: 0.3.4 predates llama.cpp's Qwen3 architecture
  support (Qwen3 released ~April 2025, after 0.3.4). Fixed by upgrading to
  `llama-cpp-python==0.3.34`, which loaded and generated correctly.
- Qwen3 defaults to a verbose `<think>...</think>` reasoning block before
  its actual answer — first test (50-token cap) came back slow (11.1s)
  and truncated mid-thought, before ever reaching the real answer.
  Confirmed via the model's own Hugging Face card that appending the
  literal suffix `/no_think` to the user message suppresses this; retested
  and got an empty think block plus a ~5x faster response (2.2s for the
  same short test). `strip_thinking()` in `prompt.py` still strips the
  tag defensively either way, in case a future prompt omits the suffix.

Three real tests were run through the actual running HTTP API (not
mocked), using the same 15-approved / 393-pending dataset state left over
from Phase 6:

1. **Positive/grounded test** — a real question about the approved OCI
   adult checklist returned a correctly cited answer, bracket numbers
   `[1][2][4][5]` each mapping to a real `chunk_id`/`source_url` from the
   retrieved set, with 2 of the 6 retrieved chunks correctly left
   uncited (they were retrieved but not actually relevant enough to use).
   Took 2m30.805s end-to-end on this sandbox's 2 CPU cores (see latency
   note below) — a real, complete round trip through retrieval and
   generation together, not a stub.
2. **Code-level short-circuit test** — a filter combination guaranteed to
   match zero approved chunks (`service_category=visa`, and there is no
   approved visa content) returned the fixed `NO_CONTEXT_ANSWER` response
   in 0.63s, with **no model call made at all** — confirmed via the log
   showing no `qwen_backend` load/generate lines for that request. This is
   a stronger correctness guarantee than trusting the model to decline on
   an empty context: it's structurally impossible for this path to
   hallucinate, since it never reaches the model.
3. **Adversarial grounding test** — asked "What is the capital of
   France?" with no filters. Retrieval returned 4 weakly-related OCI
   chunks (there's no France-capital content in the dataset, so hybrid
   search returned its best available — irrelevant — matches). The model
   correctly declined to answer from its own pretrained knowledge, cited
   nothing, and pointed back at what the sources actually cover, in
   33.6s. This is the test that actually matters for a RAG system: strong
   general-knowledge pull on an easy factual question, held in check by
   the system prompt's "sources only" rule.

**Real observed CPU-only latency** (2 CPU cores, Qwen3-4B-Q4_K_M,
`n_ctx=4096`, `generation_max_tokens=350`) — documented because it's a
genuine, load-bearing property of "modest CPU-only hardware" self-hosting,
not a bug: ~5-6s one-time model load (cached in-process after, via a
class-level singleton in `QwenGenerator`); ~1-11s retrieval (BGE-M3 embed
+ Qdrant hybrid search); generation itself ranging from ~30s for a short
decline to ~130-150s for a ~300-token detailed cited answer with a
~2400-token prompt. `generation_max_tokens` was reduced from an initial
700 to 350 specifically to keep worst-case latency bounded after seeing
these real numbers, not as a guess.

**Model choice — real architectural decision, not a silent downgrade:**
partway through this phase, self-hosting Qwen3-8B per the original brief
was interrupted (an install of the model-appropriate `llama-cpp-python`
was rejected mid-command), and the alternative of using Claude Haiku (a
third-party SaaS API) as a swappable placeholder was proposed. That would
have directly conflicted with the brief's self-hosting requirement, so it
was raised explicitly rather than assumed — the confirmed choice was
**self-hosted, but a smaller Qwen3 model** (4B instead of 8B), keeping the
"no third-party API in the core pipeline" property intact. This is a real
trade-off (smaller model, somewhat weaker generation quality) made
transparently, with the swap path back to 8B on real hardware documented
above and in `config.py`/`download_generation_model.py`'s comments.

## How to verify Phase 8

```bash
# 1. Backend (from repo root) — same as earlier phases
uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# 2. Frontend
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
# open http://127.0.0.1:5173 — the widget launcher is bottom-right
```

Two real environment issues turned up getting this far (both fixed, both
worth documenting rather than papering over):

- After a container/session restart, `llama-cpp-python` started crashing
  the whole Python process with `Illegal instruction` (SIGILL) — not a
  Python exception, an actual CPU fault — the instant it tried to load
  the GGUF model. `/proc/cpuinfo` on this rebuilt sandbox reports AVX-512
  support, but the prebuilt wheel's AVX-512 code path apparently isn't
  safe to execute in this particular virtualized environment (a known
  class of issue with cloud VMs whose reported CPU flags don't perfectly
  match what's safe to execute). Fixed by rebuilding `llama-cpp-python`
  from source with AVX-512 explicitly disabled
  (`CMAKE_ARGS="-DGGML_AVX512=off -DGGML_AVX512_VBMI=off
  -DGGML_AVX512_VNNI=off -DGGML_AVX_VNNI=off" pip install
  llama-cpp-python==0.3.34 --no-cache-dir --force-reinstall
  --no-binary llama-cpp-python`), confirmed by loading the model directly
  in Python before touching the API again. Real deployment hardware
  should be checked the same way rather than assumed safe.
- Same container restart also meant `dockerd` and the Postgres/Qdrant
  containers had stopped — see the operational note at the end of this
  file; this happened again during this phase and was fixed the same way
  as before (restart `dockerd`, `docker compose up -d` restores the
  existing containers and their data volumes intact).

Verification covered three things — the chat API standalone, genuine
multi-turn memory (the feature added mid-phase), and the widget UI itself
end-to-end with real backend responses:

1. **`/chat` standalone**, same content as Phase 7's `/generate` tests
   (grounded answer with citations; zero-context short-circuit) but
   through the new endpoint shape, confirming the wrapper adds nothing
   incorrect.
2. **Multi-turn memory, tested for real, not assumed:** re-approved the
   same 12-chunk real OCI checklist used in earlier phases' tests, then
   ran an actual 3-turn conversation through `/chat`:
   - Turn 1: *"What documents does an adult need for a new OCI
     application?"* → correct grounded answer, citations `[1][2][4][6]`.
   - Turn 2 (history = turn 1): *"Does that Green Card copy need to be
     notarized?"* → correctly answered "Yes... Notarized valid Green Card
     — FRONT AND BACK [2]", i.e. it used the sources newly retrieved for
     *this* turn, not just echoed turn 1.
   - **Turn 3 (the test that actually matters): *"Does it need to be
     notarized?"*** — no mention of OCI, Green Card, or any noun at all,
     genuinely unanswerable without conversational context. It correctly
     resolved to the OCI application's notarization requirements across
     multiple documents, fully cited (`[1][2][3][4][5][6]`, 0 uncited),
     proving both halves of the Phase 8 memory design work together: the
     retrieval-side heuristic pulled back the right chunks despite a
     content-free query, and the prompt-side history let the model
     understand what "it" meant.
   - All 12 chunks were reverted to `pending_review` in both Postgres and
     Qdrant afterward (confirmed 410/410 pending again), same as every
     prior phase's test-approval cleanup.
3. **The widget itself, end-to-end, with screenshots:** re-approved the
   same 12 chunks again, drove the actual running frontend with
   Playwright (not a mock), and captured: the closed launcher button; the
   open panel's empty state (welcome copy, no messages); the panel
   mid-request, showing the sent message bubble and the thinking
   indicator with no empty placeholder bubble above it (an early version
   did show a stray empty bubble there — fixed by wrapping the assistant
   message in a `hasContent` check so it only renders once real content
   exists, letting the thinking indicator carry that in-between state
   instead); and the final answer rendered with clickable citation chips.
   Reverted the 12
   chunks to `pending_review` again afterward (confirmed 410/410).

## Notes for later phases

- `sources.id` / all PKs are UUIDs (via `pgcrypto`'s `gen_random_uuid()`) —
  chunk UUIDs are intended to double as Qdrant point IDs in Phase 4.
- `audit_log` is the append-only home for **every** fetch attempt
  (success or failure), not just successful ones — `source_versions` only
  gets a row when a fetch actually produces a new content version. Phase 2's
  "log every attempt" requirement is satisfied by writing to `audit_log`.
- `chunks` already carries every field the brief requires (`source_url`,
  `retrieval_date`, `source_last_modified`, `content_hash`,
  `service_category`, `canonical`, `jurisdiction`, `applicant_variant`,
  `review_status`, `version`) plus `embedding_model` / `qdrant_point_id`
  columns left empty until Phase 4, and `superseded_by` for the
  don't-overwrite/version-and-re-review flow.
- `service_category` enum includes `passport`, `birth_death_certificates`,
  and `police_clearance` in addition to the categories named in the brief's
  example list, since those are explicitly in-scope source topics.
- Content hashing in Phase 2 is over the **raw** fetched content (full
  rendered HTML or raw PDF bytes) — the brief's eventual intent is to hash
  the *extracted region* instead, once Phase 3's extractor exists. Whoever
  wires the full pipeline together should re-hash post-extraction and treat
  that as canonical; Phase 2's raw hash is a placeholder sufficient to prove
  the fetcher works and to detect gross raw-page changes.
- The registry itself flags several real conflicts/staleness issues (items
  46/47's contradictory OCI re-issue fee wording, 7/8's duplicate
  passport-loss checklists, 24's three form versions in circulation, 30's
  2021-dated PCC checklist) — carried for now in `sources.notes` as
  free text, per your call to defer real conflict-suppression logic to the
  Phase 5 admin review workflow rather than build it now.
- `not_yet_located` and `pending_from_office` items from the registry are
  intentionally NOT in the `sources` table (no URL to fetch) — they're
  preserved in `db/seed/source_registry.json` and the Claude project doc
  as a gap list for office follow-up, not as fetchable rows.
- The fetcher's request-context fallback for HTML (see the environment
  quirk note above) means a `source_versions` row can end up backed by
  content that never executed JavaScript. `used_request_fallback` is
  logged on every successful fetch specifically so this is visible later,
  rather than silently indistinguishable from a full render.
- Phase 3's `chunk_text` is what Phase 4 should hash and treat as the
  canonical `content_hash` / re-versioning trigger — not Phase 2's raw-page
  hash (see the note above it). Once Phase 4 exists, re-fetching a source
  whose *chunks* are unchanged (even if raw bytes drifted due to a nonce)
  should NOT create a new chunk version; that comparison has to happen at
  the chunk level, not the raw-content level.
- PDF heading detection is a font-size heuristic (relative to the
  document's most common body size), not real semantic structure — it's
  reasonable for the checklist-style PDFs in this registry (clear visual
  hierarchy) but will misclassify documents with unusual/inconsistent
  typography. If Phase 4+ retrieval quality looks off for a specific PDF,
  check `data/chunks_preview/<source_id>.json` first before assuming a
  retrieval bug.
- `Chunk.used_ocr` (surfaced as `_used_ocr` in `tag_metadata`'s output) is
  now carried into the `chunks` table (migration `0003_chunks_used_ocr.sql`,
  backfilled during Phase 4 — see the "Bug found and fixed" note above) as
  well as into every Qdrant point's payload, so both stores agree on which
  chunks came from OCR fallback rather than a native text layer.
