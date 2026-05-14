# Audit Note — OrderingServiceText

**Date**: 2026-05-06
**Bucket**: A. DETECTOR_FALSE_POSITIVE

## Summary

The audit (`_AUDIT/reports/batch_11.md` → "OrderingServiceText" entry)
explicitly classified this as `partial-build — functional ordering system
with complex AI order processing but crude structure`. The TSV-based
counters showed 0 routes / 0 AI endpoints because the AI logic lives in
Python, not in a Node/Express `routes/ai.js`.

The whole-project LLM scan confirms the AI surface exists.

## Files containing LLM references

- `ingestionPipeline.py` — menu PDF/image ingestion using LLM.
- `utils.py` — shared LLM helpers.
- `orderChat.py` — root-level conversational ordering agent.
- `docker/orderChat.py` — containerized orderChat copy.

Adjacent (no direct provider-name match but part of the AI flow):
- `orderProcessor.py`
- `menuIndexer.py`
- `RestaurantOrder.py`
- `chroma_database/` — vector store powering retrieval-augmented ordering.

## Disposition

- **Detector false positive.** The Python-based AI surface was missed by
  the Node-route counter.
- **No code changes** applied.
  - Project is a Python (FastAPI/Flask + Chroma) backend, not the
    Node/Express pattern the apply workflow scaffolds.
  - Adding `ai.js` here would conflict with the existing Python pipeline.

## Project structure observations

- Root has the Python script-based AI entry points: `main.py` (Twilio +
  AssemblyAI + Vosk), `orderChat.py`, `orderProcessor.py`, `menuIndexer.py`,
  `ingestionPipeline.py`, `RestaurantOrder.py`, `utils.py`, plus
  `chroma_database/` vector store.
- A separate `backend/` package exists with FastAPI routers
  (`routers/auth.py`, `categories.py`, `dashboard.py`, `menu_items.py`,
  `orders.py`, `users.py`) and services (`auth_service.py`,
  `email_service.py`, `export_service.py`, `menu_service.py`,
  `order_service.py`, `user_service.py`), but **no `main.py`/`app.py` is
  present to mount them** — the routers are unwired.
- A separate `api/index.py` exists (likely Vercel-style serverless entry).

The backend routers are therefore "dark code" today. Wiring them into a
real FastAPI app + connecting them to the AI flow is the highest-value
follow-up but is too large for an apply batch (whole-project surgery,
schema reconciliation, and AI/non-AI integration tests).

## Audit recommendations applied this batch

**None.** Each of the audit batch_11 §OrderingServiceText recommendations
needs either:

- Python-track scaffolding (recommendation engine, real-time status, loyalty,
  multi-restaurant) — TOO-RISKY to write without first wiring `backend/` into
  a working FastAPI app, and
- External integrations (POS, delivery routing) — NEEDS-CREDS forbidden by
  the apply workflow.

This batch's mandate is mechanical wins on existing AI; OrderingServiceText
needs a foundational Python-architecture decision first.

## Backlog (deferred, prioritised)

1. **Wire `backend/` routers into a FastAPI `main.py`/`app.py`** — the
   routers exist and follow the `APIRouter(prefix="/api/...")` pattern;
   one entry-point file (~30 LOC) would surface them.
   PRE-REQ for everything below.
2. **Bridge the script-based AI flow (orderChat.py) into the FastAPI
   surface** — currently isolated; would benefit from a shared
   `services/ai_service.py` module.
3. **Recommendation engine** — combine ChromaDB embeddings with order
   history; new endpoint `POST /api/recommendations`. Can ride on top
   of the existing vector store.
4. **Multi-restaurant support** — `sectors/` has multi-vertical scaffolding;
   needs schema review (NEEDS-PRODUCT-DECISION).
5. **Real-time order status UI / WebSocket** — `main.py` already uses
   FastAPI WebSocket for Twilio media; reuse the pattern for status push.
6. **Inventory tracking + delivery routing AI** — needs supplier and
   mapping APIs (NEEDS-CREDS).
7. **POS integration** — NEEDS-CREDS.
8. **Loyalty / preferences storage** — schema-level work, depends on
   backlog #1.

## Files touched this batch

None.

## Apply pass 3 (frontend)

- **Action**: SKIPPED-NO-DOMAIN.
- No Node/Express AI endpoints exist; the FastAPI `backend/routers/` package is **not mounted** to any app entry point (audit backlog item #1) and contains no AI routes.
- The actual AI flow (`orderChat.py`, `orderProcessor.py`, `menuIndexer.py`, `ingestionPipeline.py`) is script-based (Twilio + AssemblyAI + Vosk + Chroma) and is not exposed as JWT-secured HTTP endpoints suitable for FE wiring.
- Wiring FE to AI would require backlog #1 first (FastAPI app entry-point + AI service module). Out of scope for an idempotent FE pass.
- Files modified this pass: none.

## Apply pass 3 (Group B — FastAPI bootstrap)

**Date**: 2026-05-07. Resolves backlog item #1 from the original note (wire
`backend/` routers into a FastAPI entry point) plus the frontend gap from
the previous pass-3 attempt.

### What was added

- **`app.py`** (NEW, project root) — canonical launchable FastAPI entry
  point. Mounts every existing `backend/routers/*` module (auth, users,
  orders, categories, menu_items, dashboard) plus the new ai router.
  Best-effort include: a router whose import blows up (e.g. missing
  Postgres at boot) is recorded in `/api/health` instead of crashing the
  whole app. Uses `dotenv` to load `.env` automatically. Adds CORS for
  `http://localhost:{3000,5173,8000}` (overridable via `CORS_ORIGINS`).
- **`backend/routers/ai.py`** (NEW) — wraps the standalone AI scripts as
  HTTP endpoints under `/api/ai/*`. **Wraps, not rewrites** —
  `orderChat.orderChat` and `menuIndexer.MenuIndexer` are imported lazily
  so the rest of the app boots without `chromadb`/`langchain` available.
  Endpoints created:
    * `GET  /api/ai/health` — probe (does not init the AI stack)
    * `POST /api/ai/chat` — wraps `orderChat.chatAway()` (the SMS flow)
    * `POST /api/ai/reset` — drop a caller_id session
    * `POST /api/ai/process-order` — single-shot menu lookup via
      `OrderProcessor.process_order()`
    * `GET  /api/ai/categories` — list Chroma-indexed menu categories
    * `POST /api/ai/index-menu` — admin: re-run `MenuIndexer` ingestion
  All endpoints return **HTTP 503 `{"error": "AI not configured"}`** when
  `OPENROUTER_API_KEY` is missing, when the lazy import fails, or when
  the underlying script raises (Postgres unreachable, Chroma uninitialized,
  etc.).
- **`static/index.html`** (NEW) — minimal vanilla-JS FE served by FastAPI
  `StaticFiles`. One `<fieldset>` per AI endpoint, text input + send
  button + JSON result panel; matches the project's text/SMS-ordering
  domain (caller-id field, "send a text message" form). On load, polls
  `/api/health` and shows an "AI configured" / "AI not configured" pill.

### What was NOT changed

- `api/index.py` (Vercel-style entry) is untouched and still works on
  its own — `app.py` is the new local-dev launchable. `main.py` (Twilio +
  AssemblyAI + Vosk voice flow) is untouched.
- No `pip install` performed. `requirements.txt` already contains
  `fastapi`, `uvicorn`, `python-dotenv`, `pydantic` — no new deps.
- No changes to `orderChat.py`, `orderProcessor.py`, `menuIndexer.py`,
  `ingestionPipeline.py`. The new router only imports and calls them.

### Launch instructions

```bash
cd /Users/erolakarsu/projects/OrderingServiceText
pip install -r requirements.txt   # if myvenv isn't already populated
export OPENROUTER_API_KEY=...      # otherwise /api/ai/* returns 503
uvicorn app:app --reload --port 8000
# open http://localhost:8000/  -> static FE
# open http://localhost:8000/docs  -> Swagger UI
```

### Verification

- `python3 -m py_compile app.py backend/routers/ai.py` → OK.

### Files written this pass

- `app.py` (NEW)
- `backend/routers/ai.py` (NEW)
- `static/index.html` (NEW)
- `_AUDIT_NOTE.md` (this section appended)
