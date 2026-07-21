# Completeness Review: OrderingServiceText2

**Review date:** 2026-07-18

## Assessment basis

Static inspection of project-owned source and configuration only; no dependency installation, build, database migration, external-service call, or runtime launch was performed. The scan considered 211 project files (96 source files), 3 manifest(s), 0 test-like file(s), and 0 CI workflow(s), excluding dependency/generated directories.

## Classification

**Functional but incomplete**

This is a substantive but unfinished commerce/order operations application, not just an empty scaffold. Inspection found 96 source files across `docker/`, `frontend/`, `backend/`, `chroma_database/` using Next.js, React, Python; however, the checked-in workflow and delivery controls do not yet demonstrate a complete, production-operable product.

## Why it is not complete

- Generated gap/visualization routes describe missing capabilities or simulate recommendations; they do not implement the underlying domain operation.
- Generic LLM calls are used as product behavior without enough typed tools, grounded evidence, deterministic rules, or output evaluation.
- Mock, demo, sample, fixture, or placeholder behavior remains in executable/product paths.
- No recognizable project-owned automated tests were found for the main workflow.
- No checked-in CI workflow proves builds, tests, migrations, and security checks on every change.

## Needed features

1. Implement an idempotent order state machine covering reservation, payment, cancellation, refund, fulfillment, and exception recovery.
2. Connect real inventory, tax, payment, shipping/delivery, and partner-webhook providers behind retry-safe adapters.
3. Add role-scoped customer, operator, and merchant workflows with immutable order and refund audit history.
4. Test duplicate webhooks, partial fulfillment, payment failure, overselling, and reconciliation end to end.
5. Add risk-based unit, integration, and end-to-end tests in CI, including migration and failure-path coverage.

## Risks or launch blockers

- Automation contains destructive process, filesystem, or database operations; do not run it on a shared machine without review.
- Startup appears coupled to seed/migration behavior, risking data mutation or non-repeatable launches.
- AI-provider availability, cost, privacy, prompt injection, and unvalidated output are launch risks until bounded and evaluated.
- Regression risk is high because no recognizable project-owned automated tests cover the main path.

## Evidence inspected

- `README.md`
- `frontend/src/App.tsx:23`
- `static/index.html:58`
- `app.py`
- `requirements.txt`
- `start.sh`

## Recommended next action

Choose one real commerce/order operations journey, define acceptance criteria and external contracts, then close its persistence, permission, integration, failure, and test gaps before expanding features.

## Implementation progress (2026-07-19)

1. Implemented a deterministic, idempotent order state machine in `backend/services/governed_order.py`, persisted by `backend/routers/governed_orders.py` and `backend/migrations/002_governed_order_workflow.sql`. It covers reservation, exact-cent payment, cancellation, bounded refund, partial/full fulfillment, exceptions, operator recovery, optimistic version conflicts, inventory oversell prevention, reconciliation, and a durable outbox. Direct and bulk legacy status mutations now return HTTP 409 instead of bypassing the command reducer.
2. Added configured HTTPS adapters for inventory, tax, payment, and shipping in `backend/services/provider_adapters.py`. Calls carry stable provider idempotency keys, use bounded transient retries, reject unknown operations/non-TLS or missing configuration, and record hashed response evidence rather than raw provider data. Signed, timestamped partner webhooks are HMAC-verified, provider/event deduplicated, translated to typed commands, transactionally persisted, and emitted to the same outbox. Real provider acceptance remains an external gate rather than a simulated success claim.
3. Added customer, merchant, and operator ownership/role enforcement on every governed read and command. Order snapshots use atomic versioned writes; `governed_order_events` is protected by a database trigger against update/delete and exposes a scoped audit endpoint. Generated/direct-model `ai`, `ai-extras`, and `custom-views` product surfaces are no longer mounted and return HTTP 410, so model output cannot mutate inventory, money, fulfillment, or partner state.
4. Added 16 project-owned unit, architecture, adapter, and PostgreSQL-backed HTTP tests covering duplicate commands/webhooks, invalid and over-reservation inventory, payment failure/recovery, partial/over-fulfillment, refund limits, ownership, webhook signatures, retryable and terminal provider failures, reconciliation, route quarantine, immutable audit records, persisted payment, and event ordering. All 16 passed on 2026-07-19; the database test ran through the real FastAPI router against a disposable PostgreSQL 15 instance.
5. Added CI that installs locked frontend dependencies, compiles/import-smokes the backend, applies the base and governed migrations to fresh PostgreSQL, reapplies the governed migration to prove repeatability, executes the full database-backed suite, audits production frontend dependencies, and builds the frontend. Locally, the migration applied twice, the live HTTP journey passed, Python compilation and shell/diff checks passed, the frontend production build passed, and `npm audit --omit=dev` reported zero vulnerabilities. `start.sh` now binds to loopback, refuses occupied ports, owns and cleans up only its child process, and never kills, migrates, seeds, or backgrounds shared services. Configuration and incident/recovery guidance are recorded in `.env.example` and `RUNBOOK_GOVERNED_ORDERS.md`.

External launch gates remain: configure and certify real inventory/tax/payment/shipping contracts and webhook schemas; complete payment/PCI, tax-jurisdiction, privacy, partner, and merchant-operations review; rotate the historically tracked ngrok credential at its provider and scrub/approve repository history; rehearse backup/restore and rollback; and run load, concurrency, provider-outage, fraud, and disaster-recovery exercises at intended scale. No provider credential, payment authorization, tax determination, shipment, production capacity, or regulatory certification is claimed from repository-only validation.

## Runtime verification (2026-07-20)

`start.sh` was exercised with disposable PostgreSQL on `127.0.0.1:55613`, API on `127.0.0.1:6040`, and reserved UI port `6041`. Attempts at `2026-07-20T19:46:15Z`, `19:48:10Z`, and `19:48:36Z` recorded `FAILED/login_failed` while fresh-database migration/bootstrap and validator-compatible login input were repaired. The final attempt at `2026-07-20T19:49:33Z` recorded `API_VERIFIED/startup_login_session_api`: the admin was persisted in PostgreSQL, login returned session tokens, and the authenticated `GET /api/users/me` request reloaded the current user from the database.

All 16 maintained tests passed with the PostgreSQL HTTP integration enabled on port 55613, including persisted payment webhook, partial fulfillment, ordered audit events, and database-enforced audit immutability. The frontend production build passed (2,472 modules; its existing chunk-size warning remains non-fatal). `start.sh` passed `bash -n`, the relevant Python files compiled, `git diff --check` passed, and all assigned listeners were released.
