# Governed order workflow runbook

## Safe setup and launch

1. Create an isolated PostgreSQL database and copy `.env.example` to an untracked `.env`.
2. Install `requirements.txt` in a virtual environment and run both SQL migrations in order.
3. Install the frontend with `npm ci --prefix frontend`.
4. Run `./start.sh`. It binds to loopback, refuses occupied ports, and never kills a process, migrates, or seeds data.

Migration and seed actions are deliberately separate from startup:

```sh
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f backend/migrations/001_initial_schema.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f backend/migrations/002_governed_order_workflow.sql
```

## Workflow controls

- Create an order at `POST /api/governed-orders`, then change it only through `POST /api/governed-orders/{id}/commands` with a stable `Idempotency-Key`.
- Customer, merchant, and operator ownership is checked on every read and mutation. Provider callbacks use a configured HMAC secret and deterministic provider/event key.
- Provider operations are exposed at `POST /api/governed-orders/{id}/providers/{capability}`. Configure inventory, tax, payment, and shipping endpoints independently; missing or non-TLS configuration fails closed.
- The current snapshot uses optimistic versioning. `governed_order_events` is the immutable audit stream and `governed_order_outbox` provides retry state for downstream delivery.

## Operations and incidents

- Reuse the same idempotency key after network uncertainty. Do not invent a new key until the original provider outcome is reconciled.
- Inspect `/api/governed-orders/{id}/audit` and the reconciliation object before manual recovery.
- Restrict `recover` to operators and record the incident reference in its payload.
- Rotate all provider tokens and webhook secrets on suspected exposure. The historical tracked ngrok token must be revoked in the provider console; replacing the working-tree file does not erase Git history.
- Back up both snapshot and event tables together. Test restore and migration reapplication before release.

## Required release evidence

Run `python -m unittest discover -s tests -v`, compile the backend, apply the migration twice to a fresh database, and build/audit the frontend. Production provider and partner certification still require sandbox credentials and signed test events.
