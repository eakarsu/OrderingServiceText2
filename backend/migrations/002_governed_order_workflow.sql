BEGIN;

CREATE TABLE IF NOT EXISTS governed_orders (
    id UUID PRIMARY KEY,
    customer_id BIGINT NOT NULL,
    merchant_id BIGINT NOT NULL,
    currency CHAR(3) NOT NULL,
    total_cents BIGINT NOT NULL CHECK (total_cents >= 0),
    state VARCHAR(32) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    items JSONB NOT NULL,
    reserved JSONB NOT NULL DEFAULT '{}'::jsonb,
    fulfilled JSONB NOT NULL DEFAULT '{}'::jsonb,
    captured_cents BIGINT NOT NULL DEFAULT 0 CHECK (captured_cents >= 0),
    refunded_cents BIGINT NOT NULL DEFAULT 0 CHECK (refunded_cents >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (refunded_cents <= captured_cents)
);

CREATE INDEX IF NOT EXISTS governed_orders_customer_idx ON governed_orders(customer_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS governed_orders_merchant_idx ON governed_orders(merchant_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS governed_order_events (
    sequence BIGSERIAL PRIMARY KEY,
    order_id UUID NOT NULL REFERENCES governed_orders(id),
    actor_id BIGINT NOT NULL,
    actor_role VARCHAR(32) NOT NULL,
    command VARCHAR(40) NOT NULL,
    from_state VARCHAR(32) NOT NULL,
    to_state VARCHAR(32) NOT NULL,
    idempotency_key VARCHAR(128) NOT NULL,
    payload JSONB NOT NULL,
    source_timestamp TIMESTAMPTZ,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(order_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS governed_order_events_order_idx ON governed_order_events(order_id, sequence);

CREATE OR REPLACE FUNCTION reject_governed_order_event_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'governed_order_events is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS governed_order_events_immutable ON governed_order_events;
CREATE TRIGGER governed_order_events_immutable
BEFORE UPDATE OR DELETE ON governed_order_events
FOR EACH ROW EXECUTE FUNCTION reject_governed_order_event_mutation();

CREATE TABLE IF NOT EXISTS governed_provider_receipts (
    id BIGSERIAL PRIMARY KEY,
    provider VARCHAR(40) NOT NULL,
    external_event_id VARCHAR(200) NOT NULL,
    order_id UUID NOT NULL REFERENCES governed_orders(id),
    event_type VARCHAR(80) NOT NULL,
    source_timestamp TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('accepted','duplicate','rejected')),
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(provider, external_event_id)
);

CREATE TABLE IF NOT EXISTS governed_order_outbox (
    id BIGSERIAL PRIMARY KEY,
    order_id UUID NOT NULL REFERENCES governed_orders(id),
    topic VARCHAR(80) NOT NULL,
    idempotency_key VARCHAR(128) NOT NULL UNIQUE,
    payload JSONB NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    delivered_at TIMESTAMPTZ,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMIT;
