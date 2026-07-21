"""Persisted, role-scoped order workflow and retry-safe provider webhooks."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from hashlib import sha256
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.services.governed_order import (
    Actor,
    OrderControlError,
    OrderSnapshot,
    apply_command,
    idempotency_key,
    reconciliation,
    validate_new_order,
    verify_webhook_signature,
)
from backend.services.provider_adapters import ProviderError, ProviderRegistry

router = APIRouter(prefix="/api/governed-orders", tags=["governed-orders"])


class CreateOrderRequest(BaseModel):
    customer_id: int
    merchant_id: int
    currency: str = Field(min_length=3, max_length=3)
    total_cents: int = Field(ge=0)
    items: dict[str, int]

    @field_validator("items")
    @classmethod
    def validate_items(cls, value):
        if not value or any(not sku.strip() or quantity <= 0 for sku, quantity in value.items()):
            raise ValueError("items require non-empty SKUs and positive quantities")
        return value


class CommandRequest(BaseModel):
    command: str = Field(min_length=2, max_length=40)
    payload: dict = Field(default_factory=dict)
    source_timestamp: datetime | None = None


class ProviderCallRequest(BaseModel):
    operation: str = Field(min_length=2, max_length=40)
    payload: dict = Field(default_factory=dict)


def _actor(user: dict) -> Actor:
    return Actor(user_id=int(user["id"]), role=str(user["role"]))


def _json_object(value) -> dict:
    if isinstance(value, dict):
        return value
    return json.loads(value or "{}")


def _snapshot(row, seen_keys=()) -> OrderSnapshot:
    return OrderSnapshot(
        order_id=str(row[0]),
        customer_id=int(row[1]),
        merchant_id=int(row[2]),
        currency=row[3],
        total_cents=int(row[4]),
        state=row[5],
        version=int(row[6]),
        items=tuple(sorted(_json_object(row[7]).items())),
        reserved=tuple(sorted(_json_object(row[8]).items())),
        fulfilled=tuple(sorted(_json_object(row[9]).items())),
        captured_cents=int(row[10]),
        refunded_cents=int(row[11]),
        seen_keys=frozenset(seen_keys),
    )


def _select_order(conn, order_id: UUID, for_update=False):
    suffix = " FOR UPDATE" if for_update else ""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id, customer_id, merchant_id, currency, total_cents, state,
                      version, items, reserved, fulfilled, captured_cents, refunded_cents
                 FROM governed_orders WHERE id = %s""" + suffix,
            (str(order_id),),
        )
        row = cur.fetchone()
        if not row:
            return None
        cur.execute("SELECT idempotency_key FROM governed_order_events WHERE order_id = %s", (str(order_id),))
        return _snapshot(row, [event[0] for event in cur.fetchall()])


def _check_read_scope(actor: Actor, order: OrderSnapshot):
    role = actor.canonical_role
    if role == "operator":
        return
    if role == "customer" and actor.user_id == order.customer_id:
        return
    if role == "merchant" and actor.user_id == order.merchant_id:
        return
    raise OrderControlError("ORDER_SCOPE", "Order is outside the actor's scope")


def _error(error: OrderControlError):
    status = 403 if error.code in {"ROLE_FORBIDDEN", "ORDER_SCOPE"} else 409
    raise HTTPException(status_code=status, detail={"code": error.code, "message": str(error)})


@router.post("", status_code=201)
def create_order(req: CreateOrderRequest, user=Depends(get_current_user), conn=Depends(get_db)):
    actor = _actor(user)
    role = actor.canonical_role
    if role not in {"customer", "merchant", "operator"}:
        raise HTTPException(status_code=403, detail="Unsupported order role")
    if role == "customer" and actor.user_id != req.customer_id:
        raise HTTPException(status_code=403, detail="Customer identity mismatch")
    if role == "merchant" and actor.user_id != req.merchant_id:
        raise HTTPException(status_code=403, detail="Merchant identity mismatch")
    order = OrderSnapshot(
        order_id=str(uuid4()),
        customer_id=req.customer_id,
        merchant_id=req.merchant_id,
        currency=req.currency.upper(),
        total_cents=req.total_cents,
        items=tuple(sorted(req.items.items())),
    )
    try:
        validate_new_order(order)
    except OrderControlError as error:
        _error(error)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO governed_orders
               (id, customer_id, merchant_id, currency, total_cents, state, version,
                items, reserved, fulfilled, captured_cents, refunded_cents)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,'{}'::jsonb,'{}'::jsonb,0,0)""",
            (order.order_id, order.customer_id, order.merchant_id, order.currency,
             order.total_cents, order.state, order.version, json.dumps(dict(order.items))),
        )
        cur.execute(
            """INSERT INTO governed_order_events
               (order_id, actor_id, actor_role, command, from_state, to_state,
                idempotency_key, payload)
               VALUES (%s,%s,%s,'create','none','draft',%s,'{}'::jsonb)""",
            (order.order_id, actor.user_id, actor.canonical_role, f"create:{order.order_id}"),
        )
    conn.commit()
    return {"id": order.order_id, "state": order.state, "version": order.version}


@router.post("/{order_id}/commands")
def execute_command(
    order_id: UUID,
    req: CommandRequest,
    idempotency: str = Header(alias="Idempotency-Key", min_length=8, max_length=128),
    user=Depends(get_current_user),
    conn=Depends(get_db),
):
    actor = _actor(user)
    try:
        order = _select_order(conn, order_id, for_update=True)
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        updated, result = apply_command(order, actor, req.command, req.payload, idempotency)
        if result["duplicate"]:
            conn.rollback()
            return {**result, "order_id": str(order_id)}
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO governed_order_events
                   (order_id, actor_id, actor_role, command, from_state, to_state,
                    idempotency_key, payload, source_timestamp)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)""",
                (str(order_id), actor.user_id, actor.canonical_role, req.command,
                 order.state, updated.state, idempotency, json.dumps(req.payload), req.source_timestamp),
            )
            cur.execute(
                """UPDATE governed_orders
                      SET state=%s, version=%s, reserved=%s::jsonb, fulfilled=%s::jsonb,
                          captured_cents=%s, refunded_cents=%s, updated_at=NOW()
                    WHERE id=%s AND version=%s""",
                (updated.state, updated.version, json.dumps(dict(updated.reserved)),
                 json.dumps(dict(updated.fulfilled)), updated.captured_cents,
                 updated.refunded_cents, str(order_id), order.version),
            )
            if cur.rowcount != 1:
                raise OrderControlError("VERSION_CONFLICT", "Order changed during command processing")
            cur.execute(
                """INSERT INTO governed_order_outbox
                   (order_id, topic, idempotency_key, payload)
                   VALUES (%s,'order.state.changed',%s,%s::jsonb)
                   ON CONFLICT (idempotency_key) DO NOTHING""",
                (str(order_id), f"outbox:{idempotency}", json.dumps({
                    "order_id": str(order_id), "from": order.state, "to": updated.state,
                    "version": updated.version,
                })),
            )
        conn.commit()
        return {**result, "order_id": str(order_id), "reconciliation": reconciliation(updated)}
    except OrderControlError as error:
        conn.rollback()
        _error(error)


@router.get("/{order_id}")
def get_governed_order(order_id: UUID, user=Depends(get_current_user), conn=Depends(get_db)):
    order = _select_order(conn, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    try:
        _check_read_scope(_actor(user), order)
    except OrderControlError as error:
        _error(error)
    return {
        "id": order.order_id,
        "state": order.state,
        "version": order.version,
        "customer_id": order.customer_id,
        "merchant_id": order.merchant_id,
        "currency": order.currency,
        "total_cents": order.total_cents,
        "items": dict(order.items),
        "reserved": dict(order.reserved),
        "fulfilled": dict(order.fulfilled),
        "captured_cents": order.captured_cents,
        "refunded_cents": order.refunded_cents,
        "reconciliation": reconciliation(order),
    }


@router.get("/{order_id}/audit")
def order_audit(order_id: UUID, user=Depends(get_current_user), conn=Depends(get_db)):
    order = _select_order(conn, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    try:
        _check_read_scope(_actor(user), order)
    except OrderControlError as error:
        _error(error)
    with conn.cursor() as cur:
        cur.execute(
            """SELECT sequence, actor_id, actor_role, command, from_state, to_state,
                      idempotency_key, payload, source_timestamp, occurred_at
                 FROM governed_order_events WHERE order_id=%s ORDER BY sequence""",
            (str(order_id),),
        )
        rows = cur.fetchall()
    return {"order_id": str(order_id), "events": [
        {"sequence": row[0], "actor_id": row[1], "actor_role": row[2],
         "command": row[3], "from_state": row[4], "to_state": row[5],
         "idempotency_key": row[6], "payload": row[7],
         "source_timestamp": row[8], "occurred_at": row[9]} for row in rows
    ]}


@router.post("/{order_id}/providers/{capability}")
def execute_provider_call(
    order_id: UUID,
    capability: str,
    req: ProviderCallRequest,
    idempotency: str = Header(alias="Idempotency-Key", min_length=8, max_length=128),
    user=Depends(get_current_user),
    conn=Depends(get_db),
):
    """Execute one configured provider operation without bypassing order state controls."""
    actor = _actor(user)
    order = _select_order(conn, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    try:
        _check_read_scope(actor, order)
        if actor.canonical_role == "customer" and capability not in {"payment", "tax", "shipping"}:
            raise OrderControlError("ROLE_FORBIDDEN", "Customer cannot call this provider capability")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM governed_order_events WHERE order_id=%s AND idempotency_key=%s",
                (str(order_id), idempotency),
            )
            if cur.fetchone():
                return {"duplicate": True, "order_id": str(order_id)}
        result = ProviderRegistry().execute(capability, req.operation, req.payload, idempotency)
        response_json = json.dumps(result.payload, sort_keys=True, separators=(",", ":"))
        audit_payload = {
            "provider_reference": result.provider_reference,
            "response_sha256": sha256(response_json.encode()).hexdigest(),
        }
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO governed_order_events
                   (order_id, actor_id, actor_role, command, from_state, to_state,
                    idempotency_key, payload)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                   ON CONFLICT (order_id, idempotency_key) DO NOTHING RETURNING sequence""",
                (str(order_id), actor.user_id, actor.canonical_role,
                 f"provider:{capability}:{req.operation}"[:40], order.state, order.state,
                 idempotency, json.dumps(audit_payload)),
            )
            inserted = cur.fetchone()
        conn.commit()
        return {
            "duplicate": inserted is None,
            "order_id": str(order_id),
            "provider_reference": result.provider_reference,
            "result": result.payload,
        }
    except OrderControlError as error:
        conn.rollback()
        _error(error)
    except ProviderError as error:
        conn.rollback()
        status = 503 if error.retryable or error.code == "PROVIDER_NOT_CONFIGURED" else 422
        raise HTTPException(status_code=status, detail={
            "code": error.code,
            "message": str(error),
            "retryable": error.retryable,
        }) from error


WEBHOOK_COMMANDS = {
    "payment.succeeded": "payment_succeeded",
    "payment.failed": "payment_failed",
    "refund.succeeded": "refund_succeeded",
    "refund.failed": "refund_failed",
}


@router.post("/webhooks/{provider}")
async def provider_webhook(
    provider: str,
    request: Request,
    signature: str = Header(alias="X-Webhook-Signature"),
    conn=Depends(get_db),
):
    if not re.fullmatch(r"[a-z0-9_-]{2,40}", provider):
        raise HTTPException(status_code=400, detail="Invalid provider")
    secret = os.getenv(f"ORDER_WEBHOOK_{provider.upper()}_SECRET", "")
    raw = await request.body()
    if not secret:
        raise HTTPException(status_code=503, detail="Provider webhook is not configured")
    if not verify_webhook_signature(raw, signature, secret):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    try:
        body = json.loads(raw)
        event_id = str(body["id"])
        order_id = UUID(str(body["order_id"]))
        event_type = str(body["type"])
        source_timestamp = datetime.fromisoformat(str(body["created_at"]).replace("Z", "+00:00"))
        payload = body.get("payload") or {}
        command = WEBHOOK_COMMANDS[event_type]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Malformed provider event")
    key = idempotency_key(provider, event_id)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO governed_provider_receipts
                   (provider, external_event_id, order_id, event_type, source_timestamp, payload, status)
                   VALUES (%s,%s,%s,%s,%s,%s::jsonb,'accepted')
                   ON CONFLICT (provider, external_event_id) DO NOTHING RETURNING id""",
                (provider, event_id, str(order_id), event_type, source_timestamp, json.dumps(payload)),
            )
            if not cur.fetchone():
                conn.rollback()
                return {"duplicate": True, "event_id": event_id}
        order = _select_order(conn, order_id, for_update=True)
        if not order:
            raise OrderControlError("ORDER_NOT_FOUND", "Provider event references an unknown order")
        updated, result = apply_command(order, Actor(0, "operator"), command, payload, key)
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO governed_order_events
                   (order_id, actor_id, actor_role, command, from_state, to_state,
                    idempotency_key, payload, source_timestamp)
                   VALUES (%s,0,'provider',%s,%s,%s,%s,%s::jsonb,%s)""",
                (str(order_id), command, order.state, updated.state, key, json.dumps(payload), source_timestamp),
            )
            cur.execute(
                """UPDATE governed_orders SET state=%s, version=%s, captured_cents=%s,
                          refunded_cents=%s, updated_at=NOW() WHERE id=%s AND version=%s""",
                (updated.state, updated.version, updated.captured_cents, updated.refunded_cents,
                 str(order_id), order.version),
            )
            if cur.rowcount != 1:
                raise OrderControlError("VERSION_CONFLICT", "Order changed during provider processing")
            cur.execute(
                """INSERT INTO governed_order_outbox
                   (order_id, topic, idempotency_key, payload)
                   VALUES (%s,'order.state.changed',%s,%s::jsonb)
                   ON CONFLICT (idempotency_key) DO NOTHING""",
                (str(order_id), f"outbox:{key}", json.dumps({
                    "order_id": str(order_id), "from": order.state, "to": updated.state,
                    "version": updated.version,
                })),
            )
        conn.commit()
        return {**result, "event_id": event_id, "order_id": str(order_id)}
    except OrderControlError as error:
        conn.rollback()
        _error(error)
