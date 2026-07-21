"""Deterministic order controls shared by HTTP commands and tests.

No model output can reserve stock, capture/refund money, or change order state.
Provider adapters submit typed commands through the same reducer as operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import hmac
import json
from typing import Any, Mapping


class OrderControlError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


ROLE_ALIASES = {"admin": "operator", "manager": "merchant", "staff": "customer"}
COMMAND_ROLES = {
    "reserve": {"merchant", "operator"},
    "request_payment": {"customer", "merchant", "operator"},
    "payment_succeeded": {"operator"},
    "payment_failed": {"operator"},
    "start_fulfillment": {"merchant", "operator"},
    "fulfill_items": {"merchant", "operator"},
    "cancel": {"customer", "merchant", "operator"},
    "request_refund": {"customer", "merchant", "operator"},
    "refund_succeeded": {"operator"},
    "refund_failed": {"operator"},
    "recover": {"operator"},
}


@dataclass(frozen=True)
class Actor:
    user_id: int
    role: str

    @property
    def canonical_role(self) -> str:
        return ROLE_ALIASES.get(self.role, self.role)


@dataclass(frozen=True)
class OrderSnapshot:
    order_id: str
    customer_id: int
    merchant_id: int
    currency: str
    total_cents: int
    state: str = "draft"
    version: int = 1
    items: tuple[tuple[str, int], ...] = ()
    reserved: tuple[tuple[str, int], ...] = ()
    fulfilled: tuple[tuple[str, int], ...] = ()
    captured_cents: int = 0
    refunded_cents: int = 0
    seen_keys: frozenset[str] = field(default_factory=frozenset)


def _quantities(values: tuple[tuple[str, int], ...] | Mapping[str, int]) -> dict[str, int]:
    return dict(values)


def _frozen(values: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(values.items()))


def validate_new_order(order: OrderSnapshot) -> None:
    if order.total_cents < 0 or not isinstance(order.total_cents, int):
        raise OrderControlError("INVALID_MONEY", "total_cents must be a non-negative integer")
    if len(order.currency) != 3 or not order.currency.isalpha():
        raise OrderControlError("INVALID_CURRENCY", "currency must be a three-letter code")
    items = _quantities(order.items)
    if not items or any(not sku.strip() or not isinstance(quantity, int) or quantity <= 0 for sku, quantity in items.items()):
        raise OrderControlError("INVALID_ITEMS", "items require unique SKUs and positive integer quantities")
    if len(items) != len(order.items):
        raise OrderControlError("DUPLICATE_SKU", "duplicate item SKUs are not allowed")


def authorize_order(actor: Actor, order: OrderSnapshot, command: str) -> None:
    role = actor.canonical_role
    if role not in COMMAND_ROLES.get(command, set()):
        raise OrderControlError("ROLE_FORBIDDEN", f"{role} cannot execute {command}")
    if role == "customer" and actor.user_id != order.customer_id:
        raise OrderControlError("ORDER_SCOPE", "customer cannot act on another customer's order")
    if role == "merchant" and actor.user_id != order.merchant_id:
        raise OrderControlError("ORDER_SCOPE", "merchant cannot act on another merchant's order")


def idempotency_key(provider: str, external_id: str) -> str:
    normalized = f"{provider.strip().lower()}\0{external_id.strip().lower()}"
    return sha256(normalized.encode("utf-8")).hexdigest()


def verify_webhook_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    if not raw_body or not signature or not secret:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body, "sha256").hexdigest()
    return hmac.compare_digest(signature.removeprefix("sha256="), expected)


def apply_command(
    order: OrderSnapshot,
    actor: Actor,
    command: str,
    payload: Mapping[str, Any],
    request_key: str,
) -> tuple[OrderSnapshot, dict[str, Any]]:
    if not request_key:
        raise OrderControlError("IDEMPOTENCY_REQUIRED", "Every order command requires an idempotency key")
    if request_key in order.seen_keys:
        return order, {"duplicate": True, "state": order.state, "version": order.version}
    authorize_order(actor, order, command)

    next_state = order.state
    reserved = _quantities(order.reserved)
    fulfilled = _quantities(order.fulfilled)
    captured = order.captured_cents
    refunded = order.refunded_cents
    items = _quantities(order.items)

    if command == "reserve":
        if order.state not in {"draft", "exception"}:
            raise OrderControlError("INVALID_TRANSITION", "Reservation is allowed only from draft or exception")
        available = payload.get("available", {})
        if not isinstance(available, Mapping):
            raise OrderControlError("INVALID_INVENTORY", "available inventory map is required")
        shortages = {sku: quantity - int(available.get(sku, 0)) for sku, quantity in items.items() if int(available.get(sku, 0)) < quantity}
        if shortages:
            raise OrderControlError("OVERSELL_BLOCKED", json.dumps(shortages, sort_keys=True))
        reserved = dict(items)
        next_state = "reserved"
    elif command == "request_payment":
        if order.state != "reserved":
            raise OrderControlError("INVALID_TRANSITION", "Payment can be requested only after reservation")
        next_state = "payment_pending"
    elif command == "payment_succeeded":
        if order.state != "payment_pending":
            raise OrderControlError("INVALID_TRANSITION", "Payment success requires payment_pending")
        amount = payload.get("amount_cents")
        if not isinstance(amount, int) or amount != order.total_cents:
            raise OrderControlError("PAYMENT_MISMATCH", "Captured amount must exactly match order total")
        captured = amount
        next_state = "paid"
    elif command == "payment_failed":
        if order.state != "payment_pending":
            raise OrderControlError("INVALID_TRANSITION", "Payment failure requires payment_pending")
        next_state = "exception"
    elif command == "start_fulfillment":
        if order.state != "paid":
            raise OrderControlError("INVALID_TRANSITION", "Fulfillment requires captured payment")
        next_state = "fulfillment_pending"
    elif command == "fulfill_items":
        if order.state not in {"fulfillment_pending", "partially_fulfilled"}:
            raise OrderControlError("INVALID_TRANSITION", "Items cannot be fulfilled in the current state")
        quantities = payload.get("quantities")
        if not isinstance(quantities, Mapping) or not quantities:
            raise OrderControlError("INVALID_FULFILLMENT", "A non-empty quantities map is required")
        for sku, quantity in quantities.items():
            if sku not in items or not isinstance(quantity, int) or quantity <= 0:
                raise OrderControlError("INVALID_FULFILLMENT", "Unknown SKU or invalid quantity")
            new_quantity = fulfilled.get(sku, 0) + quantity
            if new_quantity > reserved.get(sku, 0):
                raise OrderControlError("OVER_FULFILL_BLOCKED", f"{sku} exceeds reserved quantity")
            fulfilled[sku] = new_quantity
        next_state = "fulfilled" if all(fulfilled.get(sku, 0) == quantity for sku, quantity in items.items()) else "partially_fulfilled"
    elif command == "cancel":
        if order.state in {"fulfilled", "refunded", "cancelled"}:
            raise OrderControlError("INVALID_TRANSITION", "Completed orders cannot be cancelled")
        next_state = "refund_pending" if captured > refunded else "cancelled"
    elif command == "request_refund":
        if order.state not in {"paid", "fulfillment_pending", "partially_fulfilled", "fulfilled", "exception"}:
            raise OrderControlError("INVALID_TRANSITION", "Refund is not available in the current state")
        amount = payload.get("amount_cents")
        if not isinstance(amount, int) or amount <= 0 or amount > captured - refunded:
            raise OrderControlError("REFUND_LIMIT", "Refund exceeds the unrefunded captured amount")
        next_state = "refund_pending"
    elif command == "refund_succeeded":
        if order.state != "refund_pending":
            raise OrderControlError("INVALID_TRANSITION", "Refund success requires refund_pending")
        amount = payload.get("amount_cents")
        if not isinstance(amount, int) or amount <= 0 or refunded + amount > captured:
            raise OrderControlError("REFUND_LIMIT", "Provider refund exceeds captured amount")
        refunded += amount
        next_state = "refunded" if refunded == captured else "paid"
    elif command == "refund_failed":
        if order.state != "refund_pending":
            raise OrderControlError("INVALID_TRANSITION", "Refund failure requires refund_pending")
        next_state = "exception"
    elif command == "recover":
        if order.state != "exception":
            raise OrderControlError("INVALID_TRANSITION", "Only exception orders can be recovered")
        target = payload.get("target_state")
        if target not in {"draft", "reserved", "paid", "refund_pending"}:
            raise OrderControlError("INVALID_RECOVERY", "Recovery target is not permitted")
        next_state = str(target)
    else:
        raise OrderControlError("UNKNOWN_COMMAND", f"Unknown order command: {command}")

    updated = replace(
        order,
        state=next_state,
        version=order.version + 1,
        reserved=_frozen(reserved),
        fulfilled=_frozen(fulfilled),
        captured_cents=captured,
        refunded_cents=refunded,
        seen_keys=order.seen_keys | {request_key},
    )
    return updated, {"duplicate": False, "state": updated.state, "version": updated.version}


def reconciliation(order: OrderSnapshot) -> dict[str, Any]:
    item_count = sum(_quantities(order.items).values())
    fulfilled_count = sum(_quantities(order.fulfilled).values())
    outstanding_cents = order.captured_cents - order.refunded_cents
    exceptions = []
    if order.captured_cents not in {0, order.total_cents}:
        exceptions.append("payment_total_mismatch")
    if fulfilled_count > item_count:
        exceptions.append("over_fulfilled")
    if order.refunded_cents > order.captured_cents:
        exceptions.append("refund_exceeds_capture")
    return {
        "status": "exception" if exceptions else "reconciled",
        "exceptions": exceptions,
        "outstanding_cents": outstanding_cents,
        "ordered_items": item_count,
        "fulfilled_items": fulfilled_count,
    }
