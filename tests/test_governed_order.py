import hashlib
import hmac
import unittest

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


class GovernedOrderTests(unittest.TestCase):
    def setUp(self):
        self.order = OrderSnapshot(
            order_id="order-1",
            customer_id=10,
            merchant_id=20,
            currency="USD",
            total_cents=2500,
            items=(("pizza", 2), ("soda", 1)),
        )
        self.customer = Actor(10, "staff")
        self.merchant = Actor(20, "manager")
        self.operator = Actor(99, "admin")

    def command(self, order, actor, command, payload=None, key=None):
        return apply_command(order, actor, command, payload or {}, key or f"key-{command}")[0]

    def test_order_validation_rejects_duplicate_or_invalid_items(self):
        validate_new_order(self.order)
        with self.assertRaisesRegex(OrderControlError, "duplicate"):
            validate_new_order(OrderSnapshot(**{**self.order.__dict__, "items": (("pizza", 1), ("pizza", 2))}))

    def test_overselling_is_blocked(self):
        with self.assertRaisesRegex(OrderControlError, "pizza"):
            self.command(self.order, self.merchant, "reserve", {"available": {"pizza": 1, "soda": 1}})

    def test_payment_failure_and_recovery_are_explicit(self):
        order = self.command(self.order, self.merchant, "reserve", {"available": {"pizza": 2, "soda": 1}})
        order = self.command(order, self.customer, "request_payment")
        order = self.command(order, self.operator, "payment_failed")
        self.assertEqual(order.state, "exception")
        order = self.command(order, self.operator, "recover", {"target_state": "reserved"})
        self.assertEqual(order.state, "reserved")

    def test_duplicate_command_is_idempotent(self):
        order, first = apply_command(self.order, self.merchant, "reserve", {"available": {"pizza": 2, "soda": 1}}, "provider-event-1")
        replay, duplicate = apply_command(order, self.merchant, "reserve", {"available": {}}, "provider-event-1")
        self.assertFalse(first["duplicate"])
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(replay, order)

    def test_partial_fulfillment_cannot_exceed_reservation(self):
        order = self.command(self.order, self.merchant, "reserve", {"available": {"pizza": 2, "soda": 1}})
        order = self.command(order, self.customer, "request_payment")
        order = self.command(order, self.operator, "payment_succeeded", {"amount_cents": 2500})
        order = self.command(order, self.merchant, "start_fulfillment")
        order = self.command(order, self.merchant, "fulfill_items", {"quantities": {"pizza": 1}}, "partial-1")
        self.assertEqual(order.state, "partially_fulfilled")
        with self.assertRaisesRegex(OrderControlError, "exceeds"):
            self.command(order, self.merchant, "fulfill_items", {"quantities": {"pizza": 2}}, "partial-2")

    def test_refunds_cannot_exceed_capture(self):
        order = self.command(self.order, self.merchant, "reserve", {"available": {"pizza": 2, "soda": 1}})
        order = self.command(order, self.customer, "request_payment")
        order = self.command(order, self.operator, "payment_succeeded", {"amount_cents": 2500})
        with self.assertRaisesRegex(OrderControlError, "unrefunded"):
            self.command(order, self.customer, "request_refund", {"amount_cents": 2501})

    def test_roles_and_ownership_are_enforced(self):
        with self.assertRaisesRegex(OrderControlError, "another merchant"):
            self.command(self.order, Actor(21, "manager"), "reserve", {"available": {"pizza": 2, "soda": 1}})
        with self.assertRaisesRegex(OrderControlError, "cannot execute"):
            self.command(self.order, self.customer, "reserve", {"available": {"pizza": 2, "soda": 1}})

    def test_webhook_signature_and_key_are_deterministic(self):
        body = b'{"id":"evt-1"}'
        signature = hmac.new(b"secret", body, hashlib.sha256).hexdigest()
        self.assertTrue(verify_webhook_signature(body, f"sha256={signature}", "secret"))
        self.assertFalse(verify_webhook_signature(body, "bad", "secret"))
        self.assertEqual(idempotency_key("Stripe", " EVT-1 "), idempotency_key("stripe", "evt-1"))

    def test_reconciliation_exposes_financial_and_fulfillment_state(self):
        result = reconciliation(self.order)
        self.assertEqual(result["status"], "reconciled")
        self.assertEqual(result["ordered_items"], 3)
        self.assertEqual(result["outstanding_cents"], 0)


if __name__ == "__main__":
    unittest.main()
