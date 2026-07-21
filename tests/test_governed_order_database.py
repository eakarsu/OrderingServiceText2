import hashlib
import hmac
import json
import os
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.getenv("RUN_DB_TESTS") == "1", "set RUN_DB_TESTS=1 for PostgreSQL HTTP integration")
class GovernedOrderDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg2
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from backend.database import get_db
        from backend.dependencies import get_current_user
        from backend.routers.governed_orders import router

        database_url = os.environ["TEST_DATABASE_URL"]
        cls.database_url = database_url

        def database_dependency():
            connection = psycopg2.connect(database_url)
            try:
                yield connection
            finally:
                connection.close()

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_db] = database_dependency
        app.dependency_overrides[get_current_user] = lambda: {"id": 99, "role": "admin"}
        cls.client = TestClient(app)

    def command(self, order_id, command, payload, key):
        return self.client.post(
            f"/api/governed-orders/{order_id}/commands",
            headers={"Idempotency-Key": key},
            json={"command": command, "payload": payload},
        )

    def test_persisted_payment_webhook_partial_fulfillment_and_audit(self):
        created = self.client.post("/api/governed-orders", json={
            "customer_id": 10,
            "merchant_id": 20,
            "currency": "usd",
            "total_cents": 2500,
            "items": {"pizza": 2, "soda": 1},
        })
        self.assertEqual(created.status_code, 201, created.text)
        order_id = created.json()["id"]

        reserved = self.command(order_id, "reserve", {"available": {"pizza": 2, "soda": 1}}, "reserve-http-1")
        self.assertEqual(reserved.status_code, 200, reserved.text)
        payment = self.command(order_id, "request_payment", {}, "payment-request-http-1")
        self.assertEqual(payment.json()["state"], "payment_pending")

        event = json.dumps({
            "id": "payment-event-http-1",
            "order_id": order_id,
            "type": "payment.succeeded",
            "created_at": "2026-07-19T12:00:00Z",
            "payload": {"amount_cents": 2500},
        }, separators=(",", ":")).encode()
        secret = "integration-webhook-secret"
        signature = hmac.new(secret.encode(), event, hashlib.sha256).hexdigest()
        with patch.dict(os.environ, {"ORDER_WEBHOOK_PAYMENT_SECRET": secret}):
            first = self.client.post(
                "/api/governed-orders/webhooks/payment",
                content=event,
                headers={"Content-Type": "application/json", "X-Webhook-Signature": signature},
            )
            replay = self.client.post(
                "/api/governed-orders/webhooks/payment",
                content=event,
                headers={"Content-Type": "application/json", "X-Webhook-Signature": signature},
            )
        self.assertEqual(first.status_code, 200, first.text)
        self.assertTrue(replay.json()["duplicate"])

        started = self.command(order_id, "start_fulfillment", {}, "fulfillment-start-http-1")
        self.assertEqual(started.json()["state"], "fulfillment_pending")
        partial = self.command(order_id, "fulfill_items", {"quantities": {"pizza": 1}}, "partial-http-1")
        self.assertEqual(partial.json()["state"], "partially_fulfilled")

        audit = self.client.get(f"/api/governed-orders/{order_id}/audit")
        self.assertEqual(audit.status_code, 200, audit.text)
        self.assertEqual([event["command"] for event in audit.json()["events"]], [
            "create", "reserve", "request_payment", "payment_succeeded",
            "start_fulfillment", "fulfill_items",
        ])

        import psycopg2

        connection = psycopg2.connect(self.database_url)
        try:
            with self.assertRaises(psycopg2.errors.RaiseException):
                with connection.cursor() as cursor:
                    cursor.execute(
                        "UPDATE governed_order_events SET actor_role='tampered' WHERE order_id=%s",
                        (order_id,),
                    )
            connection.rollback()
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
