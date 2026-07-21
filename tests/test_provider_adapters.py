import json
import os
import unittest
from unittest.mock import patch

from backend.services.provider_adapters import ProviderError, ProviderRegistry


class ProviderAdapterTests(unittest.TestCase):
    @patch.dict(os.environ, {
        "ORDER_PAYMENT_PROVIDER_URL": "https://payments.example.test/v1",
        "ORDER_PAYMENT_PROVIDER_TOKEN": "test-only-token",
    }, clear=False)
    def test_sends_typed_retry_safe_request(self):
        calls = []

        def transport(url, body, headers, timeout):
            calls.append((url, json.loads(body), headers, timeout))
            if len(calls) == 1:
                return 503, b'{"error":"busy"}'
            return 201, b'{"id":"pay_123","status":"pending"}'

        registry = ProviderRegistry(transport=transport, sleep=lambda _: None)
        result = registry.execute("payment", "authorize", {"amount_cents": 1200}, "order-payment-1")
        self.assertEqual(result.provider_reference, "pay_123")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][2]["Idempotency-Key"], "order-payment-1")
        self.assertEqual(calls[0][1]["operation"], "authorize")

    def test_unconfigured_provider_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ProviderError, "not configured"):
                ProviderRegistry().execute("inventory", "reserve", {}, "inventory-1")

    @patch.dict(os.environ, {
        "ORDER_TAX_PROVIDER_URL": "http://tax.example.test",
        "ORDER_TAX_PROVIDER_TOKEN": "test-only-token",
    }, clear=False)
    def test_non_tls_provider_is_rejected(self):
        with self.assertRaisesRegex(ProviderError, "HTTPS"):
            ProviderRegistry().execute("tax", "quote", {}, "tax-quote-1")

    @patch.dict(os.environ, {
        "ORDER_SHIPPING_PROVIDER_URL": "https://ship.example.test",
        "ORDER_SHIPPING_PROVIDER_TOKEN": "test-only-token",
    }, clear=False)
    def test_non_retryable_failure_is_not_retried(self):
        calls = []

        def transport(*args):
            calls.append(args)
            return 422, b'{}'

        with self.assertRaises(ProviderError) as raised:
            ProviderRegistry(transport=transport, sleep=lambda _: None).execute(
                "shipping", "create", {}, "shipment-1"
            )
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
