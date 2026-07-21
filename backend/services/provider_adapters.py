"""Retry-safe boundaries for real commerce providers.

Each capability is configured with an HTTPS endpoint and bearer token. Providers
must accept the supplied idempotency key; business state is still changed only by
the governed command reducer after a signed webhook or operator command.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


CAPABILITIES = {"inventory", "tax", "payment", "shipping"}
OPERATIONS = {
    "inventory": {"availability", "reserve", "release"},
    "tax": {"quote"},
    "payment": {"authorize", "capture", "refund"},
    "shipping": {"quote", "create", "cancel", "status"},
}


class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class ProviderResult:
    capability: str
    operation: str
    provider_reference: str
    payload: dict[str, Any]


Transport = Callable[[str, bytes, dict[str, str], float], tuple[int, bytes]]


def _http_transport(url: str, body: bytes, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is validated/configured by operator
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()
    except (TimeoutError, URLError) as error:
        raise ProviderError("PROVIDER_UNAVAILABLE", str(error), retryable=True) from error


class ProviderRegistry:
    def __init__(self, transport: Transport = _http_transport, sleep: Callable[[float], None] = time.sleep):
        self.transport = transport
        self.sleep = sleep

    def execute(
        self,
        capability: str,
        operation: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> ProviderResult:
        if capability not in CAPABILITIES or operation not in OPERATIONS.get(capability, set()):
            raise ProviderError("UNSUPPORTED_PROVIDER_OPERATION", f"Unsupported {capability}.{operation}")
        if len(idempotency_key) < 8:
            raise ProviderError("IDEMPOTENCY_REQUIRED", "Provider calls require a stable idempotency key")

        prefix = f"ORDER_{capability.upper()}_PROVIDER"
        endpoint = os.getenv(f"{prefix}_URL", "").rstrip("/")
        token = os.getenv(f"{prefix}_TOKEN", "")
        parsed = urlparse(endpoint)
        allow_http = os.getenv("ALLOW_INSECURE_PROVIDER_HTTP", "false").lower() == "true"
        if not endpoint or not token:
            raise ProviderError("PROVIDER_NOT_CONFIGURED", f"{capability} provider is not configured")
        if parsed.scheme != "https" and not (allow_http and parsed.scheme == "http"):
            raise ProviderError("INSECURE_PROVIDER_URL", "Provider URL must use HTTPS")

        body = json.dumps({"operation": operation, "payload": payload}, separators=(",", ":")).encode()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key,
            "User-Agent": "OrderingServiceText2/1.0",
        }
        retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
        last_error: ProviderError | None = None
        for attempt in range(3):
            try:
                status, raw = self.transport(f"{endpoint}/{operation}", body, headers, 10.0)
            except ProviderError as error:
                last_error = error
                if not error.retryable or attempt == 2:
                    raise
            else:
                if 200 <= status < 300:
                    try:
                        response = json.loads(raw)
                        reference = str(response["id"])
                    except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as error:
                        raise ProviderError("INVALID_PROVIDER_RESPONSE", "Provider response requires a JSON id") from error
                    return ProviderResult(capability, operation, reference, response)
                last_error = ProviderError(
                    "PROVIDER_REJECTED" if status not in retryable_statuses else "PROVIDER_UNAVAILABLE",
                    f"Provider returned HTTP {status}",
                    retryable=status in retryable_statuses,
                )
                if status not in retryable_statuses or attempt == 2:
                    raise last_error
            self.sleep(0.1 * (2**attempt))
        raise last_error or ProviderError("PROVIDER_UNAVAILABLE", "Provider call failed", retryable=True)
