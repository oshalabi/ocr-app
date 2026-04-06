from __future__ import annotations

import json
import os
from urllib.request import urlopen

import pytest


BASE_URL = os.getenv("OCR_E2E_BASE_URL")

pytestmark = pytest.mark.skipif(
    not BASE_URL,
    reason="Set OCR_E2E_BASE_URL to run OCR sidecar smoke tests.",
)


def test_health_smoke() -> None:
    with urlopen(f"{BASE_URL.rstrip('/')}/health", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))

    assert payload["status"] == "ok"
