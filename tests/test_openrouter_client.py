import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

from ocr_service import logger as logger_module
from ocr_service import openrouter_client as openrouter_client_module
from ocr_service.openrouter_client import OpenRouterClient, OpenRouterRequestResult


def test_openrouter_extract_fields_builds_pdf_request_with_native_file_upload(monkeypatch, tmp_path: Path) -> None:
    client = OpenRouterClient()
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")
    captured_payload = {}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/model")
    monkeypatch.setenv("OPENROUTER_PDF_ENGINE", "native")
    monkeypatch.setenv("OPENROUTER_REQUIRE_ZDR", "true")
    monkeypatch.setenv("OPENROUTER_DISABLE_REASONING", "true")
    client = OpenRouterClient()

    def fake_request(payload):
        captured_payload["payload"] = payload
        return OpenRouterRequestResult(
            payload={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "issuer": "Acme B.V.",
                                "fields": {
                                    "invoice_number": "INV-42",
                                    "date": "2026-04-03",
                                    "amount": 123.45,
                                    "currency_code": "EUR",
                                },
                                "confidence": {
                                    "invoice_number": 0.91,
                                    "date": 0.92,
                                    "amount": 0.93,
                                    "currency_code": 0.94,
                                },
                            })
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(client, "_request", fake_request)

    response = client.extract_fields(
        invoice_path,
        country_code="NL",
        required_fields=("invoice_number", "date", "amount", "currency_code"),
        session_id="ocr-session-1",
    )

    assert response["fields"]["invoice_number"] == "INV-42"
    assert captured_payload["payload"]["provider"]["zdr"] is True
    assert captured_payload["payload"]["reasoning"] == {
        "effort": "none",
        "exclude": True,
    }
    assert captured_payload["payload"]["session_id"] == "ocr-session-1"
    assert "plugins" not in captured_payload["payload"]
    assert "response_format" in captured_payload["payload"]
    assert captured_payload["payload"]["messages"][0]["content"][1]["type"] == "file"


def test_openrouter_extract_fields_builds_image_request(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/model")
    monkeypatch.setenv("OPENROUTER_DISABLE_REASONING", "true")
    client = OpenRouterClient()
    invoice_path = tmp_path / "invoice.png"
    invoice_path.write_bytes(b"\x89PNG")
    captured_payload = {}

    def fake_request(payload):
        captured_payload["payload"] = payload

        return OpenRouterRequestResult(
            payload={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "issuer": None,
                                "fields": {
                                    "invoice_number": None,
                                    "date": None,
                                    "amount": None,
                                    "currency_code": None,
                                },
                                "confidence": {
                                    "invoice_number": 0,
                                    "date": 0,
                                    "amount": 0,
                                    "currency_code": 0,
                                },
                            })
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(client, "_request", fake_request)

    client.extract_fields(
        invoice_path,
        country_code="NL",
        required_fields=("invoice_number", "date", "amount", "currency_code"),
    )

    assert captured_payload["payload"]["messages"][0]["content"][1]["type"] == "image_url"
    assert "image or scan" in captured_payload["payload"]["messages"][0]["content"][0]["text"]
    assert captured_payload["payload"]["reasoning"] == {
        "effort": "medium",
    }
    assert "plugins" not in captured_payload["payload"]


def test_openrouter_extract_fields_accepts_flat_json_and_defaults_confidence(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/model")
    client = OpenRouterClient()
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")
    captured_payload = {}

    def fake_request(payload):
        captured_payload["payload"] = payload

        return OpenRouterRequestResult(
            payload={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "issuer": "Ozer Logistics BV",
                                "invoice_number": "202502224",
                                "date": "2025-03-07",
                                "amount": 3345.21,
                                "currency_code": "EUR",
                            })
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(client, "_request", fake_request)

    response = client.extract_fields(
        invoice_path,
        country_code="NL",
        required_fields=("invoice_number", "date", "amount", "currency_code"),
        session_id="ocr-session-2",
    )

    assert response == {
        "fields": {
            "invoice_number": "202502224",
            "date": "2025-03-07",
            "amount": 3345.21,
            "currency_code": "EUR",
        },
        "confidence": {
            "invoice_number": 0.95,
            "date": 0.95,
            "amount": 0.95,
            "currency_code": 0.95,
        },
        "issuer": "Ozer Logistics BV",
        "model": "openrouter/model",
    }
    assert captured_payload["payload"]["session_id"] == "ocr-session-2"


def test_openrouter_extract_fields_retries_without_response_format_after_http_400(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/model")
    monkeypatch.setenv("OPENROUTER_PDF_ENGINE", "native")
    monkeypatch.setenv("OPENROUTER_REQUIRE_ZDR", "false")
    client = OpenRouterClient()
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")
    captured_payloads = []

    def fake_request(payload):
        captured_payloads.append(payload)

        if len(captured_payloads) == 1:
            return OpenRouterRequestResult(
                payload=None,
                reason="http_error",
                status_code=400,
                error_body="Invalid response_format for provider",
            )

        return OpenRouterRequestResult(
            payload={
                "choices": [
                    {
                        "message": {
                            "content": """```json
                            {
                              "issuer": "Acme B.V.",
                              "fields": {
                                "invoice_number": "INV-42",
                                "date": "2026-04-03",
                                "amount": 123.45,
                                "currency_code": "EUR"
                              },
                              "confidence": {
                                "invoice_number": 0.91,
                                "date": 0.92,
                                "amount": 0.93,
                                "currency_code": 0.94
                              }
                            }
                            ```"""
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(client, "_request", fake_request)

    response = client.extract_fields(
        invoice_path,
        country_code="NL",
        required_fields=("invoice_number", "date", "amount", "currency_code"),
    )

    assert response["fields"]["invoice_number"] == "INV-42"
    assert len(captured_payloads) == 2
    assert "response_format" in captured_payloads[0]
    assert "response_format" not in captured_payloads[1]
    assert "plugins" not in captured_payloads[0]
    assert "plugins" not in captured_payloads[1]


def test_openrouter_generate_template_definition_uses_template_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/extract-model")
    monkeypatch.setenv("OPENROUTER_TEMPLATE_MODEL", "openrouter/template-model")
    client = OpenRouterClient()
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")
    captured_payload = {}

    def fake_request(payload):
        captured_payload["payload"] = payload

        return OpenRouterRequestResult(
            payload={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "issuer": "Acme B.V.",
                                "keywords": ["(?i)Acme\\s+B\\.V\\."],
                                "fields": {
                                    "invoice_number": "(INV-42)",
                                    "date": "(2026-04-03)",
                                    "amount": "(123.45)",
                                    "currency_code": "(EUR)",
                                },
                                "options": {
                                    "date_formats": ["%Y-%m-%d"],
                                    "remove_whitespace": True,
                                },
                            })
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(client, "_request", fake_request)

    response = client.generate_template_definition(
        invoice_path,
        country_code="NL",
        required_fields=("invoice_number", "date", "amount", "currency_code"),
    )

    assert response["issuer"] == "Acme B.V."
    assert captured_payload["payload"]["model"] == "openrouter/template-model"


def test_openrouter_request_logs_sanitized_http_error_body(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/model")
    monkeypatch.setenv("OCR_DEBUG", "true")
    logger_module.configure_logging(force=True)
    capsys.readouterr()

    client = OpenRouterClient()

    def fake_urlopen(*_args, **_kwargs):
        raise HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=BytesIO(b'{"error":{"message":"Invalid plugins[0].pdf.engine value"}}'),
        )

    monkeypatch.setattr(openrouter_client_module.request, "urlopen", fake_urlopen)

    result = client._request({"model": "openrouter/model", "messages": []})

    captured = capsys.readouterr().err
    log_lines = [json.loads(line) for line in captured.splitlines() if line.strip()]

    assert result.payload is None
    assert result.status_code == 400
    assert any(
        entry["action"] == "openrouter.request.failed"
        and entry["context"].get("error_body") == "Invalid plugins[0].pdf.engine value"
        for entry in log_lines
    )


def test_openrouter_request_retries_http_429_with_backoff(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/model")
    monkeypatch.setenv("OPENROUTER_RATE_LIMIT_RETRIES", "1")
    monkeypatch.setenv("OPENROUTER_RATE_LIMIT_BACKOFF_MS", "150")
    client = OpenRouterClient()
    calls = {"count": 0}
    sleep_calls: list[float] = []

    class FakeResponse:
        status = 200

        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def getcode(self) -> int:
            return self.status

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(*_args, **_kwargs):
        calls["count"] += 1

        if calls["count"] == 1:
            raise HTTPError(
                url="https://openrouter.ai/api/v1/chat/completions",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=BytesIO(b'{"error":{"message":"Rate limit exceeded"}}'),
            )

        return FakeResponse({"choices": []})

    monkeypatch.setattr(openrouter_client_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(openrouter_client_module, "sleep", lambda seconds: sleep_calls.append(seconds))

    result = client._request({
        "model": "openrouter/model",
        "messages": [],
        "session_id": "ocr-session-3",
    })

    assert result.payload == {"choices": []}
    assert calls["count"] == 2
    assert sleep_calls == [0.15]
