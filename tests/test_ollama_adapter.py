"""Tests for OllamaAdapter — payload building, PDF rendering, config."""
from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError

from ocr_service.ollama_adapter import OllamaAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chat_response(content: str) -> bytes:
    return json.dumps({
        "choices": [
            {
                "message": {
                    "content": content,
                }
            }
        ]
    }).encode("utf-8")


def _fake_urlopen_factory(response_bytes: bytes, status_code: int = 200):
    """Return a context-manager factory that yields a fake HTTP response."""
    class FakeResponse:
        status = status_code

        def read(self):
            return response_bytes

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        return FakeResponse()

    return fake_urlopen


# ---------------------------------------------------------------------------
# Basic configuration
# ---------------------------------------------------------------------------

def test_ollama_adapter_not_configured_when_model_missing(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    adapter = OllamaAdapter()
    assert adapter.is_configured() is False


def test_ollama_adapter_configured_when_model_set(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2-vision")
    adapter = OllamaAdapter()
    assert adapter.is_configured() is True
    assert adapter.provider_name == "ollama"


def test_ollama_adapter_template_model_defaults_to_model(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2-vision")
    monkeypatch.delenv("OLLAMA_TEMPLATE_MODEL", raising=False)
    adapter = OllamaAdapter()
    assert adapter.template_model == "llama3.2-vision"


def test_ollama_adapter_template_model_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2-vision")
    monkeypatch.setenv("OLLAMA_TEMPLATE_MODEL", "qwen2.5vl")
    adapter = OllamaAdapter()
    assert adapter.template_model == "qwen2.5vl"


def test_ollama_adapter_max_pdf_pages_default(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_MAX_PDF_PAGES", raising=False)
    adapter = OllamaAdapter()
    assert adapter.max_pdf_pages == 4


def test_ollama_adapter_max_pdf_pages_custom(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MAX_PDF_PAGES", "2")
    adapter = OllamaAdapter()
    assert adapter.max_pdf_pages == 2


# ---------------------------------------------------------------------------
# extract_fields returns None when not configured
# ---------------------------------------------------------------------------

def test_ollama_extract_skips_when_not_configured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    adapter = OllamaAdapter()
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")
    result = adapter.extract_fields(invoice_path, "NL", ("invoice_number",))
    assert result is None


# ---------------------------------------------------------------------------
# Image upload (JPG/PNG) — single image_url part
# ---------------------------------------------------------------------------

def test_ollama_extract_sends_image_url_part_for_png(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2-vision")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://fake-ollama/v1/chat/completions")

    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    invoice_path = tmp_path / "invoice.png"
    invoice_path.write_bytes(png_bytes)

    captured_payload: dict = {}

    extraction_json = json.dumps({
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

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        import json as _json  # noqa: PLC0415
        captured_payload.update(_json.loads(req.data.decode("utf-8")))

        class FakeResp:
            status = 200
            def read(self):
                return _make_chat_response(extraction_json)
            def getcode(self):
                return self.status
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass

        return FakeResp()

    monkeypatch.setattr("ocr_service.ollama_adapter.request.urlopen", fake_urlopen)

    adapter = OllamaAdapter()
    result = adapter.extract_fields(invoice_path, "NL", ("invoice_number", "date", "amount", "currency_code"))

    assert result is not None
    assert result["fields"]["invoice_number"] == "INV-42"
    assert result["model"] == "llama3.2-vision"

    # Verify the payload structure
    messages = captured_payload["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]

    # First part must be the text prompt
    assert content[0]["type"] == "text"
    assert "invoice_number" in content[0]["text"]

    # Second part must be an image_url with the PNG as a data URL
    assert content[1]["type"] == "image_url"
    data_url = content[1]["image_url"]["url"]
    assert data_url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(data_url.split(",", 1)[1])
    assert decoded == png_bytes

    # OpenRouter-only fields must NOT be present
    assert "plugins" not in captured_payload
    assert "provider" not in captured_payload
    assert "session_id" not in captured_payload
    assert "reasoning" not in captured_payload


def test_ollama_extract_sends_image_url_part_for_jpg(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2-vision")

    jpg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 32
    invoice_path = tmp_path / "invoice.jpg"
    invoice_path.write_bytes(jpg_bytes)

    captured_parts: list = []

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        import json as _json  # noqa: PLC0415
        body = _json.loads(req.data.decode("utf-8"))
        captured_parts.extend(body["messages"][0]["content"])

        class FakeResp:
            status = 200
            def read(self):
                return _make_chat_response(_json.dumps({
                    "fields": {"invoice_number": "TEST-1"},
                    "confidence": {"invoice_number": 0.9},
                }))
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass

        return FakeResp()

    monkeypatch.setattr("ocr_service.ollama_adapter.request.urlopen", fake_urlopen)

    adapter = OllamaAdapter()
    adapter.extract_fields(invoice_path, "NL", ("invoice_number",))

    image_parts = [p for p in captured_parts if p["type"] == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# PDF handling — images rendered, OpenRouter file/plugin NOT used
# ---------------------------------------------------------------------------

def test_ollama_extract_renders_pdf_to_images_not_file_upload(monkeypatch, tmp_path: Path) -> None:
    """OllamaAdapter must send image_url parts for PDFs, not OpenRouter file parts."""
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2-vision")
    monkeypatch.setenv("OLLAMA_MAX_PDF_PAGES", "2")

    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")

    fake_png_page = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    rendered_pages: list[bytes] = []

    # Patch _pdf_to_image_parts to avoid needing PyMuPDF in tests
    def fake_pdf_to_image_parts(self, file_path, session_id=None):  # noqa: ARG001
        parts = []
        for i in range(2):
            page_bytes = fake_png_page + bytes([i])
            rendered_pages.append(page_bytes)
            parts.append(self._image_url_part_from_bytes(page_bytes, "image/png"))
        return parts

    monkeypatch.setattr(OllamaAdapter, "_pdf_to_image_parts", fake_pdf_to_image_parts)

    captured_payload: dict = {}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        import json as _json  # noqa: PLC0415
        captured_payload.update(_json.loads(req.data.decode("utf-8")))

        class FakeResp:
            status = 200
            def read(self):
                return _make_chat_response(_json.dumps({
                    "fields": {"invoice_number": "INV-1"},
                    "confidence": {"invoice_number": 0.9},
                }))
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass

        return FakeResp()

    monkeypatch.setattr("ocr_service.ollama_adapter.request.urlopen", fake_urlopen)

    adapter = OllamaAdapter()
    result = adapter.extract_fields(invoice_path, "NL", ("invoice_number",))

    assert result is not None

    content = captured_payload["messages"][0]["content"]
    image_parts = [p for p in content if p["type"] == "image_url"]

    # Must be 2 image parts (one per rendered page), not a "file" part
    assert len(image_parts) == 2
    file_parts = [p for p in content if p["type"] == "file"]
    assert file_parts == []

    # No OpenRouter-only fields
    assert "plugins" not in captured_payload
    assert "provider" not in captured_payload


def test_ollama_pdf_truncation_logs_when_pages_exceed_limit(monkeypatch, tmp_path: Path, capsys) -> None:
    """When PDF has more pages than max, adapter renders only max and logs truncation."""
    from ocr_service import logger as logger_module  # noqa: PLC0415
    monkeypatch.setenv("OCR_DEBUG", "true")
    logger_module.configure_logging(force=True)
    capsys.readouterr()

    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2-vision")
    monkeypatch.setenv("OLLAMA_MAX_PDF_PAGES", "2")

    import sys  # noqa: PLC0415

    # Build a minimal fake fitz module
    class FakePixmap:
        def tobytes(self, fmt):  # noqa: ARG002
            return b"\x89PNG\r\n\x1a\n" + b"\x00" * 8

    class FakePage:
        def get_pixmap(self, matrix=None):  # noqa: ARG002
            return FakePixmap()

    class FakeDoc:
        def __init__(self):
            self._pages = [FakePage(), FakePage(), FakePage(), FakePage(), FakePage()]

        def __len__(self):
            return len(self._pages)

        def __getitem__(self, index):
            return self._pages[index]

        def close(self):
            pass

    class FakeFitz:
        class Matrix:
            def __init__(self, *_args):
                pass

        @staticmethod
        def open(_path):
            return FakeDoc()

    monkeypatch.setitem(sys.modules, "fitz", FakeFitz())

    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")

    adapter = OllamaAdapter()
    parts = adapter._pdf_to_image_parts(invoice_path, session_id="test-session")

    # Only 2 pages rendered out of 5
    assert len(parts) == 2

    logs = [
        json.loads(line)
        for line in capsys.readouterr().err.splitlines()
        if line.strip()
    ]
    truncation_logs = [
        entry for entry in logs
        if entry.get("action") == "ollama.pdf_render.truncated"
    ]
    assert truncation_logs
    assert truncation_logs[0]["context"]["total_pages"] == 5
    assert truncation_logs[0]["context"]["rendered_pages"] == 2


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------

def test_ollama_extract_returns_none_on_http_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2-vision")

    invoice_path = tmp_path / "invoice.png"
    invoice_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        raise HTTPError("http://fake", 503, "Service Unavailable", {}, None)

    monkeypatch.setattr("ocr_service.ollama_adapter.request.urlopen", fake_urlopen)

    adapter = OllamaAdapter()
    result = adapter.extract_fields(invoice_path, "NL", ("invoice_number",))
    assert result is None


def test_ollama_extract_returns_none_on_url_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2-vision")

    invoice_path = tmp_path / "invoice.png"
    invoice_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        raise URLError("connection refused")

    monkeypatch.setattr("ocr_service.ollama_adapter.request.urlopen", fake_urlopen)

    adapter = OllamaAdapter()
    result = adapter.extract_fields(invoice_path, "NL", ("invoice_number",))
    assert result is None


# ---------------------------------------------------------------------------
# generate_template_definition
# ---------------------------------------------------------------------------

def test_ollama_generate_template_returns_parsed_dict(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2-vision")

    invoice_path = tmp_path / "invoice.png"
    invoice_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    template_dict = {
        "issuer": "Acme B.V.",
        "keywords": ["Acme"],
        "fields": {
            "invoice_number": "Factuurnummer\\s*(\\S+)",
        },
        "options": {
            "date_formats": ["%d-%m-%Y"],
            "remove_whitespace": True,
        },
    }

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        class FakeResp:
            status = 200
            def read(self):
                return _make_chat_response(json.dumps(template_dict))
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass

        return FakeResp()

    monkeypatch.setattr("ocr_service.ollama_adapter.request.urlopen", fake_urlopen)

    adapter = OllamaAdapter()
    result = adapter.generate_template_definition(invoice_path, "NL", ("invoice_number",))

    assert result == template_dict


def test_ollama_generate_template_returns_none_on_invalid_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2-vision")

    invoice_path = tmp_path / "invoice.png"
    invoice_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        class FakeResp:
            status = 200
            def read(self):
                return _make_chat_response("not-json")
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass

        return FakeResp()

    monkeypatch.setattr("ocr_service.ollama_adapter.request.urlopen", fake_urlopen)

    adapter = OllamaAdapter()
    result = adapter.generate_template_definition(invoice_path, "NL", ("invoice_number",))
    assert result is None
