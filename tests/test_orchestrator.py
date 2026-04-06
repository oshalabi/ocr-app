import json
from pathlib import Path

from ocr_service.extractor import OcrExtractor
from ocr_service.orchestrator import OcrOrchestrator, OcrProcessOptions
from ocr_service import orchestrator as orchestrator_module
from ocr_service import logger as logger_module


# ---------------------------------------------------------------------------
# Fake adapter — satisfies the LlmAdapter protocol without any imports
# ---------------------------------------------------------------------------

class FakeLlmAdapter:
    """Provider-neutral test double for LlmAdapter."""

    def __init__(self, extraction_response=None, template_response=None) -> None:
        self.extraction_response = extraction_response
        self.template_response = template_response
        self.extraction_calls = 0
        self.template_calls = 0
        self.template_contexts: list[str | None] = []
        self.extraction_kwargs: list[dict] = []
        self.template_kwargs: list[dict] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def is_configured(self) -> bool:
        return True

    def extract_fields(self, *_args, **_kwargs):
        self.extraction_calls += 1
        self.extraction_kwargs.append(_kwargs)
        return self.extraction_response

    def generate_template_definition(self, *_args, **kwargs):
        self.template_calls += 1
        self.template_contexts.append(kwargs.get("correction_context"))
        self.template_kwargs.append(kwargs)

        if isinstance(self.template_response, list):
            return self.template_response[self.template_calls - 1]

        return self.template_response


# ---------------------------------------------------------------------------
# Orchestrator: local success path
# ---------------------------------------------------------------------------

def test_orchestrator_returns_local_success_without_llm_fallback(monkeypatch, tmp_path: Path) -> None:
    extractor = OcrExtractor(tmp_path)
    adapter = FakeLlmAdapter()
    service = OcrOrchestrator(extractor, adapter)
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")

    expected_response = {
        "status": "success",
        "message": "Invoice fields extracted successfully.",
        "fields": {
            "invoice_number": {"value": "INV-42", "confidence": 0.98},
            "invoice_date": {"value": "2026-04-03", "confidence": 0.98},
            "total_amount": {"value": 123.45, "confidence": 0.98},
            "currency_code": {"value": "EUR", "confidence": 0.98},
        },
        "unmapped_fields": {},
    }

    monkeypatch.setattr(extractor, "extract", lambda *_args, **_kwargs: expected_response)

    response = service.extract(invoice_path, OcrProcessOptions(llm_fallback_enabled=True))

    assert response == expected_response
    assert adapter.extraction_calls == 0
    assert adapter.template_calls == 0


# backward-compat: openrouter_enabled alias still works
def test_orchestrator_openrouter_enabled_alias_still_accepted(monkeypatch, tmp_path: Path) -> None:
    extractor = OcrExtractor(tmp_path)
    adapter = FakeLlmAdapter()
    service = OcrOrchestrator(extractor, adapter)
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")

    expected_response = {
        "status": "success",
        "message": "Invoice fields extracted successfully.",
        "fields": {
            "invoice_number": {"value": "INV-42", "confidence": 0.98},
        },
        "unmapped_fields": {},
    }

    monkeypatch.setattr(extractor, "extract", lambda *_args, **_kwargs: expected_response)

    # Pass the old name — the dataclass maps it to llm_fallback_enabled
    options = OcrProcessOptions(openrouter_enabled=True)
    assert options.llm_fallback_enabled is True

    response = service.extract(invoice_path, options)
    assert response == expected_response


# ---------------------------------------------------------------------------
# Merge partial local + LLM fields
# ---------------------------------------------------------------------------

def test_orchestrator_merges_missing_required_fields_from_llm(monkeypatch, tmp_path: Path) -> None:
    extractor = OcrExtractor(tmp_path)
    adapter = FakeLlmAdapter(
        extraction_response={
            "fields": {
                "currency_code": "EUR",
            },
            "confidence": {
                "currency_code": 0.92,
            },
            "issuer": "Acme B.V.",
            "model": "fake/model",
        }
    )
    service = OcrOrchestrator(extractor, adapter)
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")

    monkeypatch.setattr(
        extractor,
        "extract",
        lambda *_args, **_kwargs: {
            "status": "partial",
            "message": "Invoice matched a template, but some required fields are missing.",
            "fields": {
                "invoice_number": {"value": "INV-42", "confidence": 0.98},
                "invoice_date": {"value": "2026-04-03", "confidence": 0.98},
                "total_amount": {"value": 123.45, "confidence": 0.98},
            },
            "unmapped_fields": {
                "source_reader": {"value": "tesseract", "confidence": 1.0},
                "missing_required_fields": {"value": ["currency_code"], "confidence": 1.0},
            },
        },
    )

    response = service.extract(
        invoice_path,
        OcrProcessOptions(
            llm_fallback_enabled=True,
            required_fields=("invoice_number", "date", "amount", "currency_code"),
        ),
    )

    assert response["status"] == "success"
    assert response["fields"]["currency_code"]["value"] == "EUR"
    assert response["unmapped_fields"]["fallback_source_reader"]["value"] == "fake"


# ---------------------------------------------------------------------------
# Template generation / healing
# ---------------------------------------------------------------------------

def test_orchestrator_saves_validated_ai_template_and_retries_local_extraction(monkeypatch, tmp_path: Path) -> None:
    extractor = OcrExtractor(tmp_path)
    adapter = FakeLlmAdapter(
        extraction_response={
            "fields": {
                "invoice_number": "INV-42",
                "date": "2026-04-03",
                "amount": 123.45,
                "currency_code": "EUR",
            },
            "confidence": {
                "invoice_number": 0.91,
                "date": 0.90,
                "amount": 0.94,
                "currency_code": 0.95,
            },
            "issuer": "Acme B.V.",
            "model": "fake/model",
        },
        template_response={
            "issuer": "Acme B.V.",
            "keywords": ["(?i)Acme\\s+B\\.V\\."],
            "fields": {
                "invoice_number": {
                    "regex": "(?i)Factuurnummer\\s*[:#-]?\\s*((?=[A-Z0-9/._-]*\\d)[A-Z0-9][A-Z0-9/._-]+)"
                },
                "date": {
                    "regex": "(?i)Factuurdatum\\s*[:#-]?\\s*([0-9]{2}-[0-9]{2}-[0-9]{4})"
                },
                "amount": {
                    "regex": "(?i)Totaal\\s*[:#-]?\\s*(?:EUR|€)?\\s*([0-9.,]+)"
                },
                "currency_code": {
                    "regex": "(?i)((?:EUR|€))"
                },
            },
            "options": {
                "date_formats": ["%d-%m-%Y"],
                "remove_whitespace": True,
            },
        },
    )
    service = OcrOrchestrator(extractor, adapter)
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")

    responses = iter([
        {
            "status": "failed",
            "message": "No matching invoice template found for this document.",
            "fields": {},
            "unmapped_fields": {},
        },
        {
            "status": "success",
            "message": "Invoice fields extracted successfully.",
            "fields": {
                "invoice_number": {"value": "INV-42", "confidence": 0.98},
                "invoice_date": {"value": "2026-04-03", "confidence": 0.98},
                "total_amount": {"value": 123.45, "confidence": 0.98},
                "currency_code": {"value": "EUR", "confidence": 0.98},
            },
            "unmapped_fields": {},
        },
    ])

    monkeypatch.setattr(extractor, "extract", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(orchestrator_module, "validate_ai_template_definition", lambda **_kwargs: True)

    response = service.extract(
        invoice_path,
        OcrProcessOptions(
            llm_fallback_enabled=True,
            auto_generate_templates=True,
        ),
    )

    created_templates = list(tmp_path.rglob("template_ai*.yml"))

    assert response["status"] == "success"
    assert created_templates
    assert adapter.template_calls == 1


def test_orchestrator_backfills_missing_template_issuer_from_llm_extraction(monkeypatch, tmp_path: Path) -> None:
    extractor = OcrExtractor(tmp_path)
    adapter = FakeLlmAdapter(
        extraction_response={
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
            "model": "fake/model",
        },
        template_response={
            "keywords": ["Ozer Logistics BV", "NL822257750B01"],
            "fields": {
                "invoice_number": r"INVOICE NUMBER\s*(\S+)",
                "date": r"INVOICE DATE\s*(\d{2}-\w{3}-\d{4})",
                "amount": r"Total\s+\w{3}\s+([\d,]+\.\d{2})",
                "currency_code": r"Total\s+(\w{3})\s+[\d,]+\.\d{2}",
            },
            "options": {
                "date_formats": ["%d-%m-%Y"],
                "remove_whitespace": True,
            },
        },
    )
    service = OcrOrchestrator(extractor, adapter)
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")

    responses = iter([
        {
            "status": "failed",
            "message": "No matching invoice template found for this document.",
            "fields": {},
            "unmapped_fields": {},
        },
        {
            "status": "success",
            "message": "Invoice fields extracted successfully.",
            "fields": {
                "invoice_number": {"value": "202502224", "confidence": 0.98},
                "invoice_date": {"value": "2025-03-07", "confidence": 0.98},
                "total_amount": {"value": 3345.21, "confidence": 0.98},
                "currency_code": {"value": "EUR", "confidence": 0.98},
            },
            "unmapped_fields": {},
        },
    ])

    monkeypatch.setattr(extractor, "extract", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(orchestrator_module, "validate_ai_template_definition", lambda **_kwargs: True)

    response = service.extract(
        invoice_path,
        OcrProcessOptions(
            llm_fallback_enabled=True,
            auto_generate_templates=True,
        ),
    )

    created_templates = list(tmp_path.rglob("template_ai*.yml"))

    assert response["status"] == "success"
    assert created_templates
    assert "issuer: 'Ozer Logistics BV'" in created_templates[0].read_text(encoding="utf-8")


def test_orchestrator_returns_llm_response_when_ai_template_validation_fails(monkeypatch, tmp_path: Path) -> None:
    extractor = OcrExtractor(tmp_path)
    adapter = FakeLlmAdapter(
        extraction_response={
            "fields": {
                "invoice_number": "INV-42",
                "date": "2026-04-03",
                "amount": 123.45,
                "currency_code": "EUR",
            },
            "confidence": {
                "invoice_number": 0.91,
                "date": 0.90,
                "amount": 0.94,
                "currency_code": 0.95,
            },
            "issuer": "Acme B.V.",
            "model": "fake/model",
        },
        template_response={
            "issuer": "Acme B.V.",
            "keywords": ["INV-42"],
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
        },
    )
    service = OcrOrchestrator(extractor, adapter)
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")

    monkeypatch.setattr(
        extractor,
        "extract",
        lambda *_args, **_kwargs: {
            "status": "failed",
            "message": "No matching invoice template found for this document.",
            "fields": {},
            "unmapped_fields": {},
        },
    )
    monkeypatch.setattr(orchestrator_module, "validate_ai_template_definition", lambda **_kwargs: False)

    response = service.extract(
        invoice_path,
        OcrProcessOptions(
            llm_fallback_enabled=True,
            auto_generate_templates=True,
        ),
    )

    assert response["status"] == "success"
    assert response["fields"]["invoice_number"]["value"] == "INV-42"
    assert list(tmp_path.rglob("template_ai*.yml")) == []


def test_orchestrator_corrects_generated_template_until_retry_succeeds(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OCR_TEMPLATE_HEALING_MAX_ATTEMPTS", "3")

    extractor = OcrExtractor(tmp_path)
    adapter = FakeLlmAdapter(
        extraction_response={
            "fields": {
                "invoice_number": "INV-42",
                "date": "2026-04-03",
                "amount": 123.45,
                "currency_code": "EUR",
            },
            "confidence": {
                "invoice_number": 0.91,
                "date": 0.90,
                "amount": 0.94,
                "currency_code": 0.95,
            },
            "issuer": "Acme B.V.",
            "model": "fake/model",
        },
        template_response=[
            {
                "issuer": "Acme B.V.",
                "keywords": ["(?i)Acme\\s+B\\.V\\."],
                "fields": {
                    "invoice_number": "(?i)Factuurnummer\\s*[:#-]?\\s*((?=[A-Z0-9/._-]*\\d)[A-Z0-9][A-Z0-9/._-]+)",
                    "date": "(?i)Factuurdatum\\s*[:#-]?\\s*([0-9]{2}-[0-9]{2}-[0-9]{4})",
                    "amount": "(?i)Totaal\\s*[:#-]?\\s*(?:EUR|€)?\\s*([0-9.,]+)",
                    "currency_code": "(?i)((?:EUR|€))",
                },
                "options": {
                    "date_formats": ["%d-%m-%Y"],
                    "remove_whitespace": True,
                },
            },
            {
                "issuer": "Acme B.V.",
                "keywords": ["(?i)Acme\\s+B\\.V\\."],
                "fields": {
                    "invoice_number": "(?i)Factuurnummer\\s*[:#-]?\\s*((?=[A-Z0-9/._-]*\\d)[A-Z0-9][A-Z0-9/._-]+)",
                    "date": "(?i)Factuurdatum\\s*[:#-]?\\s*([0-9]{2}-[0-9]{2}-[0-9]{4})",
                    "amount": "(?i)Bedrag\\s*[:#-]?\\s*(?:EUR|€)?\\s*([0-9.,]+)",
                    "currency_code": "(?i)((?:EUR|€))",
                },
                "options": {
                    "date_formats": ["%d-%m-%Y"],
                    "remove_whitespace": True,
                },
            },
        ],
    )
    service = OcrOrchestrator(extractor, adapter)
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")

    responses = iter([
        {
            "status": "failed",
            "message": "No matching invoice template found for this document.",
            "fields": {},
            "unmapped_fields": {},
        },
        {
            "status": "partial",
            "message": "Invoice matched a template, but some required fields are missing.",
            "fields": {
                "invoice_number": {"value": "INV-42", "confidence": 0.98},
                "invoice_date": {"value": "2026-04-03", "confidence": 0.98},
                "currency_code": {"value": "EUR", "confidence": 0.98},
            },
            "unmapped_fields": {
                "missing_required_fields": {"value": ["total_amount"], "confidence": 1.0},
            },
        },
        {
            "status": "success",
            "message": "Invoice fields extracted successfully.",
            "fields": {
                "invoice_number": {"value": "INV-42", "confidence": 0.98},
                "invoice_date": {"value": "2026-04-03", "confidence": 0.98},
                "total_amount": {"value": 123.45, "confidence": 0.98},
                "currency_code": {"value": "EUR", "confidence": 0.98},
            },
            "unmapped_fields": {},
        },
    ])

    monkeypatch.setattr(extractor, "extract", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(orchestrator_module, "validate_ai_template_definition", lambda **_kwargs: True)

    response = service.extract(
        invoice_path,
        OcrProcessOptions(
            llm_fallback_enabled=True,
            auto_generate_templates=True,
        ),
    )

    created_templates = list(tmp_path.rglob("template_ai*.yml"))

    assert response["status"] == "success"
    assert len(created_templates) == 1
    assert adapter.template_calls == 2
    assert "Known extraction result from this same invoice" in adapter.template_contexts[0]
    assert '"invoice_number": "INV-42"' in adapter.template_contexts[0]
    assert "Previous OCR retry result" in adapter.template_contexts[1]
    assert "missing_required_fields" in adapter.template_contexts[1]
    assert adapter.extraction_kwargs[0]["session_id"] == adapter.template_kwargs[0]["session_id"]
    assert adapter.template_kwargs[0]["session_id"] == adapter.template_kwargs[1]["session_id"]


# ---------------------------------------------------------------------------
# Logs include llm_provider
# ---------------------------------------------------------------------------

def test_orchestrator_logs_include_llm_provider(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("OCR_DEBUG", "true")
    logger_module.configure_logging(force=True)
    capsys.readouterr()

    extractor = OcrExtractor(tmp_path)
    adapter = FakeLlmAdapter(
        extraction_response={
            "fields": {
                "invoice_number": "INV-42",
                "date": "2026-04-03",
                "amount": 123.45,
                "currency_code": "EUR",
            },
            "confidence": {
                "invoice_number": 0.91,
                "date": 0.90,
                "amount": 0.94,
                "currency_code": 0.95,
            },
            "issuer": "Acme B.V.",
            "model": "fake/model",
        }
    )
    service = OcrOrchestrator(extractor, adapter)
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")

    monkeypatch.setattr(
        extractor,
        "extract",
        lambda *_args, **_kwargs: {
            "status": "failed",
            "message": "No matching invoice template found for this document.",
            "fields": {},
            "unmapped_fields": {},
        },
    )

    response = service.extract(
        invoice_path,
        OcrProcessOptions(
            llm_fallback_enabled=True,
            auto_generate_templates=False,
        ),
    )

    assert response["status"] == "success"

    captured_logs = [
        json.loads(line)
        for line in capsys.readouterr().err.splitlines()
        if line.strip()
    ]

    assert "orchestrator.local.completed" in {entry["action"] for entry in captured_logs}
    assert "orchestrator.llm_extract.completed" in {entry["action"] for entry in captured_logs}

    completed_entries = [
        entry for entry in captured_logs
        if entry["action"] == "orchestrator.extract.completed"
    ]
    assert completed_entries
    # Every completed log should carry llm_provider
    for entry in completed_entries:
        assert "llm_provider" in entry.get("context", {})

    assert any(
        entry["action"] == "orchestrator.extract.completed"
        and entry.get("context", {}).get("final_source") == "llm"
        for entry in captured_logs
    )
