import json
from pathlib import Path

from fastapi.testclient import TestClient

from ocr_service import main
from ocr_service import logger as logger_module
from ocr_service.main import ocr_service

# Module-level client — stateless, safe to share across tests in this module.
# Tests that need isolation (e.g. auth env vars) still work because monkeypatch
# resets env vars per-test and the auth check reads os.getenv at request time.
client = TestClient(app)


def test_extract_rejects_missing_token_when_api_key_configured(monkeypatch) -> None:
    monkeypatch.setenv("OCR_API_KEY", "secret-token")

    response = client.post(
        "/extract",
        files={"file": ("invoice.pdf", b"%PDF-1.7", "application/pdf")},
        data={"country_code": "NL"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_extract_rejects_wrong_token_when_api_key_configured(monkeypatch) -> None:
    monkeypatch.setenv("OCR_API_KEY", "secret-token")

    response = client.post(
        "/extract",
        files={"file": ("invoice.pdf", b"%PDF-1.7", "application/pdf")},
        data={"country_code": "NL"},
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert response.status_code == 401


def test_extract_accepts_correct_token(monkeypatch) -> None:
    monkeypatch.setenv("OCR_API_KEY", "secret-token")
    monkeypatch.setattr(
        main.orchestrator,
        "extract",
        lambda *_args, **_kwargs: {
            "status": "success",
            "message": "ok",
            "fields": {},
            "unmapped_fields": {},
        },
    )

    response = client.post(
        "/extract",
        files={"file": ("invoice.pdf", b"%PDF-1.7", "application/pdf")},
        data={"country_code": "NL"},
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 200


def test_extract_allows_all_when_no_api_key_configured(monkeypatch) -> None:
    monkeypatch.delenv("OCR_API_KEY", raising=False)
    monkeypatch.setattr(
        main.orchestrator,
        "extract",
        lambda *_args, **_kwargs: {
            "status": "success",
            "message": "ok",
            "fields": {},
            "unmapped_fields": {},
        },
    )

    response = client.post(
        "/extract",
        files={"file": ("invoice.pdf", b"%PDF-1.7", "application/pdf")},
        data={"country_code": "NL"},
    )

    assert response.status_code == 200


def test_health_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(main.extractor, "template_count", lambda: 3)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "templates": 3,
    }


def test_extract_rejects_unsupported_file_type() -> None:
    response = client.post(
        "/extract",
        files={
            "file": ("invoice.gif", b"gif89a", "image/gif"),
        },
        data={"country_code": "NL"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Unsupported file type. Please upload PDF, JPG, or PNG."


def test_extract_rejects_empty_file() -> None:
    response = client.post(
        "/extract",
        files={
            "file": ("invoice.pdf", b"", "application/pdf"),
        },
        data={"country_code": "NL"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Uploaded file is empty."


def test_playground_page_renders_forms(monkeypatch) -> None:
    monkeypatch.setenv("OCR_OPENROUTER_FALLBACK", "true")
    monkeypatch.setenv("OCR_AUTO_GENERATE_TEMPLATES", "false")

    response = client.get("/")

    assert response.status_code == 200
    assert "OCR Playground" in response.text
    assert "/playground/result" in response.text
    assert "/template-generator" in response.text
    # Notice must show provider-neutral LLM text, not hardcode "OpenRouter"
    assert "LLM fallback" in response.text
    assert "OCR_AUTO_GENERATE_TEMPLATES" in response.text
    assert 'name="openrouter_enabled"' not in response.text
    assert 'name="auto_generate_templates"' not in response.text


def test_extract_delegates_to_extractor(monkeypatch) -> None:
    expected_response = {
        "status": "success",
        "message": "Invoice fields extracted successfully.",
        "fields": {
            "invoice_date": {"value": "2026-04-02", "confidence": 0.98},
        },
        "unmapped_fields": {},
        "line_items": [],
    }

    def fake_extract(input_path: Path, options) -> dict:
        assert input_path.suffix == ".pdf"
        assert options.country_code == "NL"
        assert options.required_fields == (
            "invoice_number",
            "date",
            "amount",
            "currency_code",
        )
        return expected_response

    monkeypatch.setattr(main.orchestrator, "extract", fake_extract)

    response = client.post(
        "/extract",
        files={
            "file": ("invoice.pdf", b"%PDF-1.7", "application/pdf"),
        },
        data={"country_code": "NL"},
    )

    assert response.status_code == 200
    assert response.json() == expected_response


def test_extract_reads_runtime_flags_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OCR_OPENROUTER_FALLBACK", "true")
    monkeypatch.setenv("OCR_AUTO_GENERATE_TEMPLATES", "yes")

    expected_response = {
        "status": "success",
        "message": "Invoice fields extracted successfully.",
        "fields": {
            "invoice_number": {"value": "DEMO-42", "confidence": 0.98},
        },
        "unmapped_fields": {},
        "line_items": [],
    }

    def fake_extract(input_path: Path, options) -> dict:
        assert input_path.suffix == ".pdf"
        # Legacy OCR_OPENROUTER_FALLBACK=true → llm_fallback_enabled is True
        assert options.llm_fallback_enabled is True
        assert options.auto_generate_templates is True
        return expected_response

    monkeypatch.setattr(main.orchestrator, "extract", fake_extract)

    response = client.post(
        "/extract",
        files={
            "file": ("invoice.pdf", b"%PDF-1.7", "application/pdf"),
        },
        data={"country_code": "NL"},
    )

    assert response.status_code == 200
    assert response.json() == expected_response


def test_extract_logs_request_lifecycle(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OCR_DEBUG", "true")
    logger_module.configure_logging(force=True)
    capsys.readouterr()

    expected_response = {
        "status": "success",
        "message": "Invoice fields extracted successfully.",
        "fields": {
            "invoice_number": {"value": "DEMO-42", "confidence": 0.98},
        },
        "unmapped_fields": {},
    }

    monkeypatch.setattr(main.orchestrator, "extract", lambda *_args, **_kwargs: expected_response)

    response = client.post(
        "/extract",
        files={
            "file": ("invoice.pdf", b"%PDF-1.7", "application/pdf"),
        },
        data={"country_code": "NL"},
    )

    assert response.status_code == 200

    captured_logs = [
        json.loads(line)
        for line in capsys.readouterr().err.splitlines()
        if line.strip()
    ]
    logged_actions = {entry["action"] for entry in captured_logs}

    assert {
        "http.extract.received",
        "extract.run.started",
        "extract.run.completed",
        "http.extract.completed",
    }.issubset(logged_actions)


def test_playground_result_renders_json_payload(monkeypatch) -> None:
    expected_response = {
        "status": "success",
        "message": "Invoice fields extracted successfully.",
        "fields": {
            "invoice_number": {"value": "DEMO-42", "confidence": 0.98},
        },
        "unmapped_fields": {},
    }

    monkeypatch.setattr(main, "_extract_from_bytes", lambda *_args, **_kwargs: expected_response)

    response = client.post(
        "/playground/result",
        files={
            "file": ("invoice.pdf", b"%PDF-1.7", "application/pdf"),
        },
        data={"country_code": "NL"},
    )

    assert response.status_code == 200
    assert "&quot;status&quot;: &quot;success&quot;" in response.text
    assert "&quot;invoice_number&quot;" in response.text


def test_template_generator_page_renders_form() -> None:
    response = client.get("/template-generator")

    assert response.status_code == 200
    assert "Starter Template Generator" in response.text
    assert "/template-generator/result" in response.text
    assert "Leave label fields blank for unlabeled receipts or invoices." in response.text


def test_template_generator_result_writes_template(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main.extractor, "writable_template_dir", tmp_path)
    monkeypatch.setattr(
        main.extractor,
        "_extract_ocr_text",
        lambda *_args, **_kwargs: (
            "Acme B.V.\nKvK 12345678\nFactuurnummer: ACME-42\nFactuurdatum: 03-04-2026\n"
            "Totaal incl. btw: EUR 123,45"
        ),
    )

    response = client.post(
        "/template-generator/result",
        files={
            "file": ("invoice.png", b"png-data", "image/png"),
        },
        data={
            "issuer": "Acme B.V.",
            "invoice_number_label": "Factuurnummer",
            "date_label": "Factuurdatum",
            "amount_label": "Totaal incl. btw",
            "country_code": "NL",
            "currency_code": "EUR",
            "currency_label": "",
            "keywords": "KvK 12345678",
        },
    )

    created_template = tmp_path / "nl" / "acme_b_v" / "template.yml"

    assert response.status_code == 200
    assert "Starter Template Result" in response.text
    assert created_template.exists()
    assert "Factuurnummer" in created_template.read_text(encoding="utf-8")

def test_template_generator_result_renders_line_items_preview_table(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main.extractor, "writable_template_dir", tmp_path)
    monkeypatch.setattr(main.extractor, "_extract_ocr_text", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        main.extractor,
        "_load_input_text",
        lambda *_args, **_kwargs: (
            "Factuur\n"
            "Factuurnummer\n"
            "107002\n"
            "Factuurdatum\n"
            "04/08/2024\n"
            "Aantal\n\n"
            "Artikelomschrijving\n\n"
            "HE-Prijs\n\n"
            "Ex. BTW\n\n"
            "BTW\n\n"
            "Bedrag\n\n"
            "1\n"
            "2\n\n"
            "Product A\n"
            "Product B\n\n"
            "€ 10,00\n"
            "€ 5,00\n\n"
            "€ 10,00\n"
            "€ 10,00\n\n"
            "9%\n"
            "9%\n\n"
            "€ 10,90\n"
            "€ 10,90\n\n"
            "Factuurbedrag\n"
            "€ 21,80"
        ),
    )

    response = client.post(
        "/template-generator/result",
        files={
            "file": ("invoice.pdf", b"%PDF-1.7", "application/pdf"),
        },
        data={
            "issuer": "Mercado",
            "invoice_number_label": "Factuurnummer",
            "date_label": "Factuurdatum",
            "amount_label": "Factuurbedrag",
            "country_code": "NL",
            "currency_code": "EUR",
            "currency_label": "",
            "keywords": "",
        },
    )

    assert response.status_code == 200
    assert "Document Preview" in response.text
    assert "Invoice Number" in response.text
    assert "107002" in response.text
    assert "Line Items Preview" in response.text
    assert "Product A" in response.text
    assert "Artikelomschrijving" in response.text


def test_template_generator_result_accepts_blank_labels_for_receipts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main.extractor, "writable_template_dir", tmp_path)
    monkeypatch.setattr(
        main.extractor,
        "_extract_ocr_text",
        lambda *_args, **_kwargs: (
            "ACTION\n"
            "1348 Tilburg is\n"
            "Wagnerplein 113\n"
            "12-08-2024 12:34:53 ‚A 1348102-10743227\n"
            "ARTIKELEN\n"
            "3206092 gootsteenze € 0.99\n"
            "TOTAAL je Se AE 2.70 3.27\n"
        ),
    )

    response = client.post(
        "/template-generator/result",
        files={
            "file": ("receipt.jpeg", b"jpeg-data", "image/jpeg"),
        },
        data={
            "issuer": "Action Receipt",
            "invoice_number_label": "",
            "date_label": "",
            "amount_label": "",
            "country_code": "NL",
            "currency_code": "EUR",
            "currency_label": "",
            "keywords": "",
        },
    )

    created_template = tmp_path / "nl" / "action_receipt" / "template.yml"

    assert response.status_code == 200
    assert created_template.exists()
    assert "Invoice Number" in response.text
    assert "1348102-10743227" in response.text
    assert "12-08-2024" in response.text
    assert "3.27" in response.text
