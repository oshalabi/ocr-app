from pathlib import Path

import pytest

from ocr_service.extractor import OcrExtractor


def test_build_response_marks_searchable_template_match_as_success() -> None:
    extractor = OcrExtractor(Path(__file__).resolve().parent.parent / "templates")

    response = extractor._build_response(
        payload={
            "issuer": "Demo Leverancier B.V.",
            "date": "2026-04-02",
            "invoice_number": "DEMO-2026-0042",
            "amount": "123.45",
            "currency": "EUR",
        },
        source_reader="pdftotext",
        country_code="NL",
    )

    assert response["status"] == "success"
    assert response["fields"]["invoice_date"]["value"] == "2026-04-02"
    assert response["fields"]["invoice_number"]["value"] == "DEMO-2026-0042"
    assert response["fields"]["total_amount"]["value"] == 123.45
    assert response["fields"]["currency_code"]["value"] == "EUR"


def test_build_response_uses_aliases_and_normalizes_currency_symbol() -> None:
    extractor = OcrExtractor(Path(__file__).resolve().parent.parent / "templates")

    response = extractor._build_response(
        payload={
            "invoice_date": "2026-04-02",
            "invoice_no": "DEMO-2026-0042",
            "invoice_total": "€ 1.234,56",
            "currency_symbol": "€",
        },
        source_reader="pdftotext",
        country_code="NL",
    )

    assert response["status"] == "success"
    assert response["fields"]["total_amount"]["value"] == 1234.56
    assert response["fields"]["currency_code"]["value"] == "EUR"


def test_build_response_marks_missing_required_fields_as_partial() -> None:
    extractor = OcrExtractor(Path(__file__).resolve().parent.parent / "templates")

    response = extractor._build_response(
        payload={
            "issuer": "Demo Leverancier B.V.",
            "invoice_number": "DEMO-2026-0042",
        },
        source_reader="tesseract",
        country_code="NL",
    )

    assert response["status"] == "partial"
    assert response["unmapped_fields"]["missing_required_fields"]["value"] == [
        "currency_code",
        "invoice_date",
        "total_amount",
    ]


def test_flatten_templates_copies_nested_templates(tmp_path: Path) -> None:
    extractor = OcrExtractor(Path(__file__).resolve().parent.parent / "templates")

    extractor._flatten_templates(tmp_path)

    flattened_templates = sorted(path.name for path in tmp_path.iterdir())

    assert "templates__nl__demo_leverancier_bv__template.yml" in flattened_templates


def test_extract_returns_searchable_match_without_ocr_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extractor = OcrExtractor(Path(__file__).resolve().parent.parent / "templates")
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.7")

    def fake_run_invoice2data(input_path: Path, input_reader: str) -> dict | None:
        assert input_path == invoice_path
        assert input_reader == "pdftotext"

        return {
            "issuer": "Demo Leverancier B.V.",
            "date": "2026-04-02",
            "invoice_number": "DEMO-2026-0042",
            "amount": "123.45",
            "currency_code": "EUR",
        }

    def fail_if_ocr_runs(*_args, **_kwargs) -> str:
        raise AssertionError("OCR fallback should not run when searchable extraction succeeded.")

    monkeypatch.setattr(extractor, "_run_invoice2data", fake_run_invoice2data)
    monkeypatch.setattr(extractor, "_extract_ocr_text", fail_if_ocr_runs)
    monkeypatch.setattr(extractor, "_load_input_text", lambda *_args, **_kwargs: "")

    response = extractor.extract(invoice_path, country_code="NL")

    assert response["status"] == "success"
    assert response["unmapped_fields"]["source_reader"]["value"] == "pdftotext"


def test_extract_uses_ocr_fallback_for_scanned_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extractor = OcrExtractor(Path(__file__).resolve().parent.parent / "templates")
    invoice_path = tmp_path / "invoice.jpg"
    invoice_path.write_bytes(b"jpeg-data")

    def fake_run_invoice2data(input_path: Path, input_reader: str) -> dict | None:
        assert input_reader == "text"
        assert input_path.suffix == ".txt"

        return {
            "issuer": "Demo Leverancier B.V.",
            "date": "2026-04-02",
            "invoice_number": "DEMO-2026-0042",
            "amount": "123,45",
            "currency_code": "EUR",
        }

    monkeypatch.setattr(extractor, "_run_invoice2data", fake_run_invoice2data)
    monkeypatch.setattr(
        extractor,
        "_extract_ocr_text",
        lambda input_path, language: (
            "Factuurnummer: DEMO-2026-0042\nFactuurdatum: 02-04-2026\nTotaal incl. btw: EUR 123,45"
            if input_path == invoice_path and language == "nld+eng"
            else ""
        ),
    )

    response = extractor.extract(invoice_path, country_code="NL")

    assert response["status"] == "success"
    assert response["fields"]["total_amount"]["value"] == 123.45
    assert response["fields"]["invoice_number"]["confidence"] == 0.78
    assert response["unmapped_fields"]["source_reader"]["value"] == "tesseract"


def test_extract_returns_failed_when_no_template_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extractor = OcrExtractor(Path(__file__).resolve().parent.parent / "templates")
    invoice_path = tmp_path / "invoice.png"
    invoice_path.write_bytes(b"png-data")

    monkeypatch.setattr(extractor, "_run_invoice2data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(extractor, "_extract_ocr_text", lambda *_args, **_kwargs: "")

    response = extractor.extract(invoice_path, country_code="NL")

    assert response["status"] == "failed"
    assert response["fields"] == {}
    assert response["unmapped_fields"]["country_code"]["value"] == "NL"


def test_regex_template_fallback_matches_text_input_when_invoice2data_returns_nothing(tmp_path: Path) -> None:
    template_dir = tmp_path / "templates"
    template_path = template_dir / "nl" / "acme_b_v" / "template.yml"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(
        "\n".join(
            [
                "issuer: 'Acme B.V.'",
                "keywords:",
                "  - 'Acme\\s+B\\.V\\.'",
                "fields:",
                "  invoice_number: 'Factuurnummer\\s*[:#-]?\\s*([A-Z0-9-]+)'",
                "  date: 'Factuurdatum\\s*[:#-]?\\s*([0-9]{2}-[0-9]{2}-[0-9]{4})'",
                "  amount: 'Totaal\\s+incl\\.\\s+btw\\s*[:#-]?\\s*(?:EUR|€)?\\s*([0-9.,]+)'",
                "  currency_code: 'Totaal\\s+incl\\.\\s+btw\\s*[:#-]?\\s*((?:EUR|€))\\s*[0-9.,]+'",
                "options:",
                "  remove_whitespace: true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    input_path = tmp_path / "invoice.txt"
    input_path.write_text(
        "\n".join(
            [
                "Acme B.V.",
                "Factuurnummer: ACME-42",
                "Factuurdatum: 03-04-2026",
                "Totaal incl. btw: EUR 123,45",
                "",
            ]
        ),
        encoding="utf-8",
    )

    extractor = OcrExtractor(template_dir)

    payload = extractor._run_template_regex_fallback(input_path, input_reader="text")

    assert payload == {
        "issuer": "Acme B.V.",
        "invoice_number": "ACME-42",
        "date": "03-04-2026",
        "amount": "123,45",
        "currency_code": "EUR",
    }


def test_score_payload_match_returns_full_priority_tuple() -> None:
    extractor = OcrExtractor(Path(__file__).resolve().parent.parent / "templates")

    score = extractor._score_payload_match(
        {
            "invoice_number": "ACME-42",
            "date": "03-04-2026",
            "amount": "123,45",
            "currency_code": "EUR",
        }
    )

    assert score == (4, 1, 1, 1, 4)


def test_repair_payload_fixes_invalid_invoice_number_and_missing_currency() -> None:
    extractor = OcrExtractor(Path(__file__).resolve().parent.parent / "templates")

    payload = extractor._repair_payload(
        {
            "issuer": "Croco shop",
            "date": "26-06-2015",
            "invoice_number": "datum",
            "amount": "2.405,63",
        },
        (
            "Croco Shop\nFactuurdatum: 26-06-2015 2015384\n"
            "valuta EUR\nSubtotaal €2.405,63"
        ),
    )

    assert payload["invoice_number"] == "2015384"
    assert payload["currency_code"] == "EUR"


def test_repair_payload_replaces_generic_issuer_with_detected_company_name() -> None:
    extractor = OcrExtractor(Path(__file__).resolve().parent.parent / "templates")

    payload = extractor._repair_payload(
        {
            "issuer": "Onbekend",
            "invoice_number": "F2022-0021",
            "date": "05-07-2022",
            "amount": "933,52",
        },
        (
            "MyCompany zen\nBedrijf B.V.\nBTW nummer: NL0123456789B01\n"
            "Factuurnummer: F2022-0021\nFactuurdatum: 05-07-2022\nTotaal incl. BTW € 933,52"
        ),
    )

    assert payload["issuer"] == "Bedrijf B.V."


def test_repair_payload_infers_invoice_date_from_multicolumn_header_row() -> None:
    extractor = OcrExtractor(Path(__file__).resolve().parent.parent / "templates")

    payload = extractor._repair_payload(
        {
            "issuer": "Media Markt",
            "invoice_number": "N022-97947-4999-0-21032026",
            "amount": "1.629,00",
            "currency_code": "EUR",
        },
        (
            "Factuur\n"
            "Factuurnummer N022-97947-4999-0-21032026\n"
            "Factuurdatum Bestelnummer Besteldatum Betaalwijze\n"
            "21.03.2026 305091007 20.03.2026 Online betaling\n"
            "Totaal 1.629,00 EUR"
        ),
    )

    assert payload["date"] == "21.03.2026"


def test_repair_payload_infers_unlabeled_receipt_invoice_number_and_date() -> None:
    extractor = OcrExtractor(Path(__file__).resolve().parent.parent / "templates")

    payload = extractor._repair_payload(
        {
            "issuer": "Action",
            "amount": "3.27",
            "currency_code": "EUR",
        },
        (
            "ACTION\n"
            "1348 Tilburg\n"
            "Wagnerplein 113\n"
            "12-08-2024 12:34:53 1348102- 10743227\n"
            "134810270186527\n"
            "ARTIKELEN\n"
            "TOTAAL 3.27\n"
        ),
    )

    assert payload["invoice_number"] == "1348102-10743227"
    assert payload["date"] == "12-08-2024"



def test_repair_payload_infers_receipt_invoice_number_with_ocr_noise_prefix() -> None:
    extractor = OcrExtractor(Path(__file__).resolve().parent.parent / "templates")

    payload = extractor._repair_payload(
        {
            "issuer": "Action",
            "amount": "2.70",
            "currency_code": "EUR",
        },
        (
            "1348 Tilburg is\n"
            "Wagnerplein 113\n"
            "12-08-2024 12:34:53 ‚A 1348102-10743227\n"
            "TOTAAL 2.70 3.27\n"
        ),
    )

    assert payload["invoice_number"] == "1348102-10743227"
    assert payload["date"] == "12-08-2024"


def test_match_template_definition_treats_literal_ai_values_as_literals() -> None:
    extractor = OcrExtractor(Path(__file__).resolve().parent.parent / "templates")

    payload = extractor._match_template_definition(
        {
            "issuer": "ACTION",
            "keywords": ["ACTION", "Wagnerplein", "ARTIKELEN"],
            "fields": {
                "invoice_number": "1348102-10743227",
                "date": "12-08-2024",
                "amount": "3.27",
                "currency_code": "EUR",
            },
        },
        (
            "ACTION\n"
            "Wagnerplein 113\n"
            "12-08-2024 12:34:53 ‚A 1348102-10743227\n"
            "ARTIKELEN\n"
            "TOTAAL 2.70 3.27\n"
            "134810270186527\n"
        ),
    )

    assert payload == {
        "issuer": "ACTION",
        "invoice_number": "1348102-10743227",
        "date": "12-08-2024",
        "amount": "3.27",
    }


def test_regex_template_fallback_prefers_more_direct_template_matches_over_repaired_fields(tmp_path: Path) -> None:
    template_dir = tmp_path / "templates"
    supplier_directory = template_dir / "nl" / "action"
    supplier_directory.mkdir(parents=True, exist_ok=True)

    (supplier_directory / "template.yml").write_text(
        "\n".join(
            [
                "issuer: 'Action'",
                "keywords:",
                "  - '(?i)ACTION'",
                "  - '(?i)Wagnerplein'",
                "fields:",
                "  amount: '(?i)Totaal[\\s\\S]{0,40}?([0-9]+[.,][0-9]{2})'",
                "  currency_code: '€'",
                "options:",
                "  remove_whitespace: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (supplier_directory / "template_ai.yml").write_text(
        "\n".join(
            [
                "issuer: 'ACTION'",
                "keywords:",
                "  - 'ACTION'",
                "  - 'Wagnerplein'",
                "  - 'ARTIKELEN'",
                "fields:",
                "  invoice_number: '1348102-10743227'",
                "  date: '12-08-2024'",
                "  amount: '3.27'",
                "  currency_code: 'EUR'",
                "options:",
                "  remove_whitespace: true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    input_path = tmp_path / "receipt.txt"
    input_path.write_text(
        "\n".join(
                [
                    "ACTION",
                    "Wagnerplein 113",
                    "12-08-2024 12:34:53 ‚A 1348102-10743227",
                    "ARTIKELEN",
                    "3206092 gootsteenze € 0.99",
                    "TOTAAL 2.70 3.27",
                    "134810270186527",
                    "",
                ]
        ),
        encoding="utf-8",
    )

    extractor = OcrExtractor(template_dir)

    payload = extractor._run_template_regex_fallback(input_path, input_reader="text")

    assert payload == {
        "issuer": "ACTION",
        "invoice_number": "1348102-10743227",
        "date": "12-08-2024",
        "amount": "3.27",
        "currency_code": "EUR",
    }


def test_regex_template_fallback_prefers_the_more_complete_payload(tmp_path: Path) -> None:
    template_dir = tmp_path / "templates"
    supplier_directory = template_dir / "nl" / "croco_shop"
    supplier_directory.mkdir(parents=True, exist_ok=True)

    (supplier_directory / "template.yml").write_text(
        "\n".join(
            [
                "issuer: 'Croco shop'",
                "keywords:",
                "  - '(?i)Croco\\s+shop'",
                "  - '(?i)NL10093674500'",
                "fields:",
                "  invoice_number: '(?i)Factuur\\s*[:#-]?\\s*((?=[A-Z0-9/._-]*\\d)[A-Z0-9][A-Z0-9/._-]+)'",
                "  date: '(?i)Factuurdatum\\s*[:#-]?\\s*([0-9]{2}-[0-9]{2}-[0-9]{4})'",
                "  amount: '(?i)Subtotaal\\s*(?:EUR|€)?\\s*([0-9.,]+)'",
                "options:",
                "  remove_whitespace: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (supplier_directory / "template_1.yml").write_text(
        "\n".join(
            [
                "issuer: 'Croco shop'",
                "keywords:",
                "  - '(?i)Croco\\s+shop'",
                "  - '(?i)NL10093674500'",
                "fields:",
                "  invoice_number: '(?i)(?:Factuurdatum|Invoice\\s*date)\\s*[:#-]?\\s*[0-9]{2}[./-][0-9]{2}[./-][0-9]{4}\\s+((?=[A-Z0-9/._-]*\\d)[A-Z0-9][A-Z0-9/._-]+)'",
                "  date: '(?i)Factuurdatum\\s*[:#-]?\\s*([0-9]{2}-[0-9]{2}-[0-9]{4})'",
                "  amount: '(?i)Subtotaal\\s*(?:EUR|€)?\\s*([0-9.,]+)'",
                "  currency_code: '(?i)Valuta\\s*[:#-]?\\s*((?:EUR|€))'",
                "options:",
                "  remove_whitespace: true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    input_path = tmp_path / "invoice.txt"
    input_path.write_text(
        "\n".join(
            [
                "Croco Shop",
                "ke 03887323 ~ BTW NL10093674500%",
                "Factuurdatum: 26-06-2015 2015384",
                "valuta EUR",
                "Subtotaal €2.405,63",
                "",
            ]
        ),
        encoding="utf-8",
    )

    extractor = OcrExtractor(template_dir)

    payload = extractor._run_template_regex_fallback(input_path, input_reader="text")

    assert payload == {
        "issuer": "Croco shop",
        "invoice_number": "2015384",
        "date": "26-06-2015",
        "amount": "2.405,63",
        "currency_code": "EUR",
    }
