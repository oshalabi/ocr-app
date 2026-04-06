from pathlib import Path

import yaml

from ocr_service.extractor import OcrExtractor
from ocr_service.template_generator import (
    DocumentPreviewRow,
    GeneratedTemplateDefinition,
    TemplateSpec,
    build_document_preview_rows,
    build_line_item_preview_rows,
    collect_text_sources,
    default_ai_template_path,
    default_template_path,
    generate_starter_template_from_sample,
    suggest_keywords,
    validate_ai_template_definition,
)


def test_suggest_keywords_prefers_issuer_and_trade_identifiers() -> None:
    keywords = suggest_keywords(
        "Acme B.V.",
        "Acme B.V.\nKvK 12345678\nBTW NL123456789B01\nFactuurnummer: ACME-42",
    )

    assert keywords[0] == r"(?i)Acme\s+B\.V\."
    assert r"(?i)KvK\s*(?:nr)?[:.]?\s*12345678" in keywords
    assert r"(?i)NL123456789B01" in keywords


def test_default_template_path_uses_country_and_slug(tmp_path: Path) -> None:
    assert default_template_path(tmp_path, "NL", "Acme B.V.") == tmp_path / "nl" / "acme_b_v" / "template.yml"


def test_default_template_path_uses_next_available_name(tmp_path: Path) -> None:
    supplier_directory = tmp_path / "nl" / "acme_b_v"
    supplier_directory.mkdir(parents=True, exist_ok=True)
    (supplier_directory / "template.yml").write_text("", encoding="utf-8")
    (supplier_directory / "template_1.yml").write_text("", encoding="utf-8")

    assert default_template_path(tmp_path, "NL", "Acme B.V.") == tmp_path / "nl" / "acme_b_v" / "template_2.yml"


def test_default_ai_template_path_uses_next_available_name(tmp_path: Path) -> None:
    supplier_directory = tmp_path / "nl" / "acme_b_v"
    supplier_directory.mkdir(parents=True, exist_ok=True)
    (supplier_directory / "template_ai.yml").write_text("", encoding="utf-8")

    assert default_ai_template_path(tmp_path, "NL", "Acme B.V.") == tmp_path / "nl" / "acme_b_v" / "template_ai_1.yml"


def test_generate_starter_template_from_sample_writes_template(monkeypatch, tmp_path: Path) -> None:
    extractor = OcrExtractor(tmp_path / "templates")
    sample_path = tmp_path / "invoice.png"
    sample_path.write_bytes(b"png-data")

    monkeypatch.setattr(
        extractor,
        "_extract_ocr_text",
        lambda *_args, **_kwargs: (
            "Acme B.V.\nKvK 12345678\nFactuurnummer: ACME-42\nFactuurdatum: 03-04-2026\n"
            "Totaal incl. btw: EUR 123,45"
        ),
    )

    result = generate_starter_template_from_sample(
        sample_path=sample_path,
        template_dir=tmp_path / "templates",
        extractor=extractor,
        spec=TemplateSpec(
            issuer="Acme B.V.",
            invoice_number_label="Factuurnummer",
            date_label="Factuurdatum",
            amount_label="Totaal incl. btw",
            currency_code="EUR",
            country_code="NL",
        ),
    )

    assert result.output_path.exists()
    assert result.output_path.name == "template.yml"
    assert "invoice_number" in result.content
    assert r"KvK\s*(?:nr)?[:.]?\s*12345678" in result.content
    assert r"\s*" in result.content
    assert r"\\s*" not in result.content
    assert result.missing_labels == ()


def test_generate_starter_template_from_sample_does_not_overwrite_existing_template(
    monkeypatch,
    tmp_path: Path,
) -> None:
    template_dir = tmp_path / "templates"
    existing_template = template_dir / "nl" / "acme_b_v" / "template.yml"
    existing_template.parent.mkdir(parents=True, exist_ok=True)
    existing_template.write_text("issuer: 'Existing Template'\n", encoding="utf-8")

    extractor = OcrExtractor(template_dir)
    sample_path = tmp_path / "invoice.png"
    sample_path.write_bytes(b"png-data")

    monkeypatch.setattr(
        extractor,
        "_extract_ocr_text",
        lambda *_args, **_kwargs: (
            "Acme B.V.\nKvK 12345678\nFactuurnummer: ACME-42\nFactuurdatum: 03-04-2026\n"
            "Totaal incl. btw: EUR 123,45"
        ),
    )

    result = generate_starter_template_from_sample(
        sample_path=sample_path,
        template_dir=template_dir,
        extractor=extractor,
        spec=TemplateSpec(
            issuer="Acme B.V.",
            invoice_number_label="Factuurnummer",
            date_label="Factuurdatum",
            amount_label="Totaal incl. btw",
            currency_code="EUR",
            country_code="NL",
        ),
    )

    assert existing_template.read_text(encoding="utf-8") == "issuer: 'Existing Template'\n"
    assert result.output_path == template_dir / "nl" / "acme_b_v" / "template_1.yml"


def test_generate_starter_template_from_sample_resolves_generic_issuer_and_dedupes_keywords(
    monkeypatch,
    tmp_path: Path,
) -> None:
    extractor = OcrExtractor(tmp_path / "templates")
    sample_path = tmp_path / "invoice.pdf"
    sample_path.write_bytes(b"%PDF-1.7")

    monkeypatch.setattr(
        extractor,
        "_extract_ocr_text",
        lambda *_args, **_kwargs: (
            "mmm\nMyCompany zen\nBedrijf B.V.\nBTW nummer: NL0123456789B01\n"
            "Factu ur Factuurnummer: F2022-0021\nFactuurdatum: 05-07-2022\n"
            "Totaal incl. BTW € 933,52"
        ),
    )
    monkeypatch.setattr(
        extractor,
        "_load_input_text",
        lambda *_args, **_kwargs: (
            "Bedrijf B.V.\nBTW nummer: NL0123456789B01\n"
            "Factuurnummer:\nF2022-0021\nFactuurdatum:\n05-07-2022\n"
            "Totaal incl. BTW\n€\n933,52"
        ),
    )

    result = generate_starter_template_from_sample(
        sample_path=sample_path,
        template_dir=tmp_path / "templates",
        extractor=extractor,
        spec=TemplateSpec(
            issuer="Onbekend",
            invoice_number_label="Factuurnummer",
            date_label="Factuurdatum",
            amount_label="Totaal incl. BTW",
            currency_code="EUR",
            country_code="NL",
        ),
    )

    assert result.output_path == tmp_path / "templates" / "nl" / "bedrijf_b_v" / "template.yml"
    assert "issuer: 'Bedrijf B.V.'" in result.content
    assert "Onbekend" not in result.content
    assert result.content.count("NL0123456789B01") == 1
    assert "currency_code" in result.content
    assert "(?:EUR|€)" in result.content


def test_generate_starter_template_from_sample_prefers_validated_fallback_patterns(
    monkeypatch,
    tmp_path: Path,
) -> None:
    extractor = OcrExtractor(tmp_path / "templates")
    sample_path = tmp_path / "invoice.png"
    sample_path.write_bytes(b"png-data")

    monkeypatch.setattr(
        extractor,
        "_extract_ocr_text",
        lambda *_args, **_kwargs: (
            "Croco Shop\nke 03887323 ~ BTW NL10093674500%\n"
            "Factuurdatum: 26-06-2015 2015384\nvaluta EUR\nSubtotaal €2.405,63"
        ),
    )

    result = generate_starter_template_from_sample(
        sample_path=sample_path,
        template_dir=tmp_path / "templates",
        extractor=extractor,
        spec=TemplateSpec(
            issuer="Croco shop",
            invoice_number_label="Factuur",
            date_label="Factuurdatum",
            amount_label="Totaal",
            currency_code="EUR",
            currency_label="Valuta",
            country_code="NL",
        ),
    )

    assert "invoice_number: '(?i)(?:Factuurdatum|Invoice\\s*date)" in result.content
    assert "amount: '(?i)(?m)^.*?(?<!\\w)Subtotaal(?!\\w)" in result.content
    assert "currency_code: '(?i)(?<!\\w)Valuta(?!\\w)" in result.content


def test_generate_starter_template_from_sample_supports_multicolumn_header_date_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    extractor = OcrExtractor(tmp_path / "templates")
    sample_path = tmp_path / "invoice.pdf"
    sample_path.write_bytes(b"%PDF-1.7")

    monkeypatch.setattr(
        extractor,
        "_extract_ocr_text",
        lambda *_args, **_kwargs: (
            "Media Markt Arnhem Velperplein 13 6811 AG Arnhem\n"
            "Factuur\n"
            "Factuurnummer N022-97947-4999-0-21032026\n"
            "Factuurdatum Bestelnummer Besteldatum Betaalwijze\n"
            "21.03.2026 305091007 20.03.2026 Online betaling\n"
            "Totaal 1.629,00 EUR"
        ),
    )
    monkeypatch.setattr(
        extractor,
        "_load_input_text",
        lambda *_args, **_kwargs: "",
    )

    result = generate_starter_template_from_sample(
        sample_path=sample_path,
        template_dir=tmp_path / "templates",
        extractor=extractor,
        spec=TemplateSpec(
            issuer="Media Markt",
            invoice_number_label="Factuurnummer",
            date_label="Factuurdatum",
            amount_label="Totaal",
            currency_code="EUR",
            country_code="NL",
        ),
    )

    assert "date:" in result.content
    assert r"Factuurdatum(?!\w)[\s\S]{0,160}?" in result.content
    assert "21\\.03\\.2026\\s+305091007\\s+20\\.03\\.2026\\s+Online\\s+betaling" not in result.content


def test_generate_starter_template_from_sample_supports_blank_labels_for_receipts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    extractor = OcrExtractor(tmp_path / "templates")
    sample_path = tmp_path / "receipt.jpeg"
    sample_path.write_bytes(b"jpeg-data")

    monkeypatch.setattr(
        extractor,
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

    result = generate_starter_template_from_sample(
        sample_path=sample_path,
        template_dir=tmp_path / "templates",
        extractor=extractor,
        spec=TemplateSpec(
            issuer="Action Receipt",
            invoice_number_label="",
            date_label="",
            amount_label="",
            currency_code="EUR",
            country_code="NL",
        ),
    )

    generated_definition = yaml.safe_load(result.content)
    payload = extractor._match_template_definition(generated_definition, result.preview_text)

    assert payload is not None

    repaired_payload = extractor._repair_payload(payload, result.preview_text)

    assert repaired_payload["invoice_number"] == "1348102-10743227"
    assert repaired_payload["date"] == "12-08-2024"
    assert repaired_payload["amount"] == "3.27"


def test_validate_ai_template_definition_rejects_invoice_specific_keywords(monkeypatch, tmp_path: Path) -> None:
    extractor = OcrExtractor(tmp_path / "templates")
    sample_path = tmp_path / "invoice.png"
    sample_path.write_bytes(b"png-data")

    monkeypatch.setattr(
        extractor,
        "_extract_ocr_text",
        lambda *_args, **_kwargs: (
            "Acme B.V.\nFactuurnummer: INV-42\nFactuurdatum: 03-04-2026\nTotaal incl. btw: EUR 123,45"
        ),
    )

    assert validate_ai_template_definition(
        definition=GeneratedTemplateDefinition(
            issuer="Acme B.V.",
            keywords=("INV-42",),
            fields={
                "invoice_number": r"Factuurnummer\s*[:#-]?\s*([A-Z0-9-]+)",
                "date": r"Factuurdatum\s*[:#-]?\s*([0-9]{2}-[0-9]{2}-[0-9]{4})",
                "amount": r"Totaal incl\. btw\s*[:#-]?\s*(?:EUR|€)?\s*([0-9.,]+)",
                "currency_code": r"Totaal incl\. btw\s*[:#-]?\s*((?:EUR|€))",
            },
        ),
        sample_path=sample_path,
        extractor=extractor,
        country_code="NL",
        required_fields=("invoice_number", "date", "amount", "currency_code"),
    ).is_valid is False


def test_validate_ai_template_definition_allows_supplier_vat_kvk_and_domain_keywords(
    monkeypatch,
    tmp_path: Path,
) -> None:
    extractor = OcrExtractor(tmp_path / "templates")
    sample_path = tmp_path / "invoice.pdf"
    sample_path.write_bytes(b"%PDF-1.7")

    monkeypatch.setattr(
        extractor,
        "_extract_ocr_text",
        lambda *_args, **_kwargs: (
            "Ozer Logistics BV\nKvK 24437828\nBTW NL822257750B01\nwww.ozerlogistics.nl\n"
            "Invoice number 202502224\nInvoice date 07-03-2025\nTotal EUR 3345.21"
        ),
    )
    monkeypatch.setattr(
        extractor,
        "_load_input_text",
        lambda *_args, **_kwargs: "",
    )

    assert validate_ai_template_definition(
        definition=GeneratedTemplateDefinition(
            issuer="Ozer Logistics BV",
            keywords=(
                r"(?i)Ozer\s+Logistics\s+BV",
                r"(?i)KvK\s*(?:nr)?[:.]?\s*24437828",
                r"(?i)ozerlogistics\.nl",
            ),
            fields={
                "invoice_number": r"(?i)Invoice\s+number\s+([A-Z0-9-]+)",
                "date": r"(?i)Invoice\s+date\s+([0-9]{2}-[0-9]{2}-[0-9]{4})",
                "amount": r"(?i)Total\s+EUR\s+([0-9.]+)",
                "currency_code": r"(?i)Total\s+(EUR)",
            },
        ),
        sample_path=sample_path,
        extractor=extractor,
        country_code="NL",
        required_fields=("invoice_number", "date", "amount", "currency_code"),
    ).is_valid is True


def test_validate_ai_template_definition_prunes_non_matching_keywords(monkeypatch, tmp_path: Path) -> None:
    extractor = OcrExtractor(tmp_path / "templates")
    sample_path = tmp_path / "invoice.png"
    sample_path.write_bytes(b"png-data")

    monkeypatch.setattr(
        extractor,
        "_extract_ocr_text",
        lambda *_args, **_kwargs: (
            "HEMA\nDatum 31-12-2024\nBon 123456\nTotaal: 12,34 EUR"
        ),
    )

    validation_result = validate_ai_template_definition(
        definition=GeneratedTemplateDefinition(
            issuer="HEMA",
            keywords=("HEMA", "hema.nl", "Dennekamp"),
            fields={
                "invoice_number": r"Bon\s+(\d+)",
                "date": r"Datum\s+(\d{2}-\d{2}-\d{4})",
                "amount": r"Totaal:\s*([\d,]+)\s+EUR",
                "currency_code": r"Totaal:\s*[\d,]+\s+(\w+)",
            },
        ),
        sample_path=sample_path,
        extractor=extractor,
        country_code="NL",
        required_fields=("invoice_number", "date", "amount", "currency_code"),
    )

    assert validation_result.is_valid is True
    assert "pruned_non_matching_keywords" in validation_result.keyword_adjustments

def test_collect_text_sources_prefers_more_complete_pdf_text_over_partial_ocr(
    monkeypatch,
    tmp_path: Path,
) -> None:
    extractor = OcrExtractor(tmp_path / "templates")
    sample_path = tmp_path / "invoice.pdf"
    sample_path.write_bytes(b"%PDF-1.7")

    monkeypatch.setattr(
        extractor,
        "_extract_ocr_text",
        lambda *_args, **_kwargs: (
            "MERCADO\n"
            "1 Product A € 13,49 € 13,49 9% €14,70\n"
            "2 Product B € 5,95 € 11,90 9% €12,97"
        ),
    )
    monkeypatch.setattr(
        extractor,
        "_load_input_text",
        lambda *_args, **_kwargs: (
            "Factuur\n"
            "Creta Grieks Utrecht\n"
            "Factuurdatum\n"
            "04/08/2024\n"
            "Factuurnummer\n"
            "107002\n"
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
            "€ 562,76"
        ),
    )

    text_sources = collect_text_sources(sample_path, extractor, "NL")

    assert text_sources[0].startswith("Factuur\nCreta Grieks Utrecht")


def test_generate_starter_template_from_sample_uses_best_text_source_for_preview(
    monkeypatch,
    tmp_path: Path,
) -> None:
    extractor = OcrExtractor(tmp_path / "templates")
    sample_path = tmp_path / "invoice.pdf"
    sample_path.write_bytes(b"%PDF-1.7")

    monkeypatch.setattr(
        extractor,
        "_extract_ocr_text",
        lambda *_args, **_kwargs: (
            "MERCADO\n"
            "1 Product A € 13,49 € 13,49 9% €14,70\n"
            "2 Product B € 5,95 € 11,90 9% €12,97"
        ),
    )
    monkeypatch.setattr(
        extractor,
        "_load_input_text",
        lambda *_args, **_kwargs: (
            "Factuur\n"
            "Creta Grieks Utrecht\n"
            "Factuurdatum\n"
            "04/08/2024\n"
            "Factuurnummer\n"
            "107002\n"
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
            "€ 562,76"
        ),
    )

    result = generate_starter_template_from_sample(
        sample_path=sample_path,
        template_dir=tmp_path / "templates",
        extractor=extractor,
        spec=TemplateSpec(
            issuer="Mercado",
            invoice_number_label="Factuurnummer",
            date_label="Factuurdatum",
            amount_label="Factuurbedrag",
            currency_code="EUR",
            country_code="NL",
        ),
    )

    assert "Factuurnummer" in result.preview_text
    assert "107002" in result.preview_text
    assert result.document_rows[1] == DocumentPreviewRow("Invoice Number", "107002", "ok")
    assert result.document_rows[2] == DocumentPreviewRow("Invoice Date", "04/08/2024", "ok")
    assert len(result.line_item_rows) == 2
    assert result.line_item_rows[0].values.get("artikelomschrijving") == "Product A"

def test_build_line_item_preview_rows_parses_columnar_pdf_text() -> None:
    rows = build_line_item_preview_rows(
        "Factuur\n"
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
        "Betaald PIN"
    )

    assert len(rows) == 2
    # Dynamic headers — derived from whatever column names the invoice has
    assert "artikelomschrijving" in rows[0].headers
    assert "aantal" in rows[0].headers
    assert "bedrag" in rows[0].headers
    assert rows[0].values["artikelomschrijving"] == "Product A"
    assert rows[1].values["aantal"] == "2"
    assert rows[1].values["bedrag"] == "€ 10,90"

def test_build_document_preview_rows_reports_extracted_values_and_issues(tmp_path: Path) -> None:
    extractor = OcrExtractor(tmp_path / "templates")
    definition = GeneratedTemplateDefinition(
        issuer="Mercado",
        keywords=(),
        fields={
            "invoice_number": r"(?i)Factuurnummer\s*([A-Z0-9-]+)",
            "date": r"(?i)Factuurdatum\s*([0-9]{2}/[0-9]{2}/[0-9]{4})",
            "amount": r"(?i)Factuurbedrag\s*(?:EUR|€)?\s*([0-9.,]+)",
            "currency_code": r"(?i)((?:EUR|€))",
        },
    )

    rows = build_document_preview_rows(
        preview_text=(
            "Factuur\n"
            "Factuurnummer 107002\n"
            "Factuurdatum 04/08/2024\n"
            "Factuurbedrag € 562,76\n"
            "Aantal Artikelomschrijving HE-Prijs Ex. BTW BTW Bedrag"
        ),
        definition=definition,
        extractor=extractor,
        country_code="NL",
        missing_labels=("currency_label",),
        line_item_rows=(),
    )

    assert rows[0] == DocumentPreviewRow("Issuer", "Mercado", "ok")
    assert rows[1] == DocumentPreviewRow("Invoice Number", "107002", "ok")
    assert rows[2] == DocumentPreviewRow("Invoice Date", "04/08/2024", "ok")
    assert rows[3] == DocumentPreviewRow("Total Amount", "562.76", "ok")
    assert rows[4] == DocumentPreviewRow("Currency", "EUR", "ok")
    assert rows[5] == DocumentPreviewRow("Line Items", "0", "warning")
    assert rows[6].label == "Issues"
    assert "missing labels: currency_label" in rows[6].value
    assert "line item table not reconstructed" in rows[6].value
    assert rows[6].status == "warning"
