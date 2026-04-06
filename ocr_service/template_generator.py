from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocr_service.extractor import OcrExtractor
from ocr_service.fields import DEFAULT_REQUIRED_FIELDS
from ocr_service.logger import get_logger, log_event

GENERIC_ISSUER_VALUES = {
    "",
    "n/a",
    "na",
    "onbekend",
    "supplier",
    "unknown",
    "vendor",
}
DATE_VALUE_PATTERN = r"([0-9]{2}[./-][0-9]{2}[./-][0-9]{4})"
INVOICE_NUMBER_VALUE_PATTERN = r"((?=[A-Z0-9/._-]*\d)[A-Z0-9][A-Z0-9/._-]+)"
AMOUNT_VALUE_PATTERN = r"([0-9]{1,3}(?:[.,][0-9]{3})*[.,][0-9]{2}|[0-9]+[.,][0-9]{2})"
logger = get_logger("ocr_service.template_generator")


@dataclass(frozen=True)
class TemplateSpec:
    issuer: str
    invoice_number_label: str
    date_label: str
    amount_label: str
    country_code: str = "NL"
    currency_code: str = "EUR"
    currency_label: str | None = None
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class LineItemPreviewRow:
    """A single line item row with dynamic columns.

    ``headers`` preserves the detected column names in order.
    ``values`` maps each header to its cell value.
    """
    headers: tuple[str, ...]
    values: dict[str, str]


@dataclass(frozen=True)
class DocumentPreviewRow:
    label: str
    value: str
    status: str


@dataclass(frozen=True)
class TemplateGenerationResult:
    output_path: Path
    content: str
    preview_text: str
    document_rows: tuple[DocumentPreviewRow, ...]
    line_item_rows: tuple[LineItemPreviewRow, ...]
    keywords: tuple[str, ...]
    missing_labels: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedTemplateDefinition:
    issuer: str
    keywords: tuple[str, ...]
    fields: dict[str, str]


@dataclass(frozen=True)
class TemplateValidationResult:
    is_valid: bool
    reason: str
    missing_required_fields: tuple[str, ...] = ()
    matched_required_fields: tuple[str, ...] = ()
    repaired_fields: tuple[str, ...] = ()
    keyword_adjustments: tuple[str, ...] = ()
    effective_keywords: tuple[str, ...] = ()
    text_preview: str | None = None

    def to_prompt_context(self) -> str:
        return (
            f"reason={self.reason}\n"
            f"matched_required_fields={', '.join(self.matched_required_fields) or 'none'}\n"
            f"missing_required_fields={', '.join(self.missing_required_fields) or 'none'}\n"
            f"repaired_fields={', '.join(self.repaired_fields) or 'none'}\n"
            f"keyword_adjustments={', '.join(self.keyword_adjustments) or 'none'}\n"
            f"text_preview=\n{self.text_preview or ''}"
        )


def generate_starter_template_from_sample(
    sample_path: Path,
    template_dir: Path,
    extractor: OcrExtractor,
    spec: TemplateSpec,
    output_path: Path | None = None,
) -> TemplateGenerationResult:
    log_event(
        logger,
        logging.INFO,
        "template_generator.started",
        sample_name=sample_path.name,
        issuer=spec.issuer,
        country_code=spec.country_code,
    )

    if sample_path.suffix.lower() not in extractor.SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Please upload PDF, JPG, or PNG.")

    text_sources = collect_text_sources(sample_path, extractor, spec.country_code)
    preview_text = text_sources[0] if text_sources else ""
    line_item_rows = build_line_item_preview_rows(preview_text, extractor=extractor)
    resolved_issuer = resolve_issuer(spec.issuer, text_sources)
    missing_labels = tuple(
        label_name
        for label_name, label_value in (
            ("invoice_number_label", spec.invoice_number_label),
            ("date_label", spec.date_label),
            ("amount_label", spec.amount_label),
            ("currency_label", spec.currency_label or ""),
        )
        if label_value and not any(contains_phrase(text, label_value) for text in text_sources)
    )
    definition = build_validated_template_definition(
        spec=spec,
        resolved_issuer=resolved_issuer,
        text_sources=text_sources,
        extractor=extractor,
    )
    document_rows = build_document_preview_rows(
        preview_text=preview_text,
        definition=definition,
        extractor=extractor,
        country_code=spec.country_code,
        missing_labels=missing_labels,
        line_item_rows=line_item_rows,
    )
    content = render_template_content(definition)
    destination = output_path or default_template_path(template_dir, spec.country_code, resolved_issuer)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")

    log_event(
        logger,
        logging.INFO,
        "template_generator.completed",
        sample_name=sample_path.name,
        output_path=destination,
        missing_labels=missing_labels,
        keyword_count=len(definition.keywords),
    )

    return TemplateGenerationResult(
        output_path=destination,
        content=content,
        preview_text=preview_text,
        document_rows=document_rows,
        line_item_rows=line_item_rows,
        keywords=definition.keywords,
        missing_labels=missing_labels,
    )


def collect_text_sources(sample_path: Path, extractor: OcrExtractor, country_code: str) -> list[str]:
    text_sources: list[str] = []
    ocr_text = extractor._extract_ocr_text(
        sample_path,
        extractor._language_for_country(country_code.upper()),
    )

    if ocr_text.strip():
        text_sources.append(ocr_text)

    if sample_path.suffix.lower() == ".pdf":
        pdf_text = extractor._load_input_text(sample_path, "pdftotext")

        if pdf_text.strip():
            text_sources.append(pdf_text)

    unique_text_sources: list[str] = []
    seen_text_sources: set[str] = set()

    for text_source in text_sources:
        normalized_text_source = normalize_space(text_source)

        if not normalized_text_source or normalized_text_source in seen_text_sources:
            continue

        seen_text_sources.add(normalized_text_source)
        unique_text_sources.append(text_source)

    ranked_text_sources = sorted(
        unique_text_sources,
        key=score_text_source,
        reverse=True,
    )

    log_event(
        logger,
        logging.DEBUG,
        "template_generator.text_sources.collected",
        sample_name=sample_path.name,
        source_count=len(ranked_text_sources),
        source_lengths=[len(text_source) for text_source in ranked_text_sources],
    )

    return ranked_text_sources


def build_validated_template_definition(
    spec: TemplateSpec,
    resolved_issuer: str,
    text_sources: list[str],
    extractor: OcrExtractor,
    required_fields: tuple[str, ...] | None = None,
) -> GeneratedTemplateDefinition:
    required_fields = required_fields or DEFAULT_REQUIRED_FIELDS
    fields = {
        "invoice_number": choose_best_pattern(
            build_invoice_number_patterns(spec),
            text_sources,
        ),
        "date": choose_best_pattern(
            build_date_patterns(spec),
            text_sources,
        ),
        "amount": choose_best_pattern(
            build_amount_patterns(spec),
            text_sources,
        ),
        "currency_code": choose_best_pattern(
            build_currency_patterns(spec),
            text_sources,
        ),
    }
    keyword_candidates = (
        dedupe_patterns(list(spec.keywords))
        if spec.keywords
        else build_keyword_candidates(resolved_issuer, text_sources)
    )
    keywords = choose_valid_keywords(
        keyword_candidates,
        fields,
        resolved_issuer,
        text_sources,
        extractor,
        required_fields,
    )

    if not validate_template_definition(
        resolved_issuer,
        keywords,
        fields,
        text_sources,
        extractor,
        required_fields,
    ) and validate_template_definition(
        resolved_issuer,
        (),
        fields,
        text_sources,
        extractor,
        required_fields,
    ):
        keywords = ()

    return GeneratedTemplateDefinition(
        issuer=resolved_issuer,
        keywords=keywords,
        fields=fields,
    )


def default_template_path(template_dir: Path, country_code: str, issuer: str) -> Path:
    supplier_directory = template_dir / country_code.lower() / slugify(issuer)
    supplier_directory.mkdir(parents=True, exist_ok=True)

    index = 0

    while True:
        file_name = "template.yml" if index == 0 else f"template_{index}.yml"
        candidate = supplier_directory / file_name

        if not candidate.exists():
            return candidate

        index += 1


def default_ai_template_path(template_dir: Path, country_code: str, issuer: str) -> Path:
    supplier_directory = template_dir / country_code.lower() / slugify(issuer)
    supplier_directory.mkdir(parents=True, exist_ok=True)

    index = 0

    while True:
        file_name = "template_ai.yml" if index == 0 else f"template_ai_{index}.yml"
        candidate = supplier_directory / file_name

        if not candidate.exists():
            return candidate

        index += 1


def render_template_content(definition: GeneratedTemplateDefinition) -> str:
    lines = [
        f"issuer: {yaml_quote(definition.issuer)}",
        "keywords:",
        *[f"  - {yaml_quote(keyword)}" for keyword in definition.keywords],
        "fields:",
        *[
            f"  {field_name}: {yaml_quote(pattern)}"
            for field_name, pattern in definition.fields.items()
        ],
        "options:",
        "  date_formats:",
        "    - '%d-%m-%Y'",
        "    - '%d/%m/%Y'",
        "    - '%Y-%m-%d'",
        "  remove_whitespace: true",
    ]

    return "\n".join(lines) + "\n"


def choose_valid_keywords(
    candidate_keywords: tuple[str, ...],
    fields: dict[str, str],
    issuer: str,
    text_sources: list[str],
    extractor: OcrExtractor,
    required_fields: tuple[str, ...],
) -> tuple[str, ...]:
    unique_keywords = dedupe_patterns(list(candidate_keywords))

    if not unique_keywords:
        return ()

    chosen_keywords: list[str] = []

    for keyword in unique_keywords:
        if not any(re.search(keyword, text, re.MULTILINE | re.IGNORECASE) for text in text_sources):
            continue

        trial_keywords = tuple([*chosen_keywords, keyword])

        if validate_template_definition(issuer, trial_keywords, fields, text_sources, extractor, required_fields):
            chosen_keywords.append(keyword)

    if chosen_keywords:
        return tuple(chosen_keywords[:3])

    if validate_template_definition(issuer, (), fields, text_sources, extractor, required_fields):
        return ()

    return unique_keywords[:1]


def validate_template_definition(
    issuer: str,
    keywords: tuple[str, ...],
    fields: dict[str, str],
    text_sources: list[str],
    extractor: OcrExtractor,
    required_fields: tuple[str, ...] | None = None,
) -> bool:
    required_fields = required_fields or DEFAULT_REQUIRED_FIELDS
    template_definition: dict[str, Any] = {
        "issuer": issuer,
        "keywords": list(keywords),
        "fields": fields,
    }

    for text in text_sources:
        payload = extractor._match_template_definition(template_definition, text)

        if not payload:
            continue

        payload = extractor._repair_payload(payload, text)

        if set(required_fields).issubset(payload.keys()):
            return True

    return False


def validate_ai_template_definition(
    definition: GeneratedTemplateDefinition,
    sample_path: Path,
    extractor: OcrExtractor,
    country_code: str,
    required_fields: tuple[str, ...],
    extracted_values: dict[str, Any] | None = None,
) -> TemplateValidationResult:
    text_sources = collect_text_sources(sample_path, extractor, country_code)
    validation_result = evaluate_ai_template_definition(
        definition=definition,
        text_sources=text_sources,
        extractor=extractor,
        required_fields=required_fields,
        extracted_values=extracted_values,
    )

    if not validation_result.is_valid:
        log_event(
            logger,
            logging.WARNING,
            "template_generator.ai_validation.failed",
            sample_name=sample_path.name,
            issuer=definition.issuer,
            reason=validation_result.reason,
            missing_required_fields=validation_result.missing_required_fields,
            matched_required_fields=validation_result.matched_required_fields,
            repaired_fields=validation_result.repaired_fields,
            keyword_adjustments=validation_result.keyword_adjustments,
        )
        return validation_result

    for keyword in (validation_result.effective_keywords or definition.keywords):
        if keyword_looks_invoice_specific(keyword, text_sources):
            log_event(
                logger,
                logging.WARNING,
                "template_generator.ai_validation.failed",
                sample_name=sample_path.name,
                issuer=definition.issuer,
                reason="invoice_specific_keyword",
                keyword=keyword,
            )
            return TemplateValidationResult(
                is_valid=False,
                reason="invoice_specific_keyword",
                keyword_adjustments=validation_result.keyword_adjustments,
                repaired_fields=validation_result.repaired_fields,
                matched_required_fields=validation_result.matched_required_fields,
                missing_required_fields=validation_result.missing_required_fields,
                effective_keywords=validation_result.effective_keywords,
                text_preview=validation_result.text_preview,
            )

    log_event(
        logger,
        logging.INFO,
        "template_generator.ai_validation.completed",
        sample_name=sample_path.name,
        issuer=definition.issuer,
        keyword_count=len(definition.keywords),
        repaired_fields=validation_result.repaired_fields,
        keyword_adjustments=validation_result.keyword_adjustments,
    )

    return validation_result


def evaluate_ai_template_definition(
    definition: GeneratedTemplateDefinition,
    text_sources: list[str],
    extractor: OcrExtractor,
    required_fields: tuple[str, ...],
    extracted_values: dict[str, Any] | None = None,
) -> TemplateValidationResult:
    repaired_definition, repaired_fields, keyword_adjustments = repair_ai_template_definition(
        definition=definition,
        text_sources=text_sources,
        extractor=extractor,
        required_fields=required_fields,
        extracted_values=extracted_values,
    )
    template_definition: dict[str, Any] = {
        "issuer": repaired_definition.issuer,
        "keywords": list(repaired_definition.keywords),
        "fields": repaired_definition.fields,
    }
    best_payload: dict[str, Any] | None = None
    best_matched_fields: tuple[str, ...] = ()
    best_missing_fields: tuple[str, ...] = required_fields

    for text in text_sources:
        payload = extractor._match_template_definition(template_definition, text)

        if not payload:
            continue

        payload = extractor._repair_payload(payload, text)
        matched_required_fields = tuple(
            field_name for field_name in required_fields if payload.get(field_name) not in (None, "")
        )
        missing_required_fields = tuple(
            field_name for field_name in required_fields if field_name not in matched_required_fields
        )

        if (
            best_payload is None
            or len(matched_required_fields) > len(best_matched_fields)
            or (
                len(matched_required_fields) == len(best_matched_fields)
                and len(payload) > len(best_payload)
            )
        ):
            best_payload = payload
            best_matched_fields = matched_required_fields
            best_missing_fields = missing_required_fields

    text_preview = build_text_preview(text_sources)

    if best_payload is None:
        return TemplateValidationResult(
            is_valid=False,
            reason="template_did_not_match_required_fields",
            missing_required_fields=required_fields,
            matched_required_fields=(),
            repaired_fields=repaired_fields,
            keyword_adjustments=keyword_adjustments,
            effective_keywords=repaired_definition.keywords,
            text_preview=text_preview,
        )

    if best_missing_fields:
        return TemplateValidationResult(
            is_valid=False,
            reason="template_did_not_match_required_fields",
            missing_required_fields=best_missing_fields,
            matched_required_fields=best_matched_fields,
            repaired_fields=repaired_fields,
            keyword_adjustments=keyword_adjustments,
            effective_keywords=repaired_definition.keywords,
            text_preview=text_preview,
        )

    return TemplateValidationResult(
        is_valid=True,
        reason="ok",
        missing_required_fields=(),
        matched_required_fields=best_matched_fields,
        repaired_fields=repaired_fields,
        keyword_adjustments=keyword_adjustments,
        effective_keywords=repaired_definition.keywords,
        text_preview=text_preview,
    )


def repair_ai_template_definition(
    definition: GeneratedTemplateDefinition,
    text_sources: list[str],
    extractor: OcrExtractor,
    required_fields: tuple[str, ...],
    extracted_values: dict[str, Any] | None = None,
) -> tuple[GeneratedTemplateDefinition, tuple[str, ...], tuple[str, ...]]:
    matched_keywords = tuple(
        keyword
        for keyword in definition.keywords
        if any(re.search(keyword, text, re.MULTILINE | re.IGNORECASE) for text in text_sources)
    )
    keyword_adjustments: list[str] = []

    if matched_keywords and matched_keywords != definition.keywords:
        keyword_adjustments.append("pruned_non_matching_keywords")

    if not matched_keywords:
        issuer_keyword = case_insensitive_pattern(flexible_pattern(definition.issuer))

        if issuer_keyword and any(re.search(issuer_keyword, text, re.MULTILINE | re.IGNORECASE) for text in text_sources):
            matched_keywords = (issuer_keyword,)
            keyword_adjustments.append("backfilled_issuer_keyword")
        else:
            matched_keywords = definition.keywords

    repaired_fields: list[str] = []
    normalized_fields: dict[str, str] = {}

    for field_name in required_fields:
        pattern = definition.fields[field_name]
        expected_value = extracted_values.get(field_name) if isinstance(extracted_values, dict) else None
        repaired_pattern = repair_generated_field_pattern(
            field_name=field_name,
            pattern=pattern,
            text_sources=text_sources,
            extractor=extractor,
            expected_value=expected_value,
        )
        normalized_fields[field_name] = repaired_pattern

        if repaired_pattern != pattern:
            repaired_fields.append(field_name)

    return (
        GeneratedTemplateDefinition(
            issuer=definition.issuer,
            keywords=matched_keywords,
            fields=normalized_fields,
        ),
        tuple(sorted(repaired_fields)),
        tuple(keyword_adjustments),
    )


def repair_generated_field_pattern(
    *,
    field_name: str,
    pattern: str,
    text_sources: list[str],
    extractor: OcrExtractor,
    expected_value: Any = None,
) -> str:
    for candidate in generate_pattern_variants(pattern):
        if field_pattern_matches(
            field_name=field_name,
            pattern=candidate,
            text_sources=text_sources,
            extractor=extractor,
            expected_value=expected_value,
        ):
            return candidate

    return pattern


def generate_pattern_variants(pattern: str) -> tuple[str, ...]:
    candidates = [pattern]

    if " " in pattern:
        candidates.append(pattern.replace(" ", r"\s+"))

    for candidate in list(candidates):
        for source, target in (
            (r"\s+(", r"\s*[:#-]?\s*("),
            (r"\s*(", r"\s*[:#-]?\s*("),
            (r"\s+(", r"[\s\S]{0,120}?("),
            (r"\s*[:#-]?\s*(", r"[\s\S]{0,120}?("),
            (r"\s*(", r"[\s\S]{0,120}?("),
        ):
            if source in candidate:
                candidates.append(candidate.replace(source, target, 1))

    return dedupe_patterns(candidates)


def field_pattern_matches(
    *,
    field_name: str,
    pattern: str,
    text_sources: list[str],
    extractor: OcrExtractor,
    expected_value: Any = None,
) -> bool:
    for text in text_sources:
        match = extractor._search_field_pattern(pattern, text)

        if match is None:
            continue

        if expected_value is None:
            return True

        matched_value = extractor._first_match_group(match)

        if normalize_validation_value(field_name, matched_value, extractor) == normalize_validation_value(field_name, expected_value, extractor):
            return True

    return False


def normalize_validation_value(field_name: str, value: Any, extractor: OcrExtractor) -> str | float | None:
    if field_name == "invoice_number":
        return extractor._normalize_invoice_number(value)

    if field_name == "amount":
        return extractor._normalize_amount(value)

    if field_name == "currency_code":
        return extractor._normalize_currency(value)

    if value is None:
        return None

    return re.sub(r"\D+", "", str(value))


def build_text_preview(text_sources: list[str], limit: int = 1200) -> str:
    combined_text = "\n\n---\n\n".join(normalize_space(text_source) for text_source in text_sources if text_source.strip())

    if len(combined_text) <= limit:
        return combined_text

    return f"{combined_text[:limit]}..."


def build_document_preview_rows(
    *,
    preview_text: str,
    definition: GeneratedTemplateDefinition,
    extractor: OcrExtractor,
    country_code: str,
    missing_labels: tuple[str, ...],
    line_item_rows: tuple[LineItemPreviewRow, ...],
) -> tuple[DocumentPreviewRow, ...]:
    payload = extractor._match_template_definition(
        {
            "issuer": definition.issuer,
            "keywords": [],
            "fields": definition.fields,
        },
        preview_text,
    ) or {"issuer": definition.issuer}
    payload = extractor._repair_payload(payload, preview_text)
    response = extractor._build_response(
        payload=payload,
        source_reader="preview",
        country_code=country_code,
    )
    fields = response.get("fields", {})
    issues: list[str] = []
    missing_required_fields = response.get("unmapped_fields", {}).get("missing_required_fields", {}).get("value", [])

    if missing_labels:
        issues.append(f"missing labels: {', '.join(missing_labels)}")

    if missing_required_fields:
        issues.append(f"missing required fields: {', '.join(missing_required_fields)}")

    if not line_item_rows:
        issues.append("line item table not reconstructed")

    total_amount = fields.get("total_amount", {}).get("value")
    total_amount_text = "-"

    if isinstance(total_amount, (int, float)):
        total_amount_text = f"{total_amount:.2f}"
    elif total_amount not in (None, ""):
        total_amount_text = str(total_amount)

    rows = (
        DocumentPreviewRow("Issuer", str(payload.get("issuer") or definition.issuer or "-"), "ok"),
        DocumentPreviewRow("Invoice Number", str(fields.get("invoice_number", {}).get("value") or "-"), "ok" if fields.get("invoice_number") else "missing"),
        DocumentPreviewRow("Invoice Date", str(fields.get("invoice_date", {}).get("value") or "-"), "ok" if fields.get("invoice_date") else "missing"),
        DocumentPreviewRow("Total Amount", total_amount_text, "ok" if fields.get("total_amount") else "missing"),
        DocumentPreviewRow("Currency", str(fields.get("currency_code", {}).get("value") or "-"), "ok" if fields.get("currency_code") else "missing"),
        DocumentPreviewRow("Line Items", str(len(line_item_rows)), "ok" if line_item_rows else "warning"),
        DocumentPreviewRow("Issues", "; ".join(issues) if issues else "None", "warning" if issues else "ok"),
    )

    return rows


def build_line_item_preview_rows(text: str, extractor: OcrExtractor | None = None) -> tuple[LineItemPreviewRow, ...]:
    """Detect and return line item rows from raw invoice text.

    Uses the extractor's generic column-detection logic so the result adapts
    to whatever column layout the invoice actually has — no hardcoded headers.
    Falls back to an empty tuple when no item table is found.
    """
    if extractor is None:
        # Lazy import to avoid circular dependency when extractor is not provided
        from ocr_service.extractor import OcrExtractor as _OcrExtractor  # noqa: PLC0415
        extractor = _OcrExtractor.__new__(_OcrExtractor)

    raw_items = extractor.extract_line_items(text)
    if not raw_items:
        return ()

    # All rows share the same set of keys; derive header order from first row
    headers = tuple(raw_items[0].keys())
    return tuple(
        LineItemPreviewRow(headers=headers, values=dict(item))
        for item in raw_items
    )


def looks_like_quantity_column(lines: list[str]) -> bool:
    return bool(lines) and all(re.fullmatch(r"\d+(?:[.,]\d+)?", line) for line in lines)


def looks_like_description_column(lines: list[str]) -> bool:
    return bool(lines) and all(not re.fullmatch(AMOUNT_VALUE_PATTERN, line) and "%" not in line for line in lines)


def looks_like_amount_column(lines: list[str]) -> bool:
    return bool(lines) and all("€" in line or re.search(AMOUNT_VALUE_PATTERN, line) for line in lines)


def looks_like_vat_column(lines: list[str]) -> bool:
    return bool(lines) and all(re.fullmatch(r"[0-9]{1,2}%", line) for line in lines)


def score_text_source(text: str) -> tuple[int, int, int]:
    lowered_text = text.lower()
    header_markers = (
        "factuurnummer",
        "invoice number",
        "factuurdatum",
        "invoice date",
        "factuurbedrag",
        "invoice total",
        "totaal netto",
        "factuur",
        "pagina",
        "betaald",
        "btw",
    )
    header_score = sum(1 for marker in header_markers if marker in lowered_text)
    value_score = (
        len(re.findall(DATE_VALUE_PATTERN, text))
        + len(re.findall(AMOUNT_VALUE_PATTERN, text))
        + len(re.findall(r"(?:EUR|€)", text, re.IGNORECASE))
    )

    return (
        header_score,
        value_score,
        len(normalize_space(text)),
    )


def choose_best_pattern(patterns: tuple[str, ...], text_sources: list[str]) -> str:
    scored_patterns = [
        (
            sum(1 for text in text_sources if re.search(pattern, text, re.MULTILINE | re.IGNORECASE)),
            -index,
            pattern,
        )
        for index, pattern in enumerate(patterns)
    ]

    return max(scored_patterns)[2]


def build_keyword_candidates(issuer: str, text_sources: list[str]) -> tuple[str, ...]:
    suggestions: list[str] = []
    seen: set[str] = set()
    scoring_lines: list[tuple[int, str]] = []

    if issuer and not is_generic_issuer(issuer) and any(contains_phrase(text, issuer) for text in text_sources):
        issuer_pattern = case_insensitive_pattern(flexible_pattern(issuer))
        suggestions.append(issuer_pattern)
        seen.add(issuer_pattern)

    for text in text_sources:
        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            score, pattern = keyword_candidate_from_line(line)

            if not pattern or pattern in seen:
                continue

            scoring_lines.append((score, pattern))

    for _score, pattern in sorted(scoring_lines, key=lambda item: item[0], reverse=True):
        if pattern in seen:
            continue

        suggestions.append(pattern)
        seen.add(pattern)

        if len(suggestions) >= 3:
            break

    if suggestions:
        return tuple(suggestions)

    fallback_issuer = detect_issuer(text_sources)

    if fallback_issuer:
        return (case_insensitive_pattern(flexible_pattern(fallback_issuer)),)

    return (case_insensitive_pattern(r"factu\W*ur"),)


def keyword_candidate_from_line(line: str) -> tuple[int, str | None]:
    lowered_line = line.lower()

    if len(line) > 100:
        return (0, None)

    if looks_like_value_line(line):
        return (0, None)

    if any(marker in lowered_line for marker in ("http://", "https://", "www.")):
        domain = extract_domain_from_text(line)

        if domain:
            return (6, case_insensitive_pattern(re.escape(domain)))

    email_match = re.search(r"[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})", line, re.IGNORECASE)

    if email_match:
        return (6, case_insensitive_pattern(re.escape(email_match.group(1).lower())))

    tax_match = re.search(r"\bNL[0-9A-Z]{9,14}\b", line, re.IGNORECASE)

    if tax_match:
        return (7, case_insensitive_pattern(re.escape(tax_match.group(0).upper())))

    kvk_match = re.search(r"\b[0-9]{8}\b", line)

    if "kvk" in lowered_line and kvk_match:
        return (6, case_insensitive_pattern(r"KvK\s*(?:nr)?[:.]?\s*" + re.escape(kvk_match.group(0))))

    iban_match = re.search(r"\b[A-Z]{2}[0-9A-Z]{13,30}\b", line)

    if iban_match:
        return (5, case_insensitive_pattern(re.escape(iban_match.group(0))))

    if looks_like_company_name(line):
        return (5, case_insensitive_pattern(flexible_pattern(line)))

    return (0, None)


def build_invoice_number_patterns(spec: TemplateSpec) -> tuple[str, ...]:
    patterns = list(
        label_patterns(
            [
                spec.invoice_number_label,
                "Factuurnummer",
                "Factuur nummer",
                "Invoice number",
                "Factuur",
            ],
            INVOICE_NUMBER_VALUE_PATTERN,
        )
    )
    patterns.extend(
        spanning_label_patterns(
            [
                spec.invoice_number_label,
                "Factuurnummer",
                "Factuur nummer",
                "Invoice number",
            ],
            INVOICE_NUMBER_VALUE_PATTERN,
            max_span=120,
        )
    )
    patterns.append(
        case_insensitive_pattern(
            r"(?:Factuurdatum|Invoice\s*date)\s*[:#-]?\s*[0-9]{2}[./-][0-9]{2}[./-][0-9]{4}\s+"
            + INVOICE_NUMBER_VALUE_PATTERN
        )
    )
    patterns.append(
        case_insensitive_pattern(
            r"(?:[0-9]{2}[./-][0-9]{2}[./-][0-9]{4})\s+[0-9]{2}:\d{2}(?::\d{2})?\s+[^\n0-9A-Z]{0,4}(?:[A-Z]\s+)?"
            r"((?=[A-Z0-9/._-]*\d)[A-Z0-9][A-Z0-9/._]*(?:\s*-\s*[A-Z0-9][A-Z0-9/._]*)+)"
        )
    )
    patterns.append(
        case_insensitive_pattern(
            r"(?:[0-9]{2}[./-][0-9]{2}[./-][0-9]{4})\s+[0-9]{2}:\d{2}(?::\d{2})?\s+"
            r"([A-Z0-9][A-Z0-9/._]*(?:\s*-\s*[A-Z0-9][A-Z0-9/._]*)+)"
        )
    )

    return dedupe_patterns(patterns)


def build_date_patterns(spec: TemplateSpec) -> tuple[str, ...]:
    return dedupe_patterns([
        case_insensitive_pattern(r"(?m)^" + DATE_VALUE_PATTERN + r"\s+[0-9]{2}:\d{2}(?::\d{2})?\b"),
        *label_patterns(
            [
                spec.date_label,
                "Factuurdatum",
                "Invoice date",
                "Datum",
            ],
            DATE_VALUE_PATTERN,
        ),
        *spanning_label_patterns(
            [
                spec.date_label,
                "Factuurdatum",
                "Invoice date",
            ],
            DATE_VALUE_PATTERN,
            max_span=160,
        ),
    ])


def build_amount_patterns(spec: TemplateSpec) -> tuple[str, ...]:
    summary_labels = [
        spec.amount_label,
        "Totaal incl. BTW",
        "Totaal incl btw",
        "Te betalen",
        "Totaal bedrag",
        "Totaal",
        "Subtotaal",
    ]

    return dedupe_patterns([
        *line_tail_label_patterns(summary_labels, AMOUNT_VALUE_PATTERN),
        *label_patterns(summary_labels, r"(?:EUR|€)?\s*" + AMOUNT_VALUE_PATTERN),
        *spanning_label_patterns(summary_labels, r"(?:EUR|€)?\s*" + AMOUNT_VALUE_PATTERN, max_span=120),
    ])


def build_currency_patterns(spec: TemplateSpec) -> tuple[str, ...]:
    preferred_currency = spec.currency_code.strip().upper() or "EUR"
    patterns = list(
        label_patterns(
            [
                spec.currency_label,
                "Valuta",
                spec.amount_label,
                "Totaal incl. BTW",
                "Totaal bedrag",
                "Te betalen",
            ],
            currency_capture_pattern(preferred_currency),
        )
    )
    patterns.extend(
        spanning_label_patterns(
            [
                spec.currency_label,
                "Valuta",
                spec.amount_label,
                "Totaal incl. BTW",
                "Totaal bedrag",
                "Te betalen",
            ],
            currency_capture_pattern(preferred_currency),
            max_span=120,
        )
    )
    patterns.append(case_insensitive_pattern(currency_capture_pattern(preferred_currency)))

    return dedupe_patterns(patterns)


def label_patterns(labels: list[str | None], value_pattern: str) -> tuple[str, ...]:
    patterns: list[str] = []

    for label in labels:
        if not label:
            continue

        patterns.append(case_insensitive_pattern(label_capture_pattern(label, value_pattern)))

    return tuple(patterns)


def spanning_label_patterns(
    labels: list[str | None],
    value_pattern: str,
    max_span: int,
) -> tuple[str, ...]:
    patterns: list[str] = []

    for label in labels:
        if not label:
            continue

        patterns.append(case_insensitive_pattern(spanning_label_capture_pattern(label, value_pattern, max_span)))

    return tuple(patterns)


def line_tail_label_patterns(labels: list[str | None], value_pattern: str) -> tuple[str, ...]:
    patterns: list[str] = []

    for label in labels:
        if not label:
            continue

        patterns.append(case_insensitive_pattern(line_tail_label_capture_pattern(label, value_pattern)))

    return tuple(patterns)


def dedupe_patterns(patterns: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique_patterns: list[str] = []

    for pattern in patterns:
        if not pattern or pattern in seen:
            continue

        unique_patterns.append(pattern)
        seen.add(pattern)

    return tuple(unique_patterns)


def resolve_issuer(issuer: str, text_sources: list[str]) -> str:
    stripped_issuer = issuer.strip()

    if stripped_issuer and not is_generic_issuer(stripped_issuer):
        return stripped_issuer

    detected_issuer = detect_issuer(text_sources)

    if detected_issuer:
        return detected_issuer

    return stripped_issuer or "Generated Supplier"


def detect_issuer(text_sources: list[str]) -> str | None:
    best_score = -1
    best_line: str | None = None

    for text in text_sources:
        for index, raw_line in enumerate(text.splitlines()[:25]):
            line = raw_line.strip()

            if not line:
                continue

            score = issuer_line_score(line, index)

            if score > best_score:
                best_score = score
                best_line = line

    return best_line


def issuer_line_score(line: str, index: int) -> int:
    lowered_line = line.lower()

    if len(line) < 4:
        return -10

    if any(marker in lowered_line for marker in ("factuur", "btw", "kvk", "iban", "@", "www.", "http")):
        return -10

    if re.search(r"\d", line) and len(line) > 24:
        return -5

    score = max(0, 15 - index)

    if looks_like_company_name(line):
        score += 10

    if re.search(r"\b(b\.?v\.?|vof|shop|company|holding|services?)\b", lowered_line, re.IGNORECASE):
        score += 10

    if len(line.split()) <= 5:
        score += 2

    return score


def looks_like_company_name(line: str) -> bool:
    lowered_line = line.lower()

    if len(line) < 3 or len(line) > 60:
        return False

    if any(marker in lowered_line for marker in ("adres", "factuur", "btw", "kvk", "iban", "omschrijving")):
        return False

    if "@" in line or "www." in lowered_line:
        return False

    if looks_like_value_line(line):
        return False

    return bool(re.search(r"[A-Za-z]", line))


def looks_like_value_line(line: str) -> bool:
    if re.search(DATE_VALUE_PATTERN, line):
        return True

    if re.search(AMOUNT_VALUE_PATTERN, line):
        return True

    if len(re.findall(r"\d+", line)) >= 3 and not re.search(r"\b(b\.?v\.?|vof|shop|company|holding|services?)\b", line, re.IGNORECASE):
        return True

    if any(marker in line.lower() for marker in ("betaling", "bestelnummer", "leveringsnummer")) and re.search(r"\d", line):
        return True

    return False


def keyword_looks_invoice_specific(keyword: str, text_sources: list[str] | None = None) -> bool:
    if keyword_looks_supplier_identifier(keyword, text_sources):
        return False

    if re.search(DATE_VALUE_PATTERN, keyword):
        return True

    if re.search(AMOUNT_VALUE_PATTERN, keyword):
        return True

    if len(re.findall(r"\d", keyword)) >= 8:
        return True

    if re.fullmatch(r"[A-Z0-9._/-]+", keyword, re.IGNORECASE) and re.search(r"\d", keyword):
        return True

    if any(marker in keyword.lower() for marker in ("factuurnummer", "invoice_number", "bestelnummer", "leveringsnummer")):
        return True

    return False


def keyword_looks_supplier_identifier(keyword: str, text_sources: list[str] | None = None) -> bool:
    lowered_keyword = keyword.lower()
    digit_count = len(re.findall(r"\d", keyword))

    if extract_domain_from_text(keyword):
        return True

    if "@" in keyword and re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", keyword, re.IGNORECASE):
        return True

    if "kvk" in lowered_keyword and digit_count >= 8:
        return True

    if any(marker in lowered_keyword for marker in ("btw", "vat")) and digit_count >= 8:
        return True

    if "nl" in lowered_keyword and "b" in lowered_keyword and digit_count >= 8:
        return True

    if not text_sources:
        return False

    keyword_without_flags = re.sub(r"\(\?[a-z]+\)", "", keyword, flags=re.IGNORECASE)
    normalized_literal = normalize_space(
        keyword_without_flags
        .replace(r"\s+", " ")
        .replace(r"\s*", " ")
        .replace(r"\.", ".")
        .replace("\\", "")
    )

    if not normalized_literal:
        return False

    for text in text_sources:
        for raw_line in text.splitlines():
            line = normalize_space(raw_line)
            lowered_line = line.lower()

            if normalized_literal.lower() not in lowered_line:
                continue

            if any(marker in lowered_line for marker in ("kvk", "btw", "vat", "www.", "http://", "https://", "@")):
                return True

    return False


def is_generic_issuer(value: str) -> bool:
    return normalize_space(value).lower() in GENERIC_ISSUER_VALUES


def contains_phrase(text: str, phrase: str) -> bool:
    return normalize_space(phrase).lower() in normalize_space(text).lower()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_domain_from_text(line: str) -> str | None:
    domain_match = re.search(r"(?:https?://)?(?:www\.)?([A-Z0-9.-]+\.[A-Z]{2,})", line, re.IGNORECASE)

    if not domain_match:
        return None

    return domain_match.group(1).lower()


def case_insensitive_pattern(pattern: str) -> str:
    return pattern if pattern.startswith("(?i)") else f"(?i){pattern}"


def suggest_keywords(issuer: str, ocr_text: str) -> tuple[str, ...]:
    return build_keyword_candidates(issuer, [ocr_text])


def label_capture_pattern(label: str, value_pattern: str) -> str:
    return f"(?<!\\w){flexible_pattern(label)}(?!\\w)\\s*[:#-]?\\s*{value_pattern}"


def spanning_label_capture_pattern(label: str, value_pattern: str, max_span: int) -> str:
    return f"(?<!\\w){flexible_pattern(label)}(?!\\w)[\\s\\S]{{0,{max_span}}}?{value_pattern}"


def line_tail_label_capture_pattern(label: str, value_pattern: str) -> str:
    return f"(?m)^.*?(?<!\\w){flexible_pattern(label)}(?!\\w)[^\\n\\r]*{value_pattern}\\s*$"


def currency_capture_pattern(currency_code: str) -> str:
    normalized_code = currency_code.strip().upper() or "EUR"

    if normalized_code == "EUR":
        return "((?:EUR|€))"

    return f"(({re.escape(normalized_code)}))"


def flexible_pattern(text: str) -> str:
    stripped_text = text.strip()

    if not stripped_text:
        return ""

    return re.escape(stripped_text).replace(r"\ ", r"\s+").replace(r"\-", r"[-–—]?")


def slugify(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return normalized.strip("_") or "supplier"


def yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an invoice2data starter template.")
    parser.add_argument("sample", type=Path, help="Path to a sample invoice file.")
    parser.add_argument("--issuer", required=True, help="Supplier or issuer name.")
    parser.add_argument("--invoice-number-label", required=True, help="Label used for invoice number.")
    parser.add_argument("--date-label", required=True, help="Label used for invoice date.")
    parser.add_argument("--amount-label", required=True, help="Label used for invoice total amount.")
    parser.add_argument("--country-code", default="NL", help="Country code used for OCR language selection.")
    parser.add_argument("--currency-code", default="EUR", help="Currency code to capture.")
    parser.add_argument("--currency-label", help="Optional dedicated label used for currency.")
    parser.add_argument(
        "--keyword",
        action="append",
        dest="keywords",
        default=[],
        help="Optional extra template keyword. Repeat the flag for multiple keywords.",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path("/app/templates"),
        help="Base template directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit output path for the generated template.",
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    extractor = OcrExtractor(arguments.template_dir)
    result = generate_starter_template_from_sample(
        sample_path=arguments.sample,
        template_dir=arguments.template_dir,
        extractor=extractor,
        spec=TemplateSpec(
            issuer=arguments.issuer,
            invoice_number_label=arguments.invoice_number_label,
            date_label=arguments.date_label,
            amount_label=arguments.amount_label,
            country_code=arguments.country_code,
            currency_code=arguments.currency_code,
            currency_label=arguments.currency_label,
            keywords=tuple(arguments.keywords),
        ),
        output_path=arguments.output,
    )

    print(f"Template written to: {result.output_path}")

    if result.missing_labels:
        print(
            "Labels not found in OCR/PDF text: "
            + ", ".join(result.missing_labels)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
