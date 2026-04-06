from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import yaml

from ocr_service.fields import DEFAULT_REQUIRED_FIELDS, public_field_name
from ocr_service.logger import get_logger, log_event


class OcrExtractor:
    SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
    INVOICE_NUMBER_VALUE_PATTERN = r"((?=[A-Z0-9/._-]*\d)[A-Z0-9][A-Z0-9/._-]+)"
    GENERIC_ISSUER_VALUES = frozenset({"", "n/a", "na", "onbekend", "supplier", "unknown", "vendor"})
    DATE_VALUE_PATTERN = r"([0-9]{2}[./-][0-9]{2}[./-][0-9]{4})"
    AMOUNT_VALUE_PATTERN = r"([0-9]{1,3}(?:[.,][0-9]{3})*[.,][0-9]{2}|[0-9]+[.,][0-9]{2})"
    logger = get_logger("ocr_service.extractor")

    def __init__(
        self,
        template_dir: Path,
        template_dirs: list[Path] | tuple[Path, ...] | None = None,
        writable_template_dir: Path | None = None,
    ) -> None:
        self.template_dir = writable_template_dir or template_dir
        self.template_dirs = list(template_dirs or [template_dir])
        self.writable_template_dir = writable_template_dir or self.template_dir

    def extract(
        self,
        file_path: Path,
        country_code: str = "NL",
        required_fields: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        country_code = (country_code or os.getenv("OCR_DEFAULT_COUNTRY_CODE", "NL")).upper()
        required_fields = required_fields or DEFAULT_REQUIRED_FIELDS
        responses: list[dict[str, Any]] = []
        started_at = perf_counter()

        log_event(
            self.logger,
            logging.INFO,
            "extractor.extract.started",
            file_name=file_path.name,
            country_code=country_code,
            required_fields=required_fields,
        )

        try:
            if file_path.suffix.lower() == ".pdf":
                searchable_text = self._load_input_text(file_path, "pdftotext")
                searchable_payload = self._run_invoice2data(file_path, input_reader="pdftotext")

                if searchable_payload:
                    searchable_payload = self._repair_payload(searchable_payload, searchable_text)
                    searchable_response = self._build_response(
                        searchable_payload,
                        source_reader="pdftotext",
                        country_code=country_code,
                        required_fields=required_fields,
                    )
                    searchable_response["line_items"] = self.extract_line_items(searchable_text)
                    responses.append(searchable_response)

                    log_event(
                        self.logger,
                        logging.INFO,
                        "extractor.searchable.completed",
                        file_name=file_path.name,
                        status=searchable_response["status"],
                        field_names=sorted(searchable_response["fields"].keys()),
                    )

                    if searchable_response["status"] == "success":
                        log_event(
                            self.logger,
                            logging.INFO,
                            "extractor.extract.completed",
                            file_name=file_path.name,
                            status=searchable_response["status"],
                            duration_ms=round((perf_counter() - started_at) * 1000, 2),
                        )
                        return searchable_response

            ocr_text = self._extract_ocr_text(file_path, self._language_for_country(country_code))

            if ocr_text.strip():
                with TemporaryDirectory() as temporary_directory:
                    temporary_path = Path(temporary_directory)
                    extracted_text_path = temporary_path / f"{file_path.stem}.txt"
                    extracted_text_path.write_text(ocr_text, encoding="utf-8")

                    ocr_payload = self._run_invoice2data(extracted_text_path, input_reader="text")

                    if ocr_payload:
                        ocr_payload = self._repair_payload(ocr_payload, ocr_text)
                        ocr_response = self._build_response(
                            ocr_payload,
                            source_reader="tesseract",
                            country_code=country_code,
                            required_fields=required_fields,
                        )
                        ocr_response["line_items"] = self.extract_line_items(ocr_text)
                        responses.append(ocr_response)
                        log_event(
                            self.logger,
                            logging.INFO,
                            "extractor.ocr.completed",
                            file_name=file_path.name,
                            status=ocr_response["status"],
                            field_names=sorted(ocr_response["fields"].keys()),
                        )

            if responses:
                selected_response = max(responses, key=self._score_response)
                log_event(
                    self.logger,
                    logging.INFO,
                    "extractor.extract.completed",
                    file_name=file_path.name,
                    status=selected_response["status"],
                    response_count=len(responses),
                    duration_ms=round((perf_counter() - started_at) * 1000, 2),
                )
                return selected_response

            failed_response = self._failed_response(country_code, required_fields)
            log_event(
                self.logger,
                logging.WARNING,
                "extractor.extract.completed",
                file_name=file_path.name,
                status=failed_response["status"],
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
            )
            return failed_response
        except Exception:
            log_event(
                self.logger,
                logging.ERROR,
                "extractor.extract.failed",
                file_name=file_path.name,
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
            )
            self.logger.exception("extractor.extract.failed")
            raise

    def template_count(self) -> int:
        return len(list(self._iter_templates()))

    def _failed_response(self, country_code: str, required_fields: tuple[str, ...] | None = None) -> dict[str, Any]:
        return {
            "status": "failed",
            "message": "No matching invoice template found for this document.",
            "fields": {},
            "unmapped_fields": {
                "country_code": {
                    "value": country_code,
                    "confidence": 1.0,
                }
            },
        }

    def _score_response(self, response: dict[str, Any]) -> tuple[int, int]:
        status_priority = {
            "failed": 0,
            "partial": 1,
            "success": 2,
        }

        return (
            status_priority.get(str(response.get("status")), 0),
            len(response.get("fields", {})),
        )

    def _language_for_country(self, country_code: str) -> str:
        if country_code == "NL":
            return os.getenv("OCR_DUTCH_LANGUAGE", "nld+eng")

        return os.getenv("OCR_DEFAULT_LANGUAGE", "eng")

    def _run_invoice2data(self, input_path: Path, input_reader: str) -> dict[str, Any] | None:
        started_at = perf_counter()

        log_event(
            self.logger,
            logging.DEBUG,
            "extractor.invoice2data.started",
            input_name=input_path.name,
            input_reader=input_reader,
        )

        with TemporaryDirectory() as temporary_directory:
            flattened_template_dir = Path(temporary_directory) / "templates"
            flattened_template_dir.mkdir(parents=True, exist_ok=True)
            self._flatten_templates(flattened_template_dir)

            command = [
                "invoice2data",
                "--exclude-built-in-templates",
                "--template-folder",
                str(flattened_template_dir),
                "--input-reader",
                input_reader,
                "--output-format",
                "json",
                "--output-date-format",
                "%Y-%m-%d",
                str(input_path),
            ]

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

        log_event(
            self.logger,
            logging.INFO,
            "extractor.invoice2data.completed",
            input_name=input_path.name,
            input_reader=input_reader,
            return_code=process.returncode,
            stdout_present=bool(process.stdout.strip()),
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
        )

        if process.returncode != 0 or not process.stdout.strip():
            log_event(
                self.logger,
                logging.DEBUG,
                "extractor.invoice2data.fallback",
                input_name=input_path.name,
                input_reader=input_reader,
                reason="empty_or_failed_process",
            )
            return self._run_template_regex_fallback(input_path, input_reader)

        try:
            payload = json.loads(process.stdout)
        except json.JSONDecodeError:
            log_event(
                self.logger,
                logging.WARNING,
                "extractor.invoice2data.fallback",
                input_name=input_path.name,
                input_reader=input_reader,
                reason="invalid_json",
            )
            return self._run_template_regex_fallback(input_path, input_reader)

        if isinstance(payload, list):
            if not payload:
                log_event(
                    self.logger,
                    logging.DEBUG,
                    "extractor.invoice2data.fallback",
                    input_name=input_path.name,
                    input_reader=input_reader,
                    reason="empty_payload",
                )
                return self._run_template_regex_fallback(input_path, input_reader)

            payload = payload[0]

        if not isinstance(payload, dict):
            log_event(
                self.logger,
                logging.WARNING,
                "extractor.invoice2data.fallback",
                input_name=input_path.name,
                input_reader=input_reader,
                reason="non_object_payload",
            )
            return self._run_template_regex_fallback(input_path, input_reader)

        return payload

    def _run_template_regex_fallback(
        self,
        input_path: Path,
        input_reader: str,
    ) -> dict[str, Any] | None:
        log_event(
            self.logger,
            logging.DEBUG,
            "extractor.regex_fallback.started",
            input_name=input_path.name,
            input_reader=input_reader,
        )

        if input_reader != "text":
            return None

        text = self._load_input_text(input_path, input_reader)

        if not text.strip():
            return None

        candidate_payloads: list[tuple[dict[str, Any], dict[str, Any]]] = []

        for template_path in self._iter_templates():
            payload = self._match_template(template_path, text)

            if payload is not None:
                candidate_payloads.append((payload, self._repair_payload(payload, text)))

        if not candidate_payloads:
            log_event(
                self.logger,
                logging.DEBUG,
                "extractor.regex_fallback.completed",
                input_name=input_path.name,
                matched_templates=0,
            )
            return None

        log_event(
            self.logger,
            logging.INFO,
            "extractor.regex_fallback.completed",
            input_name=input_path.name,
            matched_templates=len(candidate_payloads),
        )

        original_payload, repaired_payload = max(
            candidate_payloads,
            key=lambda candidate: self._score_payload_match(
                candidate[1],
                original_payload=candidate[0],
            ),
        )

        log_event(
            self.logger,
            logging.DEBUG,
            "extractor.regex_fallback.selected",
            input_name=input_path.name,
            matched_field_count=sum(1 for key, value in original_payload.items() if key != "issuer" and value),
            repaired_field_count=sum(1 for key, value in repaired_payload.items() if key != "issuer" and value),
        )

        return repaired_payload

    def _score_payload_match(
        self,
        payload: dict[str, Any],
        original_payload: dict[str, Any] | None = None,
    ) -> tuple[int, ...]:
        mapped_priority = [
            "amount",
            "total_amount",
            "date",
            "invoice_date",
            "invoice_number",
            "invoice_no",
            "invoice_id",
            "number",
            "currency_code",
            "currency",
            "currency_symbol",
        ]
        mapped_count = sum(1 for key in mapped_priority if payload.get(key))
        invoice_number = self._normalize_invoice_number(
            self._first_value(payload, ["invoice_number", "invoice_no", "invoice_id", "number"])
        )
        normalized_amount = self._normalize_amount(
            self._first_value(payload, ["total_amount", "amount", "invoice_total", "total"])
        )
        currency_code = self._normalize_currency(
            self._first_value(payload, ["currency_code", "currency", "currency_symbol"])
        )
        base_score = (
            mapped_count,
            1 if invoice_number else 0,
            1 if normalized_amount is not None else 0,
            1 if currency_code else 0,
            len(payload),
        )

        if original_payload is None:
            return base_score

        direct_match_count = sum(1 for key in mapped_priority if original_payload.get(key))

        return (
            mapped_count,
            direct_match_count,
            *base_score[1:],
        )

    def _load_input_text(self, input_path: Path, input_reader: str) -> str:
        if input_reader == "text":
            try:
                return input_path.read_text(encoding="utf-8")
            except OSError:
                log_event(
                    self.logger,
                    logging.WARNING,
                    "extractor.input_text.failed",
                    input_name=input_path.name,
                    input_reader=input_reader,
                )
                return ""

        if input_reader != "pdftotext":
            return ""

        process = subprocess.run(
            [
                "pdftotext",
                str(input_path),
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if process.returncode != 0:
            log_event(
                self.logger,
                logging.WARNING,
                "extractor.input_text.failed",
                input_name=input_path.name,
                input_reader=input_reader,
                return_code=process.returncode,
            )
            return ""

        return process.stdout

    def _match_template(self, template_path: Path, text: str) -> dict[str, Any] | None:
        try:
            template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return None

        return self._match_template_definition(template, text)

    def _match_template_definition(self, template: dict[str, Any], text: str) -> dict[str, Any] | None:
        if not isinstance(template, dict):
            return None

        keywords = template.get("keywords", [])
        keyword_matches = 0
        valid_keyword_count = 0

        if isinstance(keywords, list) and keywords:
            for keyword in keywords:
                if not isinstance(keyword, str):
                    continue

                valid_keyword_count += 1

                if re.search(keyword, text, re.MULTILINE | re.IGNORECASE) is not None:
                    keyword_matches += 1

            if valid_keyword_count and keyword_matches < max(1, math.ceil(valid_keyword_count * 0.5)):
                return None

        fields = template.get("fields", {})

        if not isinstance(fields, dict):
            return None

        payload: dict[str, Any] = {}
        issuer = template.get("issuer")

        if isinstance(issuer, str) and issuer:
            payload["issuer"] = issuer

        for field_name, pattern in fields.items():
            if not isinstance(field_name, str) or not isinstance(pattern, str):
                continue

            match = self._search_field_pattern(pattern, text)

            if match is None:
                continue

            payload[field_name] = self._first_match_group(match)

        if len(payload) <= 1:
            return None

        return payload

    def _search_field_pattern(self, pattern: str, text: str) -> re.Match[str] | None:
        effective_pattern = (
            self._literal_field_pattern(pattern)
            if self._looks_like_literal_field_pattern(pattern)
            else pattern
        )

        return re.search(effective_pattern, text, re.MULTILINE | re.IGNORECASE)

    def _looks_like_literal_field_pattern(self, pattern: str) -> bool:
        stripped_pattern = pattern.strip()

        if not stripped_pattern:
            return True

        return re.search(r"[\\\[\](){}*+?|^$]", stripped_pattern) is None

    def _literal_field_pattern(self, pattern: str) -> str:
        tokens = pattern.strip().split()

        if not tokens:
            return re.escape(pattern)

        return r"\s+".join(re.escape(token) for token in tokens)

    def _first_match_group(self, match: re.Match[str]) -> str:
        groups = [group for group in match.groups() if group]

        if groups:
            return groups[0]

        return match.group(0)

    def _extract_ocr_text(self, file_path: Path, language: str) -> str:
        started_at = perf_counter()

        with TemporaryDirectory() as temporary_directory:
            working_directory = Path(temporary_directory)
            images = (
                self._convert_pdf_to_images(file_path, working_directory)
                if file_path.suffix.lower() == ".pdf"
                else [file_path]
            )
            extracted_pages: list[str] = []

            for image_path in images:
                process = subprocess.run(
                    [
                        "tesseract",
                        str(image_path),
                        "stdout",
                        "-l",
                        language,
                        "--psm",
                        "6",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if process.returncode == 0 and process.stdout.strip():
                    extracted_pages.append(process.stdout.strip())

            extracted_text = "\n\n".join(extracted_pages)

        log_event(
            self.logger,
            logging.INFO,
            "extractor.tesseract.completed",
            file_name=file_path.name,
            language=language,
            image_count=len(images),
            extracted_page_count=len(extracted_pages),
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
        )

        return extracted_text

    def _convert_pdf_to_images(self, file_path: Path, output_directory: Path) -> list[Path]:
        output_prefix = output_directory / "page"
        started_at = perf_counter()

        process = subprocess.run(
            [
                "pdftoppm",
                "-png",
                str(file_path),
                str(output_prefix),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if process.returncode != 0:
            log_event(
                self.logger,
                logging.WARNING,
                "extractor.pdf_to_images.failed",
                file_name=file_path.name,
                return_code=process.returncode,
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
            )
            return []

        image_paths = sorted(output_directory.glob("page-*.png"))

        log_event(
            self.logger,
            logging.DEBUG,
            "extractor.pdf_to_images.completed",
            file_name=file_path.name,
            image_count=len(image_paths),
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
        )

        return image_paths

    def _build_response(
        self,
        payload: dict[str, Any],
        source_reader: str,
        country_code: str,
        required_fields: tuple[str, ...] | None = None,
        field_confidences: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        required_fields = required_fields or DEFAULT_REQUIRED_FIELDS
        field_confidences = field_confidences or {}
        confidence = 0.98 if source_reader == "pdftotext" else 0.78
        fields: dict[str, dict[str, Any]] = {}

        invoice_date = self._first_value(payload, ["invoice_date", "date"])

        if invoice_date:
            fields["invoice_date"] = {
                "value": str(invoice_date),
                "confidence": self._resolve_field_confidence(
                    field_confidences,
                    "date",
                    confidence,
                ),
            }

        invoice_number = self._normalize_invoice_number(
            self._first_value(
                payload,
                ["invoice_number", "invoice_no", "invoice_id", "number"],
            )
        )

        if invoice_number:
            fields["invoice_number"] = {
                "value": invoice_number,
                "confidence": self._resolve_field_confidence(
                    field_confidences,
                    "invoice_number",
                    confidence,
                ),
            }

        normalized_amount = self._normalize_amount(
            self._first_value(payload, ["total_amount", "amount", "invoice_total", "total"])
        )

        if normalized_amount is not None:
            fields["total_amount"] = {
                "value": normalized_amount,
                "confidence": self._resolve_field_confidence(
                    field_confidences,
                    "amount",
                    confidence,
                ),
            }

        currency_code = self._normalize_currency(
            self._first_value(payload, ["currency_code", "currency", "currency_symbol"])
        )

        if currency_code:
            fields["currency_code"] = {
                "value": currency_code,
                "confidence": self._resolve_field_confidence(
                    field_confidences,
                    "currency_code",
                    confidence,
                ),
            }

        required_public_fields = {public_field_name(field_name) for field_name in required_fields}
        missing_required_fields = required_public_fields.difference(fields.keys())

        if required_public_fields.issubset(fields.keys()):
            status = "success"
            message = "Invoice fields extracted successfully."
        elif fields:
            status = "partial"
            message = "Invoice matched a template, but some required fields are missing."
        else:
            status = "failed"
            message = "No matching invoice template found for this document."

        unmapped_fields: dict[str, dict[str, Any]] = {
            "source_reader": {
                "value": source_reader,
                "confidence": 1.0,
            },
            "country_code": {
                "value": country_code,
                "confidence": 1.0,
            },
        }

        if payload.get("issuer"):
            unmapped_fields["issuer"] = {
                "value": str(payload["issuer"]),
                "confidence": confidence,
            }

        if missing_required_fields:
            unmapped_fields["missing_required_fields"] = {
                "value": sorted(missing_required_fields),
                "confidence": 1.0,
            }

        for key, value in payload.items():
            if key in {
                "amount",
                "currency",
                "currency_code",
                "currency_symbol",
                "date",
                "invoice_date",
                "invoice_id",
                "invoice_no",
                "invoice_number",
                "invoice_total",
                "issuer",
                "number",
                "total",
                "total_amount",
            }:
                continue

            unmapped_fields[key] = {
                "value": value,
                "confidence": confidence,
            }

        log_event(
            self.logger,
            logging.DEBUG,
            "extractor.response.built",
            source_reader=source_reader,
            status=status,
            field_names=sorted(fields.keys()),
            missing_required_fields=sorted(missing_required_fields),
        )

        return {
            "status": status,
            "message": message,
            "fields": fields,
            "unmapped_fields": unmapped_fields,
            "line_items": [],  # populated separately when raw text is available
        }

    def extract_line_items(self, text: str) -> list[dict[str, Any]]:
        """Parse line items from raw invoice text without any hardcoded column headers.

        Tries two layouts that pdftotext commonly produces:

        **Layout A — space-aligned columns** (single-line rows, columns separated by 2+ spaces):
            Qty  Description        Unit Price  Amount
            1    Widget A           10.00       10.00
            2    Widget B            5.00       10.00

        **Layout B — paragraph-per-column** (each column is a blank-line-separated block):
            Qty

            Description

            Unit Price

            1
            2

            Widget A
            Widget B

            10.00
            5.00

        Returns a list of dicts keyed by the detected column headers.
        """
        # Try Layout A: space-aligned single-line rows
        items = self._extract_line_items_inline(text)
        if items:
            return items

        # Try Layout B: paragraph-per-column (pdftotext vertical block format)
        return self._extract_line_items_columnar(text)

    def _extract_line_items_inline(self, text: str) -> list[dict[str, Any]]:
        """Layout A: single-line rows with columns separated by 2+ spaces."""
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        header_idx = self._find_line_item_header_row(lines)
        if header_idx is None:
            return []

        raw_headers = self._split_columns(lines[header_idx])
        if len(raw_headers) < 2:
            return []

        headers = [h.strip().lower().replace(" ", "_") for h in raw_headers]

        _TOTALS = re.compile(
            r"(subtotal|totaal|total|sum|bedrag|btw|vat|tax|subtotaal)",
            re.IGNORECASE,
        )
        items: list[dict[str, Any]] = []

        for line in lines[header_idx + 1:]:
            stripped = line.strip()
            if not stripped:
                continue
            if _TOTALS.search(stripped) and not self._row_has_item_amount(stripped):
                break
            cols = self._split_columns(stripped)
            if not cols or not any(re.search(r"\d", c) for c in cols):
                continue
            while len(cols) < len(headers):
                cols.append("")
            items.append({headers[i]: cols[i].strip() for i in range(len(headers))})

        return items

    def _extract_line_items_columnar(self, text: str) -> list[dict[str, Any]]:
        """Layout B: paragraph-per-column (pdftotext vertical strip format).

        pdftotext renders some multi-column tables as a series of blank-line
        separated blocks.  Two sub-variants exist:

        B1 — header + values in one block per column:
            [Qty]  [Description]  [Amount]
            [1]    [Widget A]     [10.00]
            [2]    [Widget B]      [5.00]

        B2 — headers as single-line blocks, value blocks follow separately:
            [Qty]  [Description]  [Amount]   ← single-line header blocks
            [1,2]  [A,B]          [10.00,5]  ← multi-line value blocks
        """
        _WORD_ONLY = re.compile(r"^[A-Za-z\s/.()\-#%]+$")
        paragraphs = [
            [ln.strip() for ln in block.splitlines() if ln.strip()]
            for block in re.split(r"\n\s*\n+", text)
            if block.strip()
        ]
        if len(paragraphs) < 2:
            return []

        # --- Sub-variant B1: each paragraph is [header, val1, val2, ...] ---
        for start in range(len(paragraphs)):
            run: list[list[str]] = []
            for para in paragraphs[start:]:
                if len(para) < 2:
                    break
                if not _WORD_ONLY.match(para[0]):
                    break
                run.append(para)
            if len(run) >= 2 and len({len(p) for p in run}) == 1:
                if any(re.search(self.AMOUNT_VALUE_PATTERN, "\n".join(p[1:])) for p in run):
                    headers = [p[0].strip().lower().replace(" ", "_").replace(".", "") for p in run]
                    row_count = len(run[0]) - 1
                    return [
                        {headers[ci]: run[ci][ri + 1] for ci in range(len(headers))}
                        for ri in range(row_count)
                    ]

        # --- Sub-variant B2: consecutive single-line header blocks,
        #     followed by equal-length value blocks ---
        for start in range(len(paragraphs)):
            # Collect header names. A header is:
            #   • A single-line paragraph whose content is word-only, OR
            #   • The last line of the *previous* paragraph when that line is
            #     word-only (handles pdftotext artefacts like a block that ends
            #     with a column label: ['Factuur', '04/08/2024', '107002', 'Aantal'])
            header_names: list[str] = []
            idx = start

            # Check if the paragraph just before `start` contributes a trailing header
            if start > 0:
                prev = paragraphs[start - 1]
                if prev and _WORD_ONLY.match(prev[-1]):
                    header_names.append(prev[-1].strip().lower().replace(" ", "_").replace(".", ""))

            while idx < len(paragraphs):
                para = paragraphs[idx]
                if len(para) == 1 and _WORD_ONLY.match(para[0]):
                    header_names.append(para[0].strip().lower().replace(" ", "_").replace(".", ""))
                    idx += 1
                else:
                    break

            if len(header_names) < 2:
                continue

            # Collect exactly len(header_names) value blocks of equal length
            value_blocks: list[list[str]] = []
            for vi in range(len(header_names)):
                if idx + vi >= len(paragraphs):
                    break
                value_blocks.append(paragraphs[idx + vi])

            if len(value_blocks) != len(header_names):
                continue

            row_count = len(value_blocks[0])
            if row_count == 0 or any(len(b) != row_count for b in value_blocks):
                continue

            if not any(re.search(self.AMOUNT_VALUE_PATTERN, "\n".join(b)) for b in value_blocks):
                continue

            return [
                {header_names[ci]: value_blocks[ci][ri] for ci in range(len(header_names))}
                for ri in range(row_count)
            ]

        return []

    def _find_line_item_header_row(self, lines: list[str]) -> int | None:
        """Return the index of the most likely column header row.

        A header row:
        - has 2+ columns when split on 2+ spaces
        - contains mostly words (not amounts or dates)
        - is followed by at least one row that contains an amount
        """
        for idx, line in enumerate(lines):
            cols = self._split_columns(line)
            if len(cols) < 2:
                continue
            # Must be mostly word-like (no amounts dominating)
            word_cols = [c for c in cols if re.fullmatch(r"[A-Za-z\s/.()\-#%]+", c.strip())]
            if len(word_cols) < max(2, len(cols) // 2):
                continue
            # Check that at least one following line has an amount
            for follow_line in lines[idx + 1: idx + 10]:
                if re.search(self.AMOUNT_VALUE_PATTERN, follow_line):
                    return idx
        return None

    def _split_columns(self, line: str) -> list[str]:
        """Split a line on 2+ consecutive spaces — the natural column separator."""
        return [part for part in re.split(r"  +", line) if part.strip()]

    def _row_has_item_amount(self, line: str) -> bool:
        """True if the line contains an amount AND looks like an item row (has non-amount text too)."""
        has_amount = bool(re.search(self.AMOUNT_VALUE_PATTERN, line))
        has_text = bool(re.search(r"[A-Za-z]{3,}", line))
        return has_amount and has_text

    def _first_value(self, payload: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            value = payload.get(key)

            if value not in (None, ""):
                return value

        return None

    def _normalize_amount(self, value: Any) -> float | None:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        amount_text = str(value).strip()

        if not amount_text:
            return None

        normalized_amount = (
            amount_text.replace("EUR", "")
            .replace("€", "")
            .replace("\u00a0", "")
            .replace(" ", "")
        )

        if "," in normalized_amount and "." in normalized_amount:
            if normalized_amount.rfind(",") > normalized_amount.rfind("."):
                normalized_amount = normalized_amount.replace(".", "").replace(",", ".")
            else:
                normalized_amount = normalized_amount.replace(",", "")
        elif "," in normalized_amount:
            normalized_amount = normalized_amount.replace(",", ".")

        try:
            return float(Decimal(normalized_amount))
        except (InvalidOperation, ValueError):
            return None

    def _normalize_invoice_number(self, value: Any) -> str | None:
        if value is None:
            return None

        normalized_invoice_number = re.sub(r"\s+", "", str(value).strip())

        if not normalized_invoice_number:
            return None

        normalized_invoice_number = re.sub(r"^[^A-Z0-9]+|[^A-Z0-9/._-]+$", "", normalized_invoice_number, flags=re.IGNORECASE)

        if not normalized_invoice_number or not re.search(r"\d", normalized_invoice_number):
            return None

        if normalized_invoice_number.lower() in {
            "datum",
            "factuur",
            "factuurnummer",
            "invoice",
            "number",
        }:
            return None

        return normalized_invoice_number

    def _normalize_currency(self, value: Any) -> str | None:
        if value is None:
            return None

        normalized_currency = str(value).strip().upper()

        if not normalized_currency:
            return None

        if normalized_currency in {"€", "EURO", "EUROS"}:
            return "EUR"

        if len(normalized_currency) == 3:
            return normalized_currency

        return None

    def _resolve_field_confidence(
        self,
        field_confidences: dict[str, float],
        field_name: str,
        default_confidence: float,
    ) -> float:
        raw_confidence = field_confidences.get(field_name)

        if raw_confidence is None:
            return default_confidence

        try:
            return max(0.0, min(1.0, float(raw_confidence)))
        except (TypeError, ValueError):
            return default_confidence

    def _normalize_issuer(self, value: Any) -> str | None:
        if value is None:
            return None

        normalized_issuer = str(value).strip()

        if not normalized_issuer:
            return None

        if re.sub(r"\s+", " ", normalized_issuer).strip().lower() in self.GENERIC_ISSUER_VALUES:
            return None

        return normalized_issuer

    def _repair_payload(self, payload: dict[str, Any], text: str) -> dict[str, Any]:
        repaired_payload = dict(payload)
        changed_fields: set[str] = set()

        if not text.strip():
            return repaired_payload

        issuer = self._normalize_issuer(repaired_payload.get("issuer"))

        if issuer is None:
            inferred_issuer = self._infer_issuer(text)

            if inferred_issuer:
                repaired_payload["issuer"] = inferred_issuer
                changed_fields.add("issuer")

        invoice_number = self._normalize_invoice_number(
            self._first_value(repaired_payload, ["invoice_number", "invoice_no", "invoice_id", "number"])
        )

        if invoice_number is None:
            inferred_invoice_number = self._infer_invoice_number(text)

            if inferred_invoice_number:
                repaired_payload["invoice_number"] = inferred_invoice_number
                changed_fields.add("invoice_number")

        invoice_date = self._first_value(repaired_payload, ["invoice_date", "date"])

        if not invoice_date:
            inferred_invoice_date = self._infer_invoice_date(text)

            if inferred_invoice_date:
                repaired_payload["date"] = inferred_invoice_date
                changed_fields.add("date")

        total_amount = self._normalize_amount(
            self._first_value(repaired_payload, ["total_amount", "amount", "invoice_total", "total"])
        )

        if total_amount is None:
            inferred_total_amount = self._infer_total_amount(text)

            if inferred_total_amount is not None:
                repaired_payload["amount"] = inferred_total_amount
                changed_fields.add("amount")

        currency_code = self._normalize_currency(
            self._first_value(repaired_payload, ["currency_code", "currency", "currency_symbol"])
        )

        if currency_code is None:
            inferred_currency_code = self._infer_currency_code(text)

            if inferred_currency_code:
                repaired_payload["currency_code"] = inferred_currency_code
                changed_fields.add("currency_code")

        if changed_fields:
            log_event(
                self.logger,
                logging.DEBUG,
                "extractor.payload.repaired",
                changed_fields=sorted(changed_fields),
            )

        return repaired_payload

    def _infer_invoice_number(self, text: str) -> str | None:
        patterns = [
            r"Factuurnummer\s*[:#-]?\s*" + self.INVOICE_NUMBER_VALUE_PATTERN,
            r"Factuur\s+nummer\s*[:#-]?\s*" + self.INVOICE_NUMBER_VALUE_PATTERN,
            r"Invoice\s*number\s*[:#-]?\s*" + self.INVOICE_NUMBER_VALUE_PATTERN,
            r"Factuurnummer[\s\S]{0,120}?" + self.INVOICE_NUMBER_VALUE_PATTERN,
            r"Invoice\s*number[\s\S]{0,120}?" + self.INVOICE_NUMBER_VALUE_PATTERN,
            r"(?:Factuurdatum|Invoice\s*date)\s*[:#-]?\s*[0-9]{2}[./-][0-9]{2}[./-][0-9]{4}\s+" + self.INVOICE_NUMBER_VALUE_PATTERN,
            r"(?:[0-9]{2}[./-][0-9]{2}[./-][0-9]{4})\s+[0-9]{2}:\d{2}(?::\d{2})?\s+[^\n0-9A-Z]{0,4}(?:[A-Z]\s+)?((?=[A-Z0-9/._-]*\d)[A-Z0-9][A-Z0-9/._]*(?:\s*-\s*[A-Z0-9][A-Z0-9/._]*)+)",
            r"(?:[0-9]{2}[./-][0-9]{2}[./-][0-9]{4})\s+[0-9]{2}:\d{2}(?::\d{2})?\s+([A-Z0-9][A-Z0-9/._]*(?:\s*-\s*[A-Z0-9][A-Z0-9/._]*)+)",
            r"(?:[0-9]{2}[./-][0-9]{2}[./-][0-9]{4})\s+[0-9]{2}:\d{2}(?::\d{2})?\s+" + self.INVOICE_NUMBER_VALUE_PATTERN,
            r"(?:[0-9]{2}[./-][0-9]{2}[./-][0-9]{4})\s+[0-9]{2}:\d{2}(?::\d{2})?[\s\S]{0,40}?\n\s*([0-9]{10,})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)

            if match is None:
                continue

            invoice_number = self._normalize_invoice_number(self._first_match_group(match))

            if invoice_number:
                return invoice_number

        return None

    def _infer_invoice_date(self, text: str) -> str | None:
        header_row_date = self._infer_header_row_invoice_date(text)

        if header_row_date:
            return header_row_date

        patterns = [
            r"(?m)^" + self.DATE_VALUE_PATTERN + r"\s+[0-9]{2}:\d{2}(?::\d{2})?\b",
            r"Factuurdatum\s*[:#-]?\s*" + self.DATE_VALUE_PATTERN,
            r"Invoice\s*date\s*[:#-]?\s*" + self.DATE_VALUE_PATTERN,
            r"Datum\s*[:#-]?\s*" + self.DATE_VALUE_PATTERN,
            r"Factuurdatum[\s\S]{0,160}?" + self.DATE_VALUE_PATTERN,
            r"Invoice\s*date[\s\S]{0,160}?" + self.DATE_VALUE_PATTERN,
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)

            if match is not None:
                return self._first_match_group(match)

        return None

    def _infer_total_amount(self, text: str) -> str | None:
        labels = [
            "Totaal incl. BTW",
            "Totaal incl btw",
            "Te betalen",
            "Totaal bedrag",
            "Totaal",
            "Subtotaal",
        ]

        for label in labels:
            match = re.search(
                rf"{re.escape(label)}\s*[:#-]?\s*(?:EUR|€)?\s*{self.AMOUNT_VALUE_PATTERN}",
                text,
                re.MULTILINE | re.IGNORECASE,
            )

            if match is None:
                continue

            normalized_amount = self._normalize_amount(self._first_match_group(match))

            if normalized_amount is not None:
                return self._first_match_group(match)

        for label in labels:
            match = re.search(
                rf"{re.escape(label)}[\s\S]{{0,120}}?(?:EUR|€)?\s*{self.AMOUNT_VALUE_PATTERN}",
                text,
                re.MULTILINE | re.IGNORECASE,
            )

            if match is None:
                continue

            normalized_amount = self._normalize_amount(self._first_match_group(match))

            if normalized_amount is not None:
                return self._first_match_group(match)

        return None

    def _infer_currency_code(self, text: str) -> str | None:
        patterns = [
            r"Valuta\s*[:#-]?\s*((?:[A-Z]{3}|€))",
            r"Totaal\s+incl\.\s+BTW\s*[:#-]?\s*((?:[A-Z]{3}|€))",
            r"Te betalen\s*[:#-]?\s*((?:[A-Z]{3}|€))",
            r"\b((?:EUR|USD|GBP|CHF))\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)

            if match is None:
                continue

            currency_code = self._normalize_currency(self._first_match_group(match))

            if currency_code:
                return currency_code

        if "€" in text:
            return "EUR"

        return None

    def _infer_issuer(self, text: str) -> str | None:
        best_score = -1
        best_line: str | None = None

        for index, raw_line in enumerate(text.splitlines()[:20]):
            line = raw_line.strip()

            if not line:
                continue

            score = self._issuer_line_score(line, index)

            if score > best_score:
                best_score = score
                best_line = line

        return best_line

    def _infer_header_row_invoice_date(self, text: str) -> str | None:
        lines = [line.strip() for line in text.splitlines()]

        for index, line in enumerate(lines):
            lowered_line = line.lower()

            if "factuurdatum" not in lowered_line and "invoice date" not in lowered_line:
                continue

            if index + 1 >= len(lines):
                continue

            next_line = lines[index + 1]

            if not next_line:
                continue

            date_matches = re.findall(self.DATE_VALUE_PATTERN, next_line, re.MULTILINE | re.IGNORECASE)

            if date_matches:
                return date_matches[0]

        return None

    def _issuer_line_score(self, line: str, index: int) -> int:
        lowered_line = line.lower()

        if len(line) < 4 or len(line) > 60:
            return -10

        if any(marker in lowered_line for marker in ("factuur", "invoice", "btw", "kvk", "iban", "@", "www.", "http")):
            return -10

        score = max(0, 12 - index)

        if re.search(r"\b(b\.?v\.?|vof|shop|company|holding|services?)\b", lowered_line, re.IGNORECASE):
            score += 10

        if not re.search(r"\d", line):
            score += 3

        if len(line.split()) <= 5:
            score += 2

        return score

    def _flatten_templates(self, destination_directory: Path) -> None:
        copied_templates = 0

        for template_path in self._iter_templates():
            root_directory = next(
                template_root
                for template_root in self.template_dirs
                if template_path.is_relative_to(template_root)
            )
            flattened_name = "__".join((root_directory.name, *template_path.relative_to(root_directory).parts))
            shutil.copy2(template_path, destination_directory / flattened_name)
            copied_templates += 1

        log_event(
            self.logger,
            logging.DEBUG,
            "extractor.templates.flattened",
            destination_directory=destination_directory,
            copied_templates=copied_templates,
        )

    def _iter_templates(self) -> list[Path]:
        return sorted(
            {
                *[
                    template_path
                    for template_root in self.template_dirs
                    for template_path in template_root.rglob("*.yml")
                ],
                *[
                    template_path
                    for template_root in self.template_dirs
                    for template_path in template_root.rglob("*.yaml")
                ],
            }
        )
