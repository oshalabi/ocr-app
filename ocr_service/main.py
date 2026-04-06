from __future__ import annotations

import html
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from ocr_service.auth import require_api_key
from ocr_service.extractor import OcrExtractor
from ocr_service.fields import default_required_fields_from_env, normalize_required_fields
from ocr_service.llm_adapter import (
    OCR_LLM_PROVIDER_ENV,
    OCR_OPENROUTER_FALLBACK_ENV,
    build_llm_adapter,
    resolve_llm_provider,
)
from ocr_service.logger import get_logger, is_debug_enabled, log_event
from ocr_service.orchestrator import OcrOrchestrator, OcrProcessOptions
from ocr_service.schemas import ExtractResponse, HealthResponse
from ocr_service.template_generator import (
    TemplateSpec,
    generate_starter_template_from_sample,
)

app = FastAPI(title="Invoice OCR Service")
logger = get_logger("ocr_service.main")
TRUTHY_VALUES = {"1", "true", "yes", "on"}
AUTO_GENERATE_TEMPLATES_ENV = "OCR_AUTO_GENERATE_TEMPLATES"
TEMPLATE_HEALING_MAX_ATTEMPTS_ENV = "OCR_TEMPLATE_HEALING_MAX_ATTEMPTS"


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return raw_value.strip().lower() in TRUTHY_VALUES


def _llm_fallback_enabled() -> bool:
    """Return True when any LLM provider is resolved and enabled."""
    return resolve_llm_provider() != "none"


def _auto_generate_templates_enabled() -> bool:
    return _env_flag(AUTO_GENERATE_TEMPLATES_ENV)


def _runtime_ocr_options() -> tuple[bool, bool]:
    return _llm_fallback_enabled(), _auto_generate_templates_enabled()


def _runtime_configuration_notice() -> str:
    llm_fallback, auto_generate_templates = _runtime_ocr_options()
    provider = resolve_llm_provider()
    llm_status = f"enabled ({html.escape(provider)})" if llm_fallback else "disabled"
    auto_generate_status = "enabled" if auto_generate_templates else "disabled"
    template_healing_attempts = os.getenv(TEMPLATE_HEALING_MAX_ATTEMPTS_ENV, "3")

    provider_env_note = (
        f"<code>{OCR_LLM_PROVIDER_ENV}</code> or legacy "
        f"<code>{OCR_OPENROUTER_FALLBACK_ENV}</code>"
    )

    return f"""
    <div class="notice">
      <strong>Runtime config:</strong>
      LLM fallback is {llm_status} via {provider_env_note}.
      Auto-generate templates is {auto_generate_status} via <code>{AUTO_GENERATE_TEMPLATES_ENV}</code>.
      Template healing max attempts is {html.escape(template_healing_attempts)} via <code>{TEMPLATE_HEALING_MAX_ATTEMPTS_ENV}</code>.
    </div>
    """


def _parse_template_dirs() -> list[Path]:
    configured_template_dirs = os.getenv("OCR_TEMPLATE_DIRS")

    if not configured_template_dirs:
        default_template_dir = Path(os.getenv("OCR_TEMPLATE_DIR", "/app/templates"))
        return [default_template_dir]

    return [
        Path(item.strip())
        for item in configured_template_dirs.split(",")
        if item.strip()
    ]


def _build_extractor() -> OcrExtractor:
    template_dirs = _parse_template_dirs()
    writable_template_dir = Path(os.getenv("OCR_TEMPLATE_DIR", str(template_dirs[-1])))

    return OcrExtractor(
        template_dir=template_dirs[0],
        template_dirs=template_dirs,
        writable_template_dir=writable_template_dir,
    )


extractor = _build_extractor()
llm_adapter = build_llm_adapter()
orchestrator = OcrOrchestrator(extractor, llm_adapter)

log_event(
    logger,
    logging.INFO,
    "service.booted",
    debug_enabled=is_debug_enabled(),
    llm_provider=llm_adapter.provider_name,
    llm_fallback_enabled=_llm_fallback_enabled(),
    auto_generate_templates=_auto_generate_templates_enabled(),
    template_healing_max_attempts=os.getenv(TEMPLATE_HEALING_MAX_ATTEMPTS_ENV, "3"),
    template_dirs=extractor.template_dirs,
    writable_template_dir=extractor.writable_template_dir,
)


def _page_layout(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f3efe4;
        --card: #fffaf0;
        --line: #d3c6a6;
        --ink: #1f2937;
        --muted: #6b7280;
        --accent: #a84f1d;
        --accent-soft: #f4d6b8;
      }}
      * {{
        box-sizing: border-box;
      }}
      body {{
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top right, rgba(168, 79, 29, 0.15), transparent 30%),
          linear-gradient(180deg, #f7f1e4 0%, var(--bg) 100%);
      }}
      main {{
        max-width: 900px;
        margin: 0 auto;
        padding: 32px 20px 48px;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 20px 50px rgba(31, 41, 55, 0.08);
      }}
      h1, h2 {{
        margin: 0 0 12px;
      }}
      p {{
        color: var(--muted);
        line-height: 1.5;
      }}
      .nav {{
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
        flex-wrap: wrap;
      }}
      .nav a, button {{
        border: 0;
        border-radius: 999px;
        background: var(--accent);
        color: white;
        padding: 10px 18px;
        font-size: 14px;
        text-decoration: none;
        cursor: pointer;
      }}
      .nav a.secondary {{
        background: var(--accent-soft);
        color: var(--ink);
      }}
      form {{
        display: grid;
        gap: 14px;
      }}
      label {{
        display: grid;
        gap: 6px;
        font-weight: 600;
      }}
      input, textarea {{
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 12px 14px;
        font: inherit;
        background: white;
      }}
      textarea {{
        min-height: 100px;
        resize: vertical;
      }}
      .grid {{
        display: grid;
        gap: 14px;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      }}
      pre {{
        overflow: auto;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid var(--line);
        background: #1f2937;
        color: #f9fafb;
        font-size: 13px;
        line-height: 1.5;
      }}
      .notice {{
        margin: 16px 0;
        padding: 14px 16px;
        border-left: 4px solid var(--accent);
        border-radius: 12px;
        background: var(--accent-soft);
      }}
      .table-wrap {{
        overflow: auto;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: white;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
      }}
      th, td {{
        padding: 10px 12px;
        border-bottom: 1px solid var(--line);
        text-align: left;
        vertical-align: top;
      }}
      th {{
        background: #f8f1e4;
        white-space: nowrap;
      }}
      tr:last-child td {{
        border-bottom: 0;
      }}
      .status-chip {{
        display: inline-block;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }}
      .status-ok {{
        background: #dff3e4;
        color: #1f6f43;
      }}
      .status-warning {{
        background: #fff1cc;
        color: #8a6116;
      }}
      .status-missing {{
        background: #f9d8d6;
        color: #9f2f2f;
      }}
    </style>
  </head>
  <body>
    <main>{body}</main>
  </body>
</html>"""


def _validate_upload(file: UploadFile) -> tuple[str, str]:
    original_name = Path(file.filename or "receipt").name
    suffix = Path(original_name).suffix.lower()

    if suffix not in extractor.SUPPORTED_EXTENSIONS:
        log_event(
            logger,
            logging.WARNING,
            "upload.validation_failed",
            file_name=original_name,
            suffix=suffix,
            reason="unsupported_file_type",
        )
        raise HTTPException(
            status_code=422,
            detail="Unsupported file type. Please upload PDF, JPG, or PNG.",
        )

    log_event(
        logger,
        logging.DEBUG,
        "upload.validated",
        file_name=original_name,
        suffix=suffix,
    )

    return original_name, suffix


async def _store_upload(file: UploadFile) -> tuple[str, bytes]:
    original_name, _suffix = _validate_upload(file)
    file_bytes = await file.read()

    if not file_bytes:
        log_event(
            logger,
            logging.WARNING,
            "upload.validation_failed",
            file_name=original_name,
            reason="empty_file",
        )
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is empty.",
        )

    log_event(
        logger,
        logging.INFO,
        "upload.stored",
        file_name=original_name,
        size_bytes=len(file_bytes),
    )

    return original_name, file_bytes


def _extract_from_bytes(
    file_name: str,
    file_bytes: bytes,
    country_code: str,
    required_fields: tuple[str, ...] | None = None,
    llm_fallback_enabled: bool | None = None,
    auto_generate_templates: bool | None = None,
) -> dict[str, object]:
    normalized_required_fields = required_fields or default_required_fields_from_env()
    resolved_llm_fallback = _llm_fallback_enabled() if llm_fallback_enabled is None else llm_fallback_enabled
    resolved_auto_generate = (
        _auto_generate_templates_enabled() if auto_generate_templates is None else auto_generate_templates
    )

    started_at = perf_counter()
    log_event(
        logger,
        logging.INFO,
        "extract.run.started",
        file_name=file_name,
        country_code=country_code,
        required_fields=normalized_required_fields,
        llm_fallback_enabled=resolved_llm_fallback,
        auto_generate_templates=resolved_auto_generate,
    )

    try:
        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / file_name
            input_path.write_bytes(file_bytes)
            payload = orchestrator.extract(
                input_path,
                OcrProcessOptions(
                    country_code=country_code,
                    required_fields=normalized_required_fields,
                    llm_fallback_enabled=resolved_llm_fallback,
                    auto_generate_templates=resolved_auto_generate,
                ),
            )

        log_event(
            logger,
            logging.INFO,
            "extract.run.completed",
            file_name=file_name,
            status=payload.get("status"),
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
            field_names=sorted(payload.get("fields", {}).keys()),
        )
        return payload
    except Exception:
        log_event(
            logger,
            logging.ERROR,
            "extract.run.failed",
            file_name=file_name,
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
        )
        logger.exception("extract.run.failed")
        raise


def _render_json_result_page(title: str, payload: dict[str, object]) -> HTMLResponse:
    return HTMLResponse(
        _page_layout(
            title,
            f"""
            <div class="nav">
              <a href="/">OCR Playground</a>
              <a class="secondary" href="/template-generator">Template Generator</a>
            </div>
            <section class="card">
              <h1>{html.escape(title)}</h1>
              <p>The response below is the exact JSON payload produced by the OCR sidecar.</p>
              <pre>{html.escape(json.dumps(payload, indent=2, ensure_ascii=False))}</pre>
            </section>
            """,
        )
    )


@app.get("/", response_class=HTMLResponse)
def playground() -> HTMLResponse:
    return HTMLResponse(
        _page_layout(
            "OCR Playground",
            f"""
            <div class="nav">
              <a href="/">OCR Playground</a>
              <a class="secondary" href="/template-generator">Template Generator</a>
            </div>
            <section class="card">
              <h1>OCR Playground</h1>
              <p>Upload a PDF, JPG, or PNG and open the JSON response on a separate page.</p>
              {_runtime_configuration_notice()}
              <form action="/playground/result" method="post" enctype="multipart/form-data">
                <label>
                  Invoice file
                  <input type="file" name="file" accept=".pdf,.jpg,.jpeg,.png" required />
                </label>
                <label>
                  Country code
                  <input type="text" name="country_code" value="NL" maxlength="2" />
                </label>
                <label>
                  Required fields
                  <input type="text" name="required_fields" value="invoice_number,date,amount,currency_code"
                    title="Comma-separated list of fields to extract. Built-in: invoice_number, date, amount, currency_code. Add any custom field name (e.g. vat_number, iban, po_number)." />
                </label>
                <button type="submit">Run OCR</button>
              </form>
            </section>
            """,
        )
    )


@app.get("/template-generator", response_class=HTMLResponse)
def template_generator_page() -> HTMLResponse:
    return HTMLResponse(
        _page_layout(
            "Template Generator",
            """
            <div class="nav">
              <a class="secondary" href="/">OCR Playground</a>
              <a href="/template-generator">Template Generator</a>
            </div>
            <section class="card">
              <h1>Starter Template Generator</h1>
              <p>Upload a sample invoice and enter the label text printed on it. Leave label fields blank for unlabeled receipts or invoices. The sidecar will generate and save a starter <code>template.yml</code> under the writable OCR template directory.</p>
              <form action="/template-generator/result" method="post" enctype="multipart/form-data">
                <div class="grid">
                  <label>
                    Sample invoice
                    <input type="file" name="file" accept=".pdf,.jpg,.jpeg,.png" required />
                  </label>
                  <label>
                    Country code
                    <input type="text" name="country_code" value="NL" maxlength="2" />
                  </label>
                  <label>
                    Issuer
                    <input type="text" name="issuer" placeholder="Acme B.V." required />
                  </label>
                  <label>
                    Currency code
                    <input type="text" name="currency_code" value="EUR" maxlength="3" />
                  </label>
                </div>
                <div class="grid">
                  <label>
                    Invoice number label
                    <input type="text" name="invoice_number_label" placeholder="Factuurnummer or blank for unlabeled receipts" />
                  </label>
                  <label>
                    Date label
                    <input type="text" name="date_label" placeholder="Factuurdatum or blank for unlabeled receipts" />
                  </label>
                  <label>
                    Amount label
                    <input type="text" name="amount_label" placeholder="Totaal incl. btw or blank to infer totals" />
                  </label>
                  <label>
                    Currency label
                    <input type="text" name="currency_label" placeholder="Optional" />
                  </label>
                </div>
                <label>
                  Extra keywords
                  <textarea name="keywords" placeholder="One keyword per line"></textarea>
                </label>
                <button type="submit">Generate Starter Template</button>
              </form>
            </section>
            """,
        )
    )


@app.get("/health")
def health() -> HealthResponse:
    template_count = extractor.template_count()
    log_event(
        logger,
        logging.DEBUG,
        "http.health",
        templates=template_count,
    )

    return {
        "status": "ok",
        "templates": template_count,
    }


@app.post("/extract", dependencies=[Depends(require_api_key)])
async def extract(
    file: UploadFile = File(...),
    country_code: str = Form("NL"),
    required_fields: str = Form("invoice_number,date,amount,currency_code"),
) -> ExtractResponse:
    original_name, file_bytes = await _store_upload(file)
    llm_requested, auto_generate_requested = _runtime_ocr_options()

    try:
        normalized_required_fields = normalize_required_fields(required_fields)
    except ValueError as exception:
        log_event(
            logger,
            logging.WARNING,
            "http.extract.validation_failed",
            file_name=original_name,
            detail=str(exception),
        )
        raise HTTPException(status_code=422, detail=str(exception)) from exception

    log_event(
        logger,
        logging.INFO,
        "http.extract.received",
        file_name=original_name,
        country_code=country_code,
        required_fields=normalized_required_fields,
        llm_fallback_enabled=llm_requested,
        auto_generate_templates=auto_generate_requested,
    )

    payload = _extract_from_bytes(
        original_name,
        file_bytes,
        country_code,
        required_fields=normalized_required_fields,
        llm_fallback_enabled=llm_requested,
        auto_generate_templates=auto_generate_requested,
    )

    log_event(
        logger,
        logging.INFO,
        "http.extract.completed",
        file_name=original_name,
        status=payload.get("status"),
    )

    return payload


@app.post("/playground/result", response_class=HTMLResponse)
async def playground_result(
    file: UploadFile = File(...),
    country_code: str = Form("NL"),
    required_fields: str = Form("invoice_number,date,amount,currency_code"),
) -> HTMLResponse:
    original_name, file_bytes = await _store_upload(file)
    llm_requested, auto_generate_requested = _runtime_ocr_options()

    try:
        normalized_required_fields = normalize_required_fields(required_fields)
    except ValueError as exception:
        log_event(
            logger,
            logging.WARNING,
            "http.playground.validation_failed",
            file_name=original_name,
            detail=str(exception),
        )
        raise HTTPException(status_code=422, detail=str(exception)) from exception

    log_event(
        logger,
        logging.INFO,
        "http.playground.received",
        file_name=original_name,
        country_code=country_code,
        required_fields=normalized_required_fields,
        llm_fallback_enabled=llm_requested,
        auto_generate_templates=auto_generate_requested,
    )

    payload = _extract_from_bytes(
        original_name,
        file_bytes,
        country_code,
        required_fields=normalized_required_fields,
        llm_fallback_enabled=llm_requested,
        auto_generate_templates=auto_generate_requested,
    )

    log_event(
        logger,
        logging.INFO,
        "http.playground.completed",
        file_name=original_name,
        status=payload.get("status"),
    )

    return _render_json_result_page("OCR JSON Result", payload)


@app.post("/template-generator/result", response_class=HTMLResponse)
async def template_generator_result(
    file: UploadFile = File(...),
    issuer: str = Form(...),
    invoice_number_label: str = Form(""),
    date_label: str = Form(""),
    amount_label: str = Form(""),
    country_code: str = Form("NL"),
    currency_code: str = Form("EUR"),
    currency_label: str = Form(""),
    keywords: str = Form(""),
) -> HTMLResponse:
    original_name, file_bytes = await _store_upload(file)
    keyword_lines = tuple(
        keyword.strip()
        for keyword in keywords.splitlines()
        if keyword.strip()
    )

    log_event(
        logger,
        logging.INFO,
        "http.template_generator.received",
        file_name=original_name,
        issuer=issuer.strip(),
        country_code=country_code.strip().upper() or "NL",
        keyword_count=len(keyword_lines),
    )

    with TemporaryDirectory() as temporary_directory:
        input_path = Path(temporary_directory) / original_name
        input_path.write_bytes(file_bytes)
        result = generate_starter_template_from_sample(
            sample_path=input_path,
            template_dir=extractor.writable_template_dir,
            extractor=extractor,
            spec=TemplateSpec(
                issuer=issuer.strip(),
                invoice_number_label=invoice_number_label.strip(),
                date_label=date_label.strip(),
                amount_label=amount_label.strip(),
                country_code=country_code.strip().upper() or "NL",
                currency_code=currency_code.strip().upper() or "EUR",
                currency_label=currency_label.strip() or None,
                keywords=keyword_lines,
            ),
        )

    log_event(
        logger,
        logging.INFO,
        "http.template_generator.completed",
        file_name=original_name,
        output_path=result.output_path,
        missing_labels=result.missing_labels,
    )

    notices = [
        f"<div class=\"notice\"><strong>Saved:</strong> {html.escape(str(result.output_path))}</div>",
    ]

    if result.missing_labels:
        notices.append(
            "<div class=\"notice\"><strong>Labels not found in OCR text:</strong> "
            + html.escape(", ".join(result.missing_labels))
            + "</div>"
        )

    document_preview = ""
    document_row_markup = "".join(
        f"<tr><td>{html.escape(row.label)}</td><td>{html.escape(row.value)}</td>"
        f"<td><span class=\"status-chip status-{html.escape(row.status)}\">"
        f"{html.escape(row.status)}</span></td></tr>"
        for row in result.document_rows
    )

    if document_row_markup:
        document_preview = f"""
              <h2>Document Preview</h2>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Field</th>
                      <th>Value</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>{document_row_markup}</tbody>
                </table>
              </div>
        """

    line_items_preview = ""

    if result.line_item_rows:
        # Headers are dynamic — derived from whatever columns the invoice actually has
        headers = result.line_item_rows[0].headers
        header_markup = "".join(
            f"<th>{html.escape(h.replace('_', ' ').title())}</th>" for h in headers
        )
        row_markup = "".join(
            "<tr>"
            + "".join(
                f"<td>{html.escape(row.values.get(h, ''))}</td>" for h in headers
            )
            + "</tr>"
            for row in result.line_item_rows
        )
        line_items_preview = f"""
              <h2>Line Items Preview</h2>
              <div class="table-wrap">
                <table>
                  <thead><tr>{header_markup}</tr></thead>
                  <tbody>{row_markup}</tbody>
                </table>
              </div>
        """

    return HTMLResponse(
        _page_layout(
            "Starter Template Result",
            f"""
            <div class="nav">
              <a class="secondary" href="/">OCR Playground</a>
              <a href="/template-generator">Template Generator</a>
            </div>
            <section class="card">
              <h1>Starter Template Result</h1>
              <p>The template below was generated from your uploaded sample and written into the writable OCR template directory.</p>
              {''.join(notices)}
              <h2>Generated template.yml</h2>
              <pre>{html.escape(result.content)}</pre>
              {document_preview}
              {line_items_preview}
              <h2>Extracted text preview</h2>
              <pre>{html.escape(result.preview_text or 'No text was extracted from this sample.')}</pre>
            </section>
            """,
        )
    )
