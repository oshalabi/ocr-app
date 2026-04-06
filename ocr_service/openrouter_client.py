from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import logging
import mimetypes
import os
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Callable
from urllib import error, request

from ocr_service.logger import get_logger, is_debug_enabled, log_event


JSON_RESPONSE_FORMAT = "json_schema"
NO_PLUGIN_PDF_ENGINE = "native"
OPENROUTER_PDF_FALLBACK_ENGINE = "mistral-ocr"
SUPPORTED_FILE_PARSER_ENGINES = {"cloudflare-ai", "mistral-ocr", "pdf-text"}
DEFAULT_OPENROUTER_CONFIDENCE = 0.95
DEFAULT_RATE_LIMIT_RETRIES = 2
DEFAULT_RATE_LIMIT_BACKOFF_MS = 2000


@dataclass(frozen=True)
class OpenRouterRequestAttempt:
    payload: dict[str, Any]
    structured_output: bool
    pdf_engine: str | None


@dataclass(frozen=True)
class OpenRouterRequestResult:
    payload: dict[str, Any] | None
    reason: str | None = None
    status_code: int | None = None
    error_body: str | None = None


class OpenRouterClient:
    def __init__(self) -> None:
        self.logger = get_logger("ocr_service.openrouter")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.model = os.getenv("OPENROUTER_MODEL", "")
        self.template_model = os.getenv("OPENROUTER_TEMPLATE_MODEL") or self.model
        self.timeout = float(os.getenv("OPENROUTER_TIMEOUT", "30"))
        self.require_zdr = self._env_flag("OPENROUTER_REQUIRE_ZDR", default=False)
        self.disable_reasoning = self._env_flag("OPENROUTER_DISABLE_REASONING", default=False)
        self.http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "InvoiceShelf OCR")
        self.pdf_engine = os.getenv("OPENROUTER_PDF_ENGINE", NO_PLUGIN_PDF_ENGINE).strip().lower()
        self.rate_limit_retries = self._env_int("OPENROUTER_RATE_LIMIT_RETRIES", DEFAULT_RATE_LIMIT_RETRIES)
        self.rate_limit_backoff_ms = self._env_int("OPENROUTER_RATE_LIMIT_BACKOFF_MS", DEFAULT_RATE_LIMIT_BACKOFF_MS)

        log_event(
            self.logger,
            logging.DEBUG,
            "openrouter.config.loaded",
            base_url=self.base_url,
            model_configured=bool(self.model),
            template_model_configured=bool(self.template_model),
            require_zdr=self.require_zdr,
            disable_reasoning=self.disable_reasoning,
            pdf_engine=self.pdf_engine,
            rate_limit_retries=self.rate_limit_retries,
            rate_limit_backoff_ms=self.rate_limit_backoff_ms,
        )

    def is_configured(self) -> bool:
        return bool(self.api_key and self.model)

    def extract_fields(
        self,
        file_path: Path,
        country_code: str,
        required_fields: tuple[str, ...],
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.is_configured():
            log_event(
                self.logger,
                logging.INFO,
                "openrouter.extract.skipped",
                file_name=file_path.name,
                reason="client_not_configured",
            )
            return None

        log_event(
            self.logger,
            logging.INFO,
            "openrouter.extract.started",
            file_name=file_path.name,
            country_code=country_code,
            required_fields=required_fields,
            model=self.model,
            session_id=session_id,
        )

        base_payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._extraction_prompt(country_code, required_fields, structured_output=True),
                        },
                        self._file_content_part(file_path),
                    ],
                }
            ],
        }
        response_payload = self._request_with_fallbacks(
            base_payload=base_payload,
            file_path=file_path,
            schema_name="invoice_field_extraction",
            schema=self._extraction_schema(required_fields),
            prompt_factory=lambda structured_output: self._extraction_prompt(
                country_code,
                required_fields,
                structured_output=structured_output,
                careful_scan=self._is_image_upload(file_path),
            ),
            action="openrouter.extract",
            session_id=session_id,
        )

        if response_payload is None:
            log_event(
                self.logger,
                logging.WARNING,
                "openrouter.extract.failed",
                file_name=file_path.name,
                reason="request_failed",
            )
            return None

        content = self._extract_response_content(response_payload)

        if content is None:
            log_event(
                self.logger,
                logging.WARNING,
                "openrouter.extract.failed",
                file_name=file_path.name,
                reason="missing_response_content",
            )
            return None

        parsed_content = self._parse_json_content(content)

        if not isinstance(parsed_content, dict):
            log_event(
                self.logger,
                logging.WARNING,
                "openrouter.extract.failed",
                file_name=file_path.name,
                reason="invalid_json_content",
            )
            return None

        normalized_payload = self._normalize_extraction_payload(parsed_content, required_fields)

        if normalized_payload is None:
            log_event(
                self.logger,
                logging.WARNING,
                "openrouter.extract.failed",
                file_name=file_path.name,
                reason="invalid_field_payload",
                parsed_content=parsed_content if is_debug_enabled() else None,
            )
            return None

        values, confidences, issuer, defaulted_confidence_fields = normalized_payload

        log_event(
            self.logger,
            logging.INFO,
            "openrouter.extract.completed",
            file_name=file_path.name,
            field_names=sorted(values.keys()),
            model=self.model,
            defaulted_confidence_fields=defaulted_confidence_fields,
        )

        return {
            "fields": values,
            "confidence": confidences,
            "issuer": issuer if isinstance(issuer, str) else None,
            "model": self.model,
        }

    def generate_template_definition(
        self,
        file_path: Path,
        country_code: str,
        required_fields: tuple[str, ...],
        correction_context: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.api_key or not self.template_model:
            log_event(
                self.logger,
                logging.INFO,
                "openrouter.template_generation.skipped",
                file_name=file_path.name,
                reason="client_not_configured",
            )
            return None

        log_event(
            self.logger,
            logging.INFO,
            "openrouter.template_generation.started",
            file_name=file_path.name,
            country_code=country_code,
            required_fields=required_fields,
            model=self.template_model,
            correction_pass=bool(correction_context),
            session_id=session_id,
        )

        base_payload = {
            "model": self.template_model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._template_prompt(
                                country_code,
                                required_fields,
                                structured_output=True,
                                correction_context=correction_context,
                            ),
                        },
                        self._file_content_part(file_path),
                    ],
                }
            ],
        }
        response_payload = self._request_with_fallbacks(
            base_payload=base_payload,
            file_path=file_path,
            schema_name="invoice_template_generation",
            schema=self._template_schema(required_fields),
            prompt_factory=lambda structured_output: self._template_prompt(
                country_code,
                required_fields,
                structured_output=structured_output,
                correction_context=correction_context,
                careful_scan=self._is_image_upload(file_path),
            ),
            action="openrouter.template_generation",
            session_id=session_id,
        )

        if response_payload is None:
            log_event(
                self.logger,
                logging.WARNING,
                "openrouter.template_generation.failed",
                file_name=file_path.name,
                reason="request_failed",
            )
            return None

        content = self._extract_response_content(response_payload)

        if content is None:
            log_event(
                self.logger,
                logging.WARNING,
                "openrouter.template_generation.failed",
                file_name=file_path.name,
                reason="missing_response_content",
            )
            return None

        parsed_content = self._parse_json_content(content)

        if not isinstance(parsed_content, dict):
            log_event(
                self.logger,
                logging.WARNING,
                "openrouter.template_generation.failed",
                file_name=file_path.name,
                reason="invalid_json_content",
            )
            return None

        log_event(
            self.logger,
            logging.INFO,
            "openrouter.template_generation.completed",
            file_name=file_path.name,
            keyword_count=len(parsed_content.get("keywords", [])) if isinstance(parsed_content.get("keywords"), list) else 0,
            model=self.template_model,
        )

        return parsed_content

    def _request_with_fallbacks(
        self,
        *,
        base_payload: dict[str, Any],
        file_path: Path,
        schema_name: str,
        schema: dict[str, Any],
        prompt_factory: Callable[[bool], str],
        action: str,
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        last_result: OpenRouterRequestResult | None = None
        attempts = self._build_request_attempts(
            base_payload=base_payload,
            file_path=file_path,
            schema_name=schema_name,
            schema=schema,
            prompt_factory=prompt_factory,
            session_id=session_id,
        )

        for index, attempt in enumerate(attempts, start=1):
            log_event(
                self.logger,
                logging.DEBUG,
                f"{action}.attempt",
                file_name=file_path.name,
                attempt=index,
                attempt_count=len(attempts),
                structured_output=attempt.structured_output,
                pdf_engine=attempt.pdf_engine or "none",
                session_id=session_id,
            )

            result = self._request(attempt.payload)

            if result.payload is not None:
                return result.payload

            last_result = result

            if not self._should_retry_request(result):
                break

        if last_result:
            log_event(
                self.logger,
                logging.DEBUG,
                f"{action}.attempts_exhausted",
                file_name=file_path.name,
                status_code=last_result.status_code,
                reason=last_result.reason,
                session_id=session_id,
            )

        return None

    def _request(self, payload: dict[str, Any]) -> OpenRouterRequestResult:
        max_attempts = max(1, self.rate_limit_retries + 1)
        session_id = payload.get("session_id") if isinstance(payload.get("session_id"), str) else None

        for request_attempt in range(1, max_attempts + 1):
            started_at = perf_counter()
            request_payload = json.dumps(payload).encode("utf-8")
            http_request = request.Request(
                self.base_url,
                data=request_payload,
                method="POST",
                headers=self._headers(),
            )

            try:
                with request.urlopen(http_request, timeout=self.timeout) as response:
                    status_code = getattr(response, "status", response.getcode())
                    raw_payload = response.read().decode("utf-8")
            except error.HTTPError as exception:
                error_body = self._sanitize_error_body(exception.read().decode("utf-8", errors="replace"))

                if exception.code == 429 and request_attempt < max_attempts:
                    backoff_ms = self.rate_limit_backoff_ms * (2 ** (request_attempt - 1))
                    log_event(
                        self.logger,
                        logging.WARNING,
                        "openrouter.request.retrying",
                        reason="rate_limited",
                        status_code=exception.code,
                        request_attempt=request_attempt,
                        max_attempts=max_attempts,
                        backoff_ms=backoff_ms,
                        duration_ms=round((perf_counter() - started_at) * 1000, 2),
                        error_body=error_body,
                        session_id=session_id,
                    )
                    sleep(backoff_ms / 1000)
                    continue

                log_event(
                    self.logger,
                    logging.WARNING,
                    "openrouter.request.failed",
                    status_code=exception.code,
                    reason="http_error",
                    duration_ms=round((perf_counter() - started_at) * 1000, 2),
                    error_body=error_body,
                    request_attempt=request_attempt,
                    session_id=session_id,
                )
                return OpenRouterRequestResult(
                    payload=None,
                    reason="http_error",
                    status_code=exception.code,
                    error_body=error_body,
                )
            except error.URLError:
                log_event(
                    self.logger,
                    logging.WARNING,
                    "openrouter.request.failed",
                    reason="url_error",
                    duration_ms=round((perf_counter() - started_at) * 1000, 2),
                    request_attempt=request_attempt,
                    session_id=session_id,
                )
                return OpenRouterRequestResult(
                    payload=None,
                    reason="url_error",
                )
            except TimeoutError:
                log_event(
                    self.logger,
                    logging.WARNING,
                    "openrouter.request.failed",
                    reason="timeout",
                    duration_ms=round((perf_counter() - started_at) * 1000, 2),
                    request_attempt=request_attempt,
                    session_id=session_id,
                )
                return OpenRouterRequestResult(
                    payload=None,
                    reason="timeout",
                )

            log_event(
                self.logger,
                logging.INFO,
                "openrouter.request.completed",
                status_code=status_code,
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
                request_attempt=request_attempt,
                session_id=session_id,
            )

            try:
                parsed_payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                log_event(
                    self.logger,
                    logging.WARNING,
                    "openrouter.request.failed",
                    reason="invalid_json_response",
                    request_attempt=request_attempt,
                    session_id=session_id,
                )
                return OpenRouterRequestResult(
                    payload=None,
                    reason="invalid_json_response",
                )

            if not isinstance(parsed_payload, dict):
                log_event(
                    self.logger,
                    logging.WARNING,
                    "openrouter.request.failed",
                    reason="non_object_response",
                    request_attempt=request_attempt,
                    session_id=session_id,
                )
                return OpenRouterRequestResult(
                    payload=None,
                    reason="non_object_response",
                )

            return OpenRouterRequestResult(
                payload=parsed_payload,
            )

        return OpenRouterRequestResult(
            payload=None,
            reason="request_exhausted",
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer

        if self.app_name:
            headers["X-Title"] = self.app_name

        return headers

    def _provider_payload(self) -> dict[str, Any]:
        if not self.require_zdr:
            return {}

        return {
            "provider": {
                "zdr": True,
            }
        }

    def _reasoning_payload(self, file_path: Path) -> dict[str, Any]:
        if self._is_image_upload(file_path):
            return {
                "reasoning": {
                    "effort": "medium",
                }
            }

        if not self.disable_reasoning:
            return {}

        return {
            "reasoning": {
                "effort": "none",
                "exclude": True,
            }
        }

    def _file_content_part(self, file_path: Path) -> dict[str, Any]:
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        data_url = self._data_url(file_path, mime_type)

        if file_path.suffix.lower() == ".pdf":
            return {
                "type": "file",
                "file": {
                    "filename": file_path.name,
                    "file_data": data_url,
                },
            }

        return {
            "type": "image_url",
            "image_url": {
                "url": data_url,
            },
        }

    def _data_url(self, file_path: Path, mime_type: str) -> str:
        encoded_file = base64.b64encode(file_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded_file}"

    def _extract_response_content(self, payload: dict[str, Any]) -> str | None:
        choices = payload.get("choices")

        if not isinstance(choices, list) or not choices:
            return None

        first_choice = choices[0]

        if not isinstance(first_choice, dict):
            return None

        message = first_choice.get("message")

        if not isinstance(message, dict):
            return None

        content = message.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []

            for item in content:
                if not isinstance(item, dict):
                    continue

                text = item.get("text")

                if isinstance(text, str):
                    parts.append(text)

            if parts:
                return "".join(parts)

        return None

    def _extraction_prompt(
        self,
        country_code: str,
        required_fields: tuple[str, ...],
        *,
        structured_output: bool,
        careful_scan: bool = False,
    ) -> str:
        prompt = (
            "Extract invoice data from this document.\n"
            f"Country code: {country_code}.\n"
            "Return only JSON. Do not include markdown fences, comments, or explanatory text.\n"
            "Use null when a field is not clearly visible.\n"
            "Rules:\n"
            "- invoice_number is the invoice or tax-invoice identifier, not order or delivery number.\n"
            "- date is the invoice issue date, not order, due, or delivery date.\n"
            "- amount is the final gross amount payable, inclusive of tax when present.\n"
            "- currency_code must be an uppercase ISO-4217 code.\n"
            f"Required fields: {', '.join(required_fields)}.\n"
            "Also include issuer when clearly visible."
        )

        if careful_scan:
            prompt = (
                f"{prompt}\n"
                "This input is an image or scan. Read it carefully line by line.\n"
                "OCR may be noisy, labels and values may be split across lines, and some text may be faint or rotated.\n"
                "Prefer careful interpretation over speed and do not guess when a field is unclear."
            )

        if structured_output:
            return f"{prompt}\nReturn JSON matching the provided schema."

        return (
            f"{prompt}\n"
            "Return exactly this JSON shape:\n"
            f"{json.dumps(self._plain_extraction_shape(required_fields), ensure_ascii=False)}"
        )

    def _template_prompt(
        self,
        country_code: str,
        required_fields: tuple[str, ...],
        *,
        structured_output: bool,
        correction_context: str | None = None,
        careful_scan: bool = False,
    ) -> str:
        prompt = (
            "Generate an invoice2data-compatible starter template definition for this invoice.\n"
            f"Country code: {country_code}.\n"
            "Return only JSON. Do not include markdown fences, comments, or explanatory text.\n"
            "Template rules:\n"
            "- Generate 1 to 3 stable supplier-level keywords.\n"
            "- Do not use invoice numbers, dates, order numbers, delivery numbers, or totals as keywords.\n"
            "- Prefer issuer, VAT number, KvK number, domain names, or stable company identifiers.\n"
            "- VAT numbers such as NL123456789B01 and KvK numbers are allowed when they are supplier identifiers.\n"
            "- If you are unsure about a keyword, omit it instead of overfitting to this invoice.\n"
            "- Keywords must be reusable across multiple invoices from the same supplier.\n"
            "- fields must only target the required invoice fields.\n"
            "- Each fields[field_name] value must be a plain regex string, not an object.\n"
            "- Do not return nested objects such as {\"regex\": \"...\"} inside fields.\n"
            "- Use regex patterns suitable for invoice2data and capture the field value in the first capture group.\n"
            "- options.remove_whitespace must be true.\n"
            f"Required fields: {', '.join(required_fields)}."
        )

        if careful_scan:
            prompt = (
                f"{prompt}\n"
                "This input is an image or scan. Read it carefully and assume OCR noise is possible.\n"
                "Design regexes that tolerate multiline label/value layouts, optional punctuation, and inconsistent spacing."
            )

        if correction_context:
            prompt = f"{prompt}\n\nCorrection context:\n{correction_context}"

        if structured_output:
            return f"{prompt}\nReturn JSON matching the provided schema."

        return (
            f"{prompt}\n"
            "Return exactly this JSON shape:\n"
            f"{json.dumps(self._plain_template_shape(required_fields), ensure_ascii=False)}"
        )

    def _extraction_schema(self, required_fields: tuple[str, ...]) -> dict[str, Any]:
        field_properties = {
            field_name: {"type": ["string", "number", "null"]}
            for field_name in required_fields
        }
        confidence_properties = {
            field_name: {"type": "number", "minimum": 0, "maximum": 1}
            for field_name in required_fields
        }

        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "issuer": {"type": ["string", "null"]},
                "fields": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": field_properties,
                    "required": list(required_fields),
                },
                "confidence": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": confidence_properties,
                    "required": list(required_fields),
                },
            },
            "required": ["fields", "confidence", "issuer"],
        }

    def _template_schema(self, required_fields: tuple[str, ...]) -> dict[str, Any]:
        field_properties = {
            field_name: {
                "anyOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "regex": {"type": "string"},
                        },
                        "required": ["regex"],
                    },
                ]
            }
            for field_name in required_fields
        }

        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "issuer": {"type": "string"},
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "fields": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": field_properties,
                    "required": list(required_fields),
                },
                "options": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "date_formats": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                        "remove_whitespace": {"type": "boolean"},
                    },
                    "required": ["date_formats", "remove_whitespace"],
                },
            },
            "required": ["issuer", "keywords", "fields", "options"],
        }

    def _env_flag(self, name: str, default: bool) -> bool:
        value = os.getenv(name)

        if value is None:
            return default

        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _env_int(self, name: str, default: int) -> int:
        value = os.getenv(name)

        if value is None:
            return default

        try:
            return max(0, int(value.strip()))
        except ValueError:
            return default

    def _build_request_attempts(
        self,
        *,
        base_payload: dict[str, Any],
        file_path: Path,
        schema_name: str,
        schema: dict[str, Any],
        prompt_factory: Callable[[bool], str],
        session_id: str | None,
    ) -> list[OpenRouterRequestAttempt]:
        attempts: list[OpenRouterRequestAttempt] = []

        for pdf_engine in self._pdf_engine_variants(file_path):
            for structured_output in (True, False):
                payload = {
                    **base_payload,
                    "messages": self._with_prompt(
                        base_payload["messages"],
                        prompt_factory(structured_output),
                    ),
                    **self._provider_payload(),
                    **self._reasoning_payload(file_path),
                }

                if structured_output:
                    payload["response_format"] = {
                        "type": JSON_RESPONSE_FORMAT,
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        },
                    }

                plugins = self._plugins_for_pdf_engine(file_path, pdf_engine)

                if plugins:
                    payload["plugins"] = plugins

                if session_id:
                    payload["session_id"] = session_id

                attempts.append(
                    OpenRouterRequestAttempt(
                        payload=payload,
                        structured_output=structured_output,
                        pdf_engine=pdf_engine,
                    )
                )

        return attempts

    def _with_prompt(self, messages: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
        updated_messages = json.loads(json.dumps(messages))
        updated_messages[0]["content"][0]["text"] = prompt
        return updated_messages

    def _pdf_engine_variants(self, file_path: Path) -> list[str | None]:
        if file_path.suffix.lower() != ".pdf":
            return [None]

        variants: list[str | None] = []
        normalized_engine = self.pdf_engine

        if normalized_engine in {"", NO_PLUGIN_PDF_ENGINE, "auto"}:
            variants.append(None)
        elif normalized_engine in SUPPORTED_FILE_PARSER_ENGINES:
            variants.append(normalized_engine)
            variants.append(None)
        else:
            log_event(
                self.logger,
                logging.WARNING,
                "openrouter.config.invalid_pdf_engine",
                configured_pdf_engine=normalized_engine,
                fallback_pdf_engine=OPENROUTER_PDF_FALLBACK_ENGINE,
            )
            variants.append(None)

        if OPENROUTER_PDF_FALLBACK_ENGINE not in variants:
            variants.append(OPENROUTER_PDF_FALLBACK_ENGINE)

        deduplicated_variants: list[str | None] = []

        for variant in variants:
            if variant not in deduplicated_variants:
                deduplicated_variants.append(variant)

        return deduplicated_variants

    def _plugins_for_pdf_engine(self, file_path: Path, pdf_engine: str | None) -> list[dict[str, Any]]:
        if file_path.suffix.lower() != ".pdf" or pdf_engine is None:
            return []

        return [
            {
                "id": "file-parser",
                "pdf": {
                    "engine": pdf_engine,
                },
            }
        ]

    def _should_retry_request(self, result: OpenRouterRequestResult) -> bool:
        return result.reason == "http_error" and result.status_code == 400

    def _normalize_extraction_payload(
        self,
        parsed_content: dict[str, Any],
        required_fields: tuple[str, ...],
    ) -> tuple[dict[str, Any], dict[str, float], str | None, list[str]] | None:
        issuer = parsed_content.get("issuer")
        raw_fields = parsed_content.get("fields")
        raw_confidence = parsed_content.get("confidence")

        if not isinstance(raw_fields, dict):
            raw_fields = {
                field_name: parsed_content.get(field_name)
                for field_name in required_fields
                if field_name in parsed_content
            }

        if not isinstance(raw_confidence, dict):
            raw_confidence = {}

        normalized_fields: dict[str, Any] = {}
        normalized_confidence: dict[str, float] = {}
        defaulted_confidence_fields: list[str] = []

        for field_name in required_fields:
            field_value = raw_fields.get(field_name)
            field_confidence = raw_confidence.get(field_name)

            if isinstance(field_value, dict):
                field_confidence = field_value.get("confidence", field_confidence)
                field_value = field_value.get("value")

            if field_value is None:
                continue

            normalized_fields[field_name] = field_value

            if isinstance(field_confidence, (int, float)):
                normalized_confidence[field_name] = float(field_confidence)
            else:
                normalized_confidence[field_name] = DEFAULT_OPENROUTER_CONFIDENCE
                defaulted_confidence_fields.append(field_name)

        if not normalized_fields:
            return None

        return (
            normalized_fields,
            normalized_confidence,
            issuer if isinstance(issuer, str) and issuer.strip() else None,
            defaulted_confidence_fields,
        )

    def _parse_json_content(self, content: str) -> dict[str, Any] | None:
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError:
            extracted_payload = self._extract_json_object(content)

            if extracted_payload is None:
                return None

            try:
                parsed_content = json.loads(extracted_payload)
            except json.JSONDecodeError:
                return None

        if not isinstance(parsed_content, dict):
            return None

        return parsed_content

    def _extract_json_object(self, content: str) -> str | None:
        trimmed_content = content.strip()

        if trimmed_content.startswith("```"):
            lines = trimmed_content.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]

            trimmed_content = "\n".join(lines).strip()

        decoder = json.JSONDecoder()

        for index, character in enumerate(trimmed_content):
            if character != "{":
                continue

            try:
                _, end_index = decoder.raw_decode(trimmed_content[index:])
            except json.JSONDecodeError:
                continue

            return trimmed_content[index:index + end_index]

        return None

    def _sanitize_error_body(self, raw_error_body: str) -> str | None:
        if not raw_error_body:
            return None

        sanitized_body = raw_error_body.strip()

        try:
            parsed_body = json.loads(sanitized_body)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed_body, dict):
                message = parsed_body.get("error", parsed_body)

                if isinstance(message, dict):
                    for key in ("message", "detail", "metadata"):
                        value = message.get(key)

                        if isinstance(value, str):
                            sanitized_body = value
                            break
                elif isinstance(message, str):
                    sanitized_body = message

        sanitized_body = sanitized_body.replace("\r", " ").replace("\n", " ").strip()

        if len(sanitized_body) > 500:
            return f"{sanitized_body[:497]}..."

        return sanitized_body

    def _plain_extraction_shape(self, required_fields: tuple[str, ...]) -> dict[str, Any]:
        return {
            "issuer": "string or null",
            "fields": {
                field_name: "string, number, or null"
                for field_name in required_fields
            },
            "confidence": {
                field_name: 0.0
                for field_name in required_fields
            },
        }

    def _plain_template_shape(self, required_fields: tuple[str, ...]) -> dict[str, Any]:
        return {
            "issuer": "string",
            "keywords": ["stable supplier regex"],
            "fields": {
                field_name: "regex with first capture group"
                for field_name in required_fields
            },
            "options": {
                "date_formats": ["%d-%m-%Y"],
                "remove_whitespace": True,
            },
        }

    def _is_image_upload(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
