"""Ollama concrete LlmAdapter.

Uses Ollama's OpenAI-compatible /v1/chat/completions endpoint.
- JPG/PNG files are sent as base64 image_url content parts directly.
- PDF files are rendered page-by-page to PNG and sent as image content parts.
  Only the first OLLAMA_MAX_PDF_PAGES pages are rendered (default: 4).
- No auth by default.
- session_id is used for log correlation only; not sent to the API.
- No OpenRouter-only fields (plugins, provider, reasoning, session_id in body).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path
from time import perf_counter, sleep
from typing import Any
from urllib import error, request

from ocr_service.logger import get_logger, is_debug_enabled, log_event
from ocr_service.openrouter_client import (  # reuse shared helpers
    DEFAULT_OPENROUTER_CONFIDENCE,
    OpenRouterClient,
)


DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434/v1/chat/completions"
DEFAULT_OLLAMA_TIMEOUT = 60.0
DEFAULT_OLLAMA_MAX_PDF_PAGES = 4


class OllamaAdapter:
    """LlmAdapter that sends requests to a locally-hosted Ollama instance."""

    def __init__(self) -> None:
        self.logger = get_logger("ocr_service.ollama")
        self.base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        self.model = os.getenv("OLLAMA_MODEL", "")
        self.template_model = os.getenv("OLLAMA_TEMPLATE_MODEL") or self.model
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", str(DEFAULT_OLLAMA_TIMEOUT)))
        self.max_pdf_pages = self._env_int("OLLAMA_MAX_PDF_PAGES", DEFAULT_OLLAMA_MAX_PDF_PAGES)

        # Reuse JSON-parsing / normalisation helpers from the OpenRouter client
        # without subclassing — we delegate by composition.
        self._shared = OpenRouterClient.__new__(OpenRouterClient)
        self._shared.logger = self.logger
        self._shared.model = self.model
        self._shared.template_model = self.template_model

        log_event(
            self.logger,
            logging.DEBUG,
            "ollama.config.loaded",
            base_url=self.base_url,
            model_configured=bool(self.model),
            template_model_configured=bool(self.template_model),
            max_pdf_pages=self.max_pdf_pages,
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    def is_configured(self) -> bool:
        return bool(self.model)

    # ------------------------------------------------------------------
    # Public adapter methods
    # ------------------------------------------------------------------

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
                "ollama.extract.skipped",
                file_name=file_path.name,
                reason="adapter_not_configured",
            )
            return None

        log_event(
            self.logger,
            logging.INFO,
            "ollama.extract.started",
            file_name=file_path.name,
            country_code=country_code,
            required_fields=required_fields,
            model=self.model,
            session_id=session_id,
        )

        prompt = self._shared._extraction_prompt(  # noqa: SLF001
            country_code,
            required_fields,
            structured_output=False,
            careful_scan=self._is_image_file(file_path),
        )
        image_parts = self._image_parts_for_file(file_path, session_id=session_id)

        if not image_parts:
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.extract.failed",
                file_name=file_path.name,
                reason="no_image_parts_produced",
            )
            return None

        payload = self._build_payload(self.model, prompt, image_parts)
        response_payload = self._request(payload, session_id=session_id)

        if response_payload is None:
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.extract.failed",
                file_name=file_path.name,
                reason="request_failed",
            )
            return None

        content = self._shared._extract_response_content(response_payload)  # noqa: SLF001

        if content is None:
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.extract.failed",
                file_name=file_path.name,
                reason="missing_response_content",
            )
            return None

        parsed_content = self._shared._parse_json_content(content)  # noqa: SLF001

        if not isinstance(parsed_content, dict):
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.extract.failed",
                file_name=file_path.name,
                reason="invalid_json_content",
            )
            return None

        normalized_payload = self._shared._normalize_extraction_payload(  # noqa: SLF001
            parsed_content,
            required_fields,
        )

        if normalized_payload is None:
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.extract.failed",
                file_name=file_path.name,
                reason="invalid_field_payload",
                parsed_content=parsed_content if is_debug_enabled() else None,
            )
            return None

        values, confidences, issuer, defaulted_confidence_fields = normalized_payload

        log_event(
            self.logger,
            logging.INFO,
            "ollama.extract.completed",
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
        if not self.is_configured():
            log_event(
                self.logger,
                logging.INFO,
                "ollama.template_generation.skipped",
                file_name=file_path.name,
                reason="adapter_not_configured",
            )
            return None

        log_event(
            self.logger,
            logging.INFO,
            "ollama.template_generation.started",
            file_name=file_path.name,
            country_code=country_code,
            required_fields=required_fields,
            model=self.template_model,
            correction_pass=bool(correction_context),
            session_id=session_id,
        )

        prompt = self._shared._template_prompt(  # noqa: SLF001
            country_code,
            required_fields,
            structured_output=False,
            correction_context=correction_context,
            careful_scan=self._is_image_file(file_path),
        )
        image_parts = self._image_parts_for_file(file_path, session_id=session_id)

        if not image_parts:
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.template_generation.failed",
                file_name=file_path.name,
                reason="no_image_parts_produced",
            )
            return None

        payload = self._build_payload(self.template_model, prompt, image_parts)
        response_payload = self._request(payload, session_id=session_id)

        if response_payload is None:
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.template_generation.failed",
                file_name=file_path.name,
                reason="request_failed",
            )
            return None

        content = self._shared._extract_response_content(response_payload)  # noqa: SLF001

        if content is None:
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.template_generation.failed",
                file_name=file_path.name,
                reason="missing_response_content",
            )
            return None

        parsed_content = self._shared._parse_json_content(content)  # noqa: SLF001

        if not isinstance(parsed_content, dict):
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.template_generation.failed",
                file_name=file_path.name,
                reason="invalid_json_content",
            )
            return None

        log_event(
            self.logger,
            logging.INFO,
            "ollama.template_generation.completed",
            file_name=file_path.name,
            keyword_count=(
                len(parsed_content.get("keywords", []))
                if isinstance(parsed_content.get("keywords"), list)
                else 0
            ),
            model=self.template_model,
        )

        return parsed_content

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        model: str,
        prompt: str,
        image_parts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(image_parts)

        return {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }

    def _image_parts_for_file(
        self,
        file_path: Path,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return a list of OpenAI image_url content parts for the file.

        - JPG / PNG → single part with the file data directly.
        - PDF → render each page to PNG (up to max_pdf_pages), one part per page.
        """
        suffix = file_path.suffix.lower()

        if suffix in {".jpg", ".jpeg", ".png"}:
            return [self._image_url_part_from_bytes(file_path.read_bytes(), "image/png")]

        if suffix == ".pdf":
            return self._pdf_to_image_parts(file_path, session_id=session_id)

        return []

    def _pdf_to_image_parts(
        self,
        file_path: Path,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Render PDF pages to PNG images and return one content part per page."""
        try:
            import fitz  # PyMuPDF  # noqa: PLC0415
        except ImportError:
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.pdf_render.unavailable",
                file_name=file_path.name,
                reason="pymupdf_not_installed",
                session_id=session_id,
            )
            return []

        parts: list[dict[str, Any]] = []

        try:
            doc = fitz.open(str(file_path))
            total_pages = len(doc)
            pages_to_render = min(total_pages, self.max_pdf_pages)

            if total_pages > self.max_pdf_pages:
                log_event(
                    self.logger,
                    logging.INFO,
                    "ollama.pdf_render.truncated",
                    file_name=file_path.name,
                    total_pages=total_pages,
                    rendered_pages=pages_to_render,
                    max_pdf_pages=self.max_pdf_pages,
                    session_id=session_id,
                )

            for page_index in range(pages_to_render):
                page = doc[page_index]
                # 2× zoom for legibility
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                png_bytes = pix.tobytes("png")
                parts.append(self._image_url_part_from_bytes(png_bytes, "image/png"))

            doc.close()
        except Exception:
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.pdf_render.failed",
                file_name=file_path.name,
                session_id=session_id,
            )
            self.logger.exception("ollama.pdf_render.failed")

        return parts

    def _image_url_part_from_bytes(self, data: bytes, mime_type: str) -> dict[str, Any]:
        encoded = base64.b64encode(data).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{encoded}",
            },
        }

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    def _request(
        self,
        payload: dict[str, Any],
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        started_at = perf_counter()
        request_payload = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            self.base_url,
            data=request_payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                status_code = getattr(response, "status", getattr(response, "getcode", lambda: 0)())
                raw_payload = response.read().decode("utf-8")
        except error.HTTPError as exc:
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.request.failed",
                status_code=exc.code,
                reason="http_error",
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
                session_id=session_id,
            )
            return None
        except error.URLError:
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.request.failed",
                reason="url_error",
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
                session_id=session_id,
            )
            return None
        except TimeoutError:
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.request.failed",
                reason="timeout",
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
                session_id=session_id,
            )
            return None

        log_event(
            self.logger,
            logging.INFO,
            "ollama.request.completed",
            status_code=status_code,
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
            session_id=session_id,
        )

        try:
            parsed = json.loads(raw_payload)
        except json.JSONDecodeError:
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.request.failed",
                reason="invalid_json_response",
                session_id=session_id,
            )
            return None

        if not isinstance(parsed, dict):
            log_event(
                self.logger,
                logging.WARNING,
                "ollama.request.failed",
                reason="non_object_response",
                session_id=session_id,
            )
            return None

        return parsed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_image_file(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}

    def _env_int(self, name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return max(1, int(value.strip()))
        except ValueError:
            return default
