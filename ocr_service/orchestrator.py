from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from ocr_service.extractor import OcrExtractor
from ocr_service.fields import public_field_name
from ocr_service.llm_adapter import LlmAdapter
from ocr_service.logger import get_logger, is_debug_enabled, log_event
from ocr_service.template_generator import (
    GeneratedTemplateDefinition,
    TemplateValidationResult,
    default_ai_template_path,
    render_template_content,
    validate_ai_template_definition,
)


DEFAULT_TEMPLATE_HEALING_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class OcrProcessOptions:
    country_code: str = "NL"
    required_fields: tuple[str, ...] = ("invoice_number", "date", "amount", "currency_code")
    llm_fallback_enabled: bool = False
    auto_generate_templates: bool = False

    # ------------------------------------------------------------------ #
    # Backward-compat alias: old callers that still pass openrouter_enabled
    # as a keyword argument will hit __init__ via the field below.         #
    # ------------------------------------------------------------------ #
    openrouter_enabled: bool = False  # deprecated alias — maps to llm_fallback_enabled

    def __post_init__(self) -> None:
        # If the caller used the old name, promote it.
        if self.openrouter_enabled and not self.llm_fallback_enabled:
            object.__setattr__(self, "llm_fallback_enabled", True)


@dataclass(frozen=True)
class StoredGeneratedTemplate:
    definition: GeneratedTemplateDefinition
    output_path: Path


@dataclass(frozen=True)
class TemplateNormalizationResult:
    definition: GeneratedTemplateDefinition | None
    reason: str
    debug_template_context: str | None = None


@dataclass(frozen=True)
class TemplateStorageResult:
    stored_template: StoredGeneratedTemplate | None
    reason: str
    template_context: str | None = None
    validation_context: str | None = None


class OcrOrchestrator:
    def __init__(self, extractor: OcrExtractor, llm_adapter: LlmAdapter) -> None:
        self.extractor = extractor
        self.llm_adapter = llm_adapter
        self.logger = get_logger("ocr_service.orchestrator")

    def extract(self, file_path: Path, options: OcrProcessOptions) -> dict[str, Any]:
        started_at = perf_counter()
        llm_provider = self.llm_adapter.provider_name
        log_event(
            self.logger,
            logging.INFO,
            "orchestrator.extract.started",
            file_name=file_path.name,
            country_code=options.country_code,
            required_fields=options.required_fields,
            llm_fallback_enabled=options.llm_fallback_enabled,
            auto_generate_templates=options.auto_generate_templates,
            llm_provider=llm_provider,
        )
        llm_session_id = self._llm_session_id(file_path)

        local_response = self.extractor.extract(
            file_path,
            country_code=options.country_code,
            required_fields=options.required_fields,
        )

        log_event(
            self.logger,
            logging.INFO,
            "orchestrator.local.completed",
            file_name=file_path.name,
            status=local_response.get("status"),
        )

        if local_response["status"] == "success":
            log_event(
                self.logger,
                logging.INFO,
                "orchestrator.extract.completed",
                file_name=file_path.name,
                final_source="local",
                status=local_response.get("status"),
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
                llm_provider=llm_provider,
            )
            return local_response

        if not options.llm_fallback_enabled or not self.llm_adapter.is_configured():
            log_event(
                self.logger,
                logging.INFO,
                "orchestrator.extract.completed",
                file_name=file_path.name,
                final_source="local",
                status=local_response.get("status"),
                reason="llm_disabled_or_unconfigured",
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
                llm_provider=llm_provider,
            )
            return local_response

        llm_extraction = self.llm_adapter.extract_fields(
            file_path,
            country_code=options.country_code,
            required_fields=options.required_fields,
            session_id=llm_session_id,
        )
        llm_response = self._build_llm_response(llm_extraction, options)

        log_event(
            self.logger,
            logging.INFO,
            "orchestrator.llm_extract.completed",
            file_name=file_path.name,
            status=llm_response.get("status"),
            llm_provider=llm_provider,
        )

        if local_response["status"] == "partial":
            merged_response = self._merge_missing_required_fields(local_response, llm_response, options)
            log_event(
                self.logger,
                logging.INFO,
                "orchestrator.extract.completed",
                file_name=file_path.name,
                final_source="merged",
                status=merged_response.get("status"),
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
                llm_provider=llm_provider,
            )
            return merged_response

        if local_response["status"] == "failed" and options.auto_generate_templates:
            healed_response = self._heal_failed_local_extraction(
                file_path=file_path,
                options=options,
                local_response=local_response,
                llm_response=llm_response,
                llm_extraction=llm_extraction,
                llm_session_id=llm_session_id,
            )

            if healed_response is not None:
                log_event(
                    self.logger,
                    logging.INFO,
                    "orchestrator.extract.completed",
                    file_name=file_path.name,
                    final_source="generated_template_retry",
                    status=healed_response.get("status"),
                    duration_ms=round((perf_counter() - started_at) * 1000, 2),
                    llm_provider=llm_provider,
                )
                return healed_response

        if llm_response["status"] != "failed":
            log_event(
                self.logger,
                logging.INFO,
                "orchestrator.extract.completed",
                file_name=file_path.name,
                final_source="llm",
                status=llm_response.get("status"),
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
                llm_provider=llm_provider,
            )
            return llm_response

        log_event(
            self.logger,
            logging.WARNING,
            "orchestrator.extract.completed",
            file_name=file_path.name,
            final_source="local",
            status=local_response.get("status"),
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
            llm_provider=llm_provider,
        )
        return local_response

    def _build_llm_response(
        self,
        llm_extraction: dict[str, Any] | None,
        options: OcrProcessOptions,
    ) -> dict[str, Any]:
        if not llm_extraction:
            return self.extractor._failed_response(  # noqa: SLF001
                options.country_code,
                options.required_fields,
            )

        payload = dict(llm_extraction.get("fields", {}))
        issuer = llm_extraction.get("issuer")

        if issuer:
            payload["issuer"] = issuer

        response = self.extractor._build_response(  # noqa: SLF001
            payload=payload,
            source_reader=self.llm_adapter.provider_name,
            country_code=options.country_code,
            required_fields=options.required_fields,
            field_confidences=llm_extraction.get("confidence", {}),
        )

        model = llm_extraction.get("model")

        if model:
            response["unmapped_fields"]["llm_model"] = {
                "value": model,
                "confidence": 1.0,
            }

        return response

    def _merge_missing_required_fields(
        self,
        local_response: dict[str, Any],
        llm_response: dict[str, Any],
        options: OcrProcessOptions,
    ) -> dict[str, Any]:
        merged_fields = dict(local_response.get("fields", {}))

        for field_name in options.required_fields:
            public_name = public_field_name(field_name)

            if public_name in merged_fields:
                continue

            llm_field = llm_response.get("fields", {}).get(public_name)

            if llm_field:
                merged_fields[public_name] = llm_field

        merged_unmapped_fields = dict(local_response.get("unmapped_fields", {}))

        for key, value in llm_response.get("unmapped_fields", {}).items():
            if key == "source_reader":
                merged_unmapped_fields["fallback_source_reader"] = value
                continue

            if key == "missing_required_fields":
                continue

            merged_unmapped_fields.setdefault(key, value)

        missing_required_fields = [
            public_field_name(field_name)
            for field_name in options.required_fields
            if public_field_name(field_name) not in merged_fields
        ]

        if missing_required_fields:
            merged_status = "partial"
            merged_message = "Invoice matched a template, but some required fields are missing."
            merged_unmapped_fields["missing_required_fields"] = {
                "value": missing_required_fields,
                "confidence": 1.0,
            }
        else:
            merged_status = "success"
            merged_message = "Invoice fields extracted successfully."
            merged_unmapped_fields.pop("missing_required_fields", None)

        return {
            "status": merged_status,
            "message": merged_message,
            "fields": merged_fields,
            "unmapped_fields": merged_unmapped_fields,
        }

    def _heal_failed_local_extraction(
        self,
        *,
        file_path: Path,
        options: OcrProcessOptions,
        local_response: dict[str, Any],
        llm_response: dict[str, Any],
        llm_extraction: dict[str, Any] | None,
        llm_session_id: str,
    ) -> dict[str, Any] | None:
        max_attempts = self._template_healing_max_attempts()
        current_output_path: Path | None = None
        latest_retry_response = local_response
        latest_template_content: str | None = None
        latest_validation_context: str | None = None

        log_event(
            self.logger,
            logging.INFO,
            "orchestrator.template_healing.started",
            file_name=file_path.name,
            max_attempts=max_attempts,
            initial_status=local_response.get("status"),
            llm_session_id=llm_session_id,
        )

        for attempt in range(1, max_attempts + 1):
            missing_required_fields = self._missing_required_public_fields(latest_retry_response, options)

            log_event(
                self.logger,
                logging.INFO,
                "orchestrator.template_healing.iteration.started",
                file_name=file_path.name,
                attempt=attempt,
                max_attempts=max_attempts,
                prior_status=latest_retry_response.get("status"),
                missing_required_fields=missing_required_fields,
                output_path=current_output_path,
                llm_session_id=llm_session_id,
            )

            generated_template = self.llm_adapter.generate_template_definition(
                file_path,
                country_code=options.country_code,
                required_fields=options.required_fields,
                correction_context=self._template_generation_context(
                    attempt=attempt,
                    retry_response=latest_retry_response,
                    llm_response=llm_response,
                    llm_extraction=llm_extraction,
                    current_template_content=latest_template_content,
                    previous_validation_context=latest_validation_context,
                ),
                session_id=llm_session_id,
            )

            storage_result = self._store_generated_template(
                generated_template=generated_template,
                sample_path=file_path,
                options=options,
                output_path=current_output_path,
                attempt=attempt,
                llm_extraction=llm_extraction,
            )

            if storage_result.template_context:
                latest_template_content = storage_result.template_context

            if storage_result.validation_context:
                latest_validation_context = storage_result.validation_context

            if storage_result.stored_template is None:
                continue

            current_output_path = storage_result.stored_template.output_path
            retry_response = self.extractor.extract(
                file_path,
                country_code=options.country_code,
                required_fields=options.required_fields,
            )
            retry_missing_fields = self._missing_required_public_fields(retry_response, options)

            log_event(
                self.logger,
                logging.INFO,
                "orchestrator.template_healing.iteration.retry_completed",
                file_name=file_path.name,
                attempt=attempt,
                output_path=current_output_path,
                status=retry_response.get("status"),
                missing_required_fields=retry_missing_fields,
                llm_session_id=llm_session_id,
            )

            if retry_response.get("status") == "success" and not retry_missing_fields:
                log_event(
                    self.logger,
                    logging.INFO,
                    "orchestrator.template_healing.completed",
                    file_name=file_path.name,
                    attempt=attempt,
                    output_path=current_output_path,
                    status=retry_response.get("status"),
                    llm_session_id=llm_session_id,
                )
                return retry_response

            latest_retry_response = retry_response

            log_event(
                self.logger,
                logging.WARNING,
                "orchestrator.template_healing.iteration.needs_correction",
                file_name=file_path.name,
                attempt=attempt,
                output_path=current_output_path,
                status=retry_response.get("status"),
                missing_required_fields=retry_missing_fields,
                llm_session_id=llm_session_id,
            )

        log_event(
            self.logger,
            logging.WARNING,
            "orchestrator.template_healing.exhausted",
            file_name=file_path.name,
            final_status=latest_retry_response.get("status"),
            missing_required_fields=self._missing_required_public_fields(latest_retry_response, options),
            output_path=current_output_path,
            llm_session_id=llm_session_id,
        )

        return None

    def _store_generated_template(
        self,
        generated_template: dict[str, Any] | None,
        sample_path: Path,
        options: OcrProcessOptions,
        output_path: Path | None = None,
        attempt: int = 1,
        llm_extraction: dict[str, Any] | None = None,
    ) -> TemplateStorageResult:
        normalization_result = self._normalize_generated_template(
            generated_template,
            options.required_fields,
            fallback_issuer=self._fallback_issuer(llm_extraction),
        )
        debug_template_payload = generated_template if is_debug_enabled() else None

        if normalization_result.definition is None:
            log_event(
                self.logger,
                logging.WARNING,
                "orchestrator.template_generation.rejected",
                file_name=sample_path.name,
                attempt=attempt,
                reason=normalization_result.reason,
                generated_template=debug_template_payload,
            )
            return TemplateStorageResult(
                stored_template=None,
                reason=normalization_result.reason,
                template_context=normalization_result.debug_template_context,
                validation_context=normalization_result.reason,
            )

        definition = normalization_result.definition

        validation_result = validate_ai_template_definition(
            definition=definition,
            sample_path=sample_path,
            extractor=self.extractor,
            country_code=options.country_code,
            required_fields=options.required_fields,
            extracted_values=llm_extraction.get("fields") if isinstance(llm_extraction, dict) else None,
        )
        normalized_validation_result = self._normalize_validation_result(validation_result)

        if not normalized_validation_result.is_valid:
            log_event(
                self.logger,
                logging.WARNING,
                "orchestrator.template_generation.rejected",
                file_name=sample_path.name,
                attempt=attempt,
                issuer=definition.issuer,
                reason=normalized_validation_result.reason,
                missing_required_fields=normalized_validation_result.missing_required_fields,
                matched_required_fields=normalized_validation_result.matched_required_fields,
                repaired_fields=normalized_validation_result.repaired_fields,
                keyword_adjustments=normalized_validation_result.keyword_adjustments,
                generated_template=debug_template_payload,
            )
            return TemplateStorageResult(
                stored_template=None,
                reason=normalized_validation_result.reason,
                template_context=render_template_content(definition),
                validation_context=normalized_validation_result.to_prompt_context(),
            )

        resolved_output_path = output_path or default_ai_template_path(
            self.extractor.writable_template_dir,
            options.country_code,
            definition.issuer,
        )
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(render_template_content(definition), encoding="utf-8")

        log_event(
            self.logger,
            logging.INFO,
            "orchestrator.template_generation.saved",
            file_name=sample_path.name,
            attempt=attempt,
            issuer=definition.issuer,
            output_path=resolved_output_path,
        )

        return TemplateStorageResult(
            stored_template=StoredGeneratedTemplate(
                definition=definition,
                output_path=resolved_output_path,
            ),
            reason="saved",
            template_context=render_template_content(definition),
            validation_context=normalized_validation_result.to_prompt_context(),
        )

    def _missing_required_public_fields(
        self,
        response: dict[str, Any],
        options: OcrProcessOptions,
    ) -> list[str]:
        fields = response.get("fields", {})

        if not isinstance(fields, dict):
            return [public_field_name(field_name) for field_name in options.required_fields]

        return [
            public_field_name(field_name)
            for field_name in options.required_fields
            if public_field_name(field_name) not in fields
        ]

    def _template_generation_context(
        self,
        *,
        attempt: int,
        retry_response: dict[str, Any],
        llm_response: dict[str, Any],
        llm_extraction: dict[str, Any] | None,
        current_template_content: str | None,
        previous_validation_context: str | None,
    ) -> str | None:
        retry_missing_fields = self._response_summary(retry_response)
        llm_summary = self._response_summary(llm_response)
        raw_extraction_summary = self._llm_extraction_summary(llm_extraction)
        extracted_value_names = sorted(
            key
            for key, value in llm_response.get("fields", {}).items()
            if isinstance(value, dict) and value.get("value") is not None
        )
        template_section = current_template_content or "No previous template content available."

        if attempt == 1:
            if not extracted_value_names:
                return None

            return (
                "Known extraction result from this same invoice:\n"
                f"{raw_extraction_summary or llm_summary}\n"
                "Use these extracted values to generate a supplier-stable local template that reproduces them.\n"
                "Do not use invoice-specific numbers, dates, or totals as keywords.\n"
                "If you use VAT or KvK identifiers, only use supplier-stable values.\n"
                "Prefer fewer stable keywords over brittle invoice-specific ones.\n"
                "For each required field, make the regex tolerant to OCR spacing, optional punctuation, and multiline label/value layouts."
            )

        validation_section = (
            f"Previous template validation feedback:\n{previous_validation_context}\n"
            if previous_validation_context
            else ""
        )

        return (
            "This is a correction pass for a previously generated template.\n"
            f"Previous OCR retry result:\n{retry_missing_fields}\n"
            f"Direct LLM extraction summary:\n{raw_extraction_summary or llm_summary}\n"
            f"{validation_section}"
            "Current generated template content:\n"
            f"{template_section}\n"
            "Revise the template so the local OCR/template pipeline extracts all required fields "
            "from this same invoice. Keep the template supplier-stable and do not use invoice-specific "
            "numbers, dates, or totals as keywords. If a keyword does not appear in this invoice, remove it."
        )

    def _response_summary(self, response: dict[str, Any]) -> str:
        fields = response.get("fields", {})
        unmapped_fields = response.get("unmapped_fields", {})

        if not isinstance(fields, dict):
            fields = {}

        if not isinstance(unmapped_fields, dict):
            unmapped_fields = {}

        return json.dumps(
            {
                "status": response.get("status"),
                "message": response.get("message"),
                "fields": {
                    key: value.get("value")
                    for key, value in fields.items()
                    if isinstance(value, dict)
                },
                "unmapped_fields": {
                    key: value.get("value")
                    for key, value in unmapped_fields.items()
                    if isinstance(value, dict)
                },
            },
            ensure_ascii=False,
        )

    def _template_healing_max_attempts(self) -> int:
        raw_value = os.getenv("OCR_TEMPLATE_HEALING_MAX_ATTEMPTS", str(DEFAULT_TEMPLATE_HEALING_MAX_ATTEMPTS)).strip()

        try:
            parsed_value = int(raw_value)
        except ValueError:
            log_event(
                self.logger,
                logging.WARNING,
                "orchestrator.template_healing.invalid_max_attempts",
                raw_value=raw_value,
                fallback=DEFAULT_TEMPLATE_HEALING_MAX_ATTEMPTS,
            )
            return DEFAULT_TEMPLATE_HEALING_MAX_ATTEMPTS

        return max(1, parsed_value)

    def _normalize_generated_template(
        self,
        generated_template: dict[str, Any] | None,
        required_fields: tuple[str, ...],
        fallback_issuer: str | None = None,
    ) -> TemplateNormalizationResult:
        if not isinstance(generated_template, dict):
            return TemplateNormalizationResult(
                definition=None,
                reason="payload_not_object",
            )

        issuer = generated_template.get("issuer")
        keywords = generated_template.get("keywords")
        fields = generated_template.get("fields")

        if (not isinstance(issuer, str) or not issuer.strip()) and fallback_issuer:
            issuer = fallback_issuer

        if not isinstance(issuer, str) or not issuer.strip():
            return TemplateNormalizationResult(
                definition=None,
                reason="missing_issuer",
                debug_template_context=self._serialize_template_context(generated_template),
            )

        if not isinstance(keywords, list) or not all(isinstance(keyword, str) for keyword in keywords):
            return TemplateNormalizationResult(
                definition=None,
                reason="invalid_keywords",
                debug_template_context=self._serialize_template_context(generated_template),
            )

        if not isinstance(fields, dict):
            return TemplateNormalizationResult(
                definition=None,
                reason="missing_fields_object",
                debug_template_context=self._serialize_template_context(generated_template),
            )

        normalized_fields: dict[str, str] = {}

        for field_name in required_fields:
            field_pattern = fields.get(field_name)
            normalized_pattern = self._normalize_template_field_pattern(field_pattern)

            if normalized_pattern is None:
                return TemplateNormalizationResult(
                    definition=None,
                    reason=f"invalid_field_pattern:{field_name}",
                    debug_template_context=self._serialize_template_context(generated_template),
                )

            normalized_fields[field_name] = normalized_pattern

        normalized_keywords = tuple(keyword.strip() for keyword in keywords if keyword.strip())

        if not normalized_keywords:
            return TemplateNormalizationResult(
                definition=None,
                reason="empty_keywords",
                debug_template_context=self._serialize_template_context(generated_template),
            )

        return TemplateNormalizationResult(
            definition=GeneratedTemplateDefinition(
                issuer=issuer.strip(),
                keywords=normalized_keywords,
                fields=normalized_fields,
            ),
            reason="ok",
            debug_template_context=self._serialize_template_context(generated_template),
        )

    def _normalize_template_field_pattern(self, field_pattern: Any) -> str | None:
        if isinstance(field_pattern, str) and field_pattern.strip():
            return field_pattern.strip()

        if isinstance(field_pattern, dict):
            for key in ("regex", "pattern", "value"):
                nested_pattern = field_pattern.get(key)

                if isinstance(nested_pattern, str) and nested_pattern.strip():
                    return nested_pattern.strip()

        return None

    def _serialize_template_context(self, generated_template: Any) -> str | None:
        if generated_template is None:
            return None

        try:
            return json.dumps(generated_template, ensure_ascii=False, indent=2, sort_keys=True)
        except (TypeError, ValueError):
            return str(generated_template)

    def _llm_extraction_summary(self, llm_extraction: dict[str, Any] | None) -> str | None:
        if not isinstance(llm_extraction, dict):
            return None

        fields = llm_extraction.get("fields")
        confidence = llm_extraction.get("confidence")

        if not isinstance(fields, dict) or not fields:
            return None

        if not isinstance(confidence, dict):
            confidence = {}

        return json.dumps(
            {
                "issuer": llm_extraction.get("issuer"),
                "fields": {
                    key: value
                    for key, value in fields.items()
                    if value is not None
                },
                "confidence": {
                    key: value
                    for key, value in confidence.items()
                    if isinstance(value, (int, float))
                },
                "model": llm_extraction.get("model"),
            },
            ensure_ascii=False,
        )

    def _fallback_issuer(self, llm_extraction: dict[str, Any] | None) -> str | None:
        if not isinstance(llm_extraction, dict):
            return None

        issuer = llm_extraction.get("issuer")

        if not isinstance(issuer, str) or not issuer.strip():
            return None

        return issuer.strip()

    def _normalize_validation_result(self, validation_result: Any) -> TemplateValidationResult:
        if isinstance(validation_result, TemplateValidationResult):
            return validation_result

        if validation_result is True:
            return TemplateValidationResult(
                is_valid=True,
                reason="ok",
            )

        return TemplateValidationResult(
            is_valid=False,
            reason="validation_failed",
        )

    def _llm_session_id(self, file_path: Path) -> str:
        return f"ocr-{file_path.stem[:40]}-{uuid4().hex}"
