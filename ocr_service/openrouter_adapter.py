"""OpenRouter concrete LlmAdapter.

Wraps the existing OpenRouterClient so the orchestrator never imports a
provider-specific class directly.  All OpenRouter-specific behaviour
(ZDR, reasoning, PDF-plugin fallback, rate-limit retries, structured-output
fallback) is preserved unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ocr_service.openrouter_client import OpenRouterClient


class OpenRouterAdapter:
    """Thin adapter that delegates to OpenRouterClient."""

    def __init__(self) -> None:
        self._client = OpenRouterClient()

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def is_configured(self) -> bool:
        return self._client.is_configured()

    def extract_fields(
        self,
        file_path: Path,
        country_code: str,
        required_fields: tuple[str, ...],
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        return self._client.extract_fields(
            file_path,
            country_code=country_code,
            required_fields=required_fields,
            session_id=session_id,
        )

    def generate_template_definition(
        self,
        file_path: Path,
        country_code: str,
        required_fields: tuple[str, ...],
        correction_context: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        return self._client.generate_template_definition(
            file_path,
            country_code=country_code,
            required_fields=required_fields,
            correction_context=correction_context,
            session_id=session_id,
        )
