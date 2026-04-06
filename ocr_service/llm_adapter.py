"""Provider-agnostic LLM adapter contract, NullAdapter, and factory.

The orchestrator depends only on LlmAdapter.  Concrete implementations
(OpenRouterAdapter, OllamaAdapter) live in their own modules and are never
imported directly by the orchestrator.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


TRUTHY_VALUES = {"1", "true", "yes", "on"}

# Env vars that control provider selection
OCR_LLM_PROVIDER_ENV = "OCR_LLM_PROVIDER"
OCR_OPENROUTER_FALLBACK_ENV = "OCR_OPENROUTER_FALLBACK"


@runtime_checkable
class LlmAdapter(Protocol):
    """Minimal contract every LLM provider adapter must satisfy."""

    @property
    def provider_name(self) -> str:
        """Short identifier shown in logs and UI, e.g. 'openrouter' or 'ollama'."""
        ...

    def is_configured(self) -> bool:
        """Return True when the adapter has enough config to make requests."""
        ...

    def extract_fields(
        self,
        file_path: Path,
        country_code: str,
        required_fields: tuple[str, ...],
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Extract invoice fields from *file_path*.

        Returns a normalised dict with keys ``fields``, ``confidence``,
        ``issuer``, ``model``, or *None* on failure.
        """
        ...

    def generate_template_definition(
        self,
        file_path: Path,
        country_code: str,
        required_fields: tuple[str, ...],
        correction_context: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return a raw template definition dict (issuer, keywords, fields) or *None*."""
        ...


class NullLlmAdapter:
    """Disabled adapter — used when no LLM provider is configured or enabled."""

    @property
    def provider_name(self) -> str:
        return "none"

    def is_configured(self) -> bool:
        return False

    def extract_fields(
        self,
        file_path: Path,  # noqa: ARG002
        country_code: str,  # noqa: ARG002
        required_fields: tuple[str, ...],  # noqa: ARG002
        session_id: str | None = None,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return None

    def generate_template_definition(
        self,
        file_path: Path,  # noqa: ARG002
        country_code: str,  # noqa: ARG002
        required_fields: tuple[str, ...],  # noqa: ARG002
        correction_context: str | None = None,  # noqa: ARG002
        session_id: str | None = None,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUTHY_VALUES


def resolve_llm_provider() -> str:
    """Return the canonical provider name from env.

    Resolution order:
    1. ``OCR_LLM_PROVIDER`` — explicit value (``none``, ``openrouter``, ``ollama``).
    2. Legacy ``OCR_OPENROUTER_FALLBACK=true`` → ``openrouter``.
    3. Default → ``none``.
    """
    explicit = os.getenv(OCR_LLM_PROVIDER_ENV, "").strip().lower()

    if explicit in {"none", "openrouter", "ollama"}:
        return explicit

    if _env_flag(OCR_OPENROUTER_FALLBACK_ENV):
        return "openrouter"

    return "none"


def build_llm_adapter() -> LlmAdapter:
    """Instantiate and return the active LLM adapter based on env config."""
    provider = resolve_llm_provider()

    if provider == "openrouter":
        from ocr_service.openrouter_adapter import OpenRouterAdapter  # noqa: PLC0415
        return OpenRouterAdapter()

    if provider == "ollama":
        from ocr_service.ollama_adapter import OllamaAdapter  # noqa: PLC0415
        return OllamaAdapter()

    return NullLlmAdapter()
