"""Tests for the LlmAdapter contract, factory, and runtime config resolution."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from ocr_service.llm_adapter import (
    NullLlmAdapter,
    build_llm_adapter,
    resolve_llm_provider,
)


# ---------------------------------------------------------------------------
# NullLlmAdapter
# ---------------------------------------------------------------------------

def test_null_adapter_is_not_configured() -> None:
    adapter = NullLlmAdapter()
    assert adapter.is_configured() is False
    assert adapter.provider_name == "none"


def test_null_adapter_extract_returns_none(tmp_path: Path) -> None:
    adapter = NullLlmAdapter()
    result = adapter.extract_fields(tmp_path / "invoice.pdf", "NL", ("invoice_number",))
    assert result is None


def test_null_adapter_generate_template_returns_none(tmp_path: Path) -> None:
    adapter = NullLlmAdapter()
    result = adapter.generate_template_definition(tmp_path / "invoice.pdf", "NL", ("invoice_number",))
    assert result is None


# ---------------------------------------------------------------------------
# resolve_llm_provider
# ---------------------------------------------------------------------------

def test_resolve_provider_explicit_none(monkeypatch) -> None:
    monkeypatch.setenv("OCR_LLM_PROVIDER", "none")
    assert resolve_llm_provider() == "none"


def test_resolve_provider_explicit_openrouter(monkeypatch) -> None:
    monkeypatch.setenv("OCR_LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OCR_OPENROUTER_FALLBACK", raising=False)
    assert resolve_llm_provider() == "openrouter"


def test_resolve_provider_explicit_ollama(monkeypatch) -> None:
    monkeypatch.setenv("OCR_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OCR_OPENROUTER_FALLBACK", raising=False)
    assert resolve_llm_provider() == "ollama"


def test_resolve_provider_legacy_fallback_true(monkeypatch) -> None:
    monkeypatch.delenv("OCR_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OCR_OPENROUTER_FALLBACK", "true")
    assert resolve_llm_provider() == "openrouter"


def test_resolve_provider_legacy_fallback_false(monkeypatch) -> None:
    monkeypatch.delenv("OCR_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OCR_OPENROUTER_FALLBACK", "false")
    assert resolve_llm_provider() == "none"


def test_resolve_provider_default_is_none(monkeypatch) -> None:
    monkeypatch.delenv("OCR_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OCR_OPENROUTER_FALLBACK", raising=False)
    assert resolve_llm_provider() == "none"


def test_resolve_provider_explicit_overrides_legacy(monkeypatch) -> None:
    """OCR_LLM_PROVIDER takes precedence over OCR_OPENROUTER_FALLBACK."""
    monkeypatch.setenv("OCR_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OCR_OPENROUTER_FALLBACK", "true")
    assert resolve_llm_provider() == "ollama"


def test_resolve_provider_unknown_value_falls_back_to_none(monkeypatch) -> None:
    monkeypatch.setenv("OCR_LLM_PROVIDER", "claude")
    monkeypatch.delenv("OCR_OPENROUTER_FALLBACK", raising=False)
    # Unknown values are not in the allowed set → fall through to legacy check → none
    assert resolve_llm_provider() == "none"


# ---------------------------------------------------------------------------
# build_llm_adapter factory
# ---------------------------------------------------------------------------

def test_factory_returns_null_adapter_when_provider_is_none(monkeypatch) -> None:
    monkeypatch.setenv("OCR_LLM_PROVIDER", "none")
    adapter = build_llm_adapter()
    assert adapter.provider_name == "none"
    assert isinstance(adapter, NullLlmAdapter)


def test_factory_returns_openrouter_adapter(monkeypatch) -> None:
    monkeypatch.setenv("OCR_LLM_PROVIDER", "openrouter")
    adapter = build_llm_adapter()
    assert adapter.provider_name == "openrouter"


def test_factory_returns_ollama_adapter(monkeypatch) -> None:
    monkeypatch.setenv("OCR_LLM_PROVIDER", "ollama")
    adapter = build_llm_adapter()
    assert adapter.provider_name == "ollama"


def test_factory_uses_legacy_fallback_to_build_openrouter(monkeypatch) -> None:
    monkeypatch.delenv("OCR_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OCR_OPENROUTER_FALLBACK", "true")
    adapter = build_llm_adapter()
    assert adapter.provider_name == "openrouter"
