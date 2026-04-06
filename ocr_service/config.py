"""Central configuration — all env-var reads live here.

Import `get_config()` anywhere you need runtime settings instead of
calling `os.getenv` directly in business logic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on"}


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return raw.strip().lower() in _TRUTHY if raw is not None else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class OcrConfig:
    # Auth
    api_key: str

    # Templates
    template_dir: Path
    template_dirs: list[Path]

    # Runtime flags
    debug: bool
    required_fields: str
    auto_generate_templates: bool
    template_healing_max_attempts: int

    # LLM
    llm_provider: str  # "openrouter" | "ollama" | "none"


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    base_url: str
    model: str
    template_model: str
    timeout: int
    require_zdr: bool
    disable_reasoning: bool
    http_referer: str
    app_name: str
    pdf_engine: str
    rate_limit_retries: int
    rate_limit_backoff_ms: int


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model: str
    template_model: str
    timeout: int
    max_pdf_pages: int


@dataclass(frozen=True)
class AppConfig:
    ocr: OcrConfig
    openrouter: OpenRouterConfig
    ollama: OllamaConfig


def _parse_template_dirs(template_dir: Path) -> list[Path]:
    raw = os.getenv("OCR_TEMPLATE_DIRS", "").strip()
    if not raw:
        return [template_dir]
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def get_config() -> AppConfig:
    template_dir = Path(os.getenv("OCR_TEMPLATE_DIR", "/app/templates"))
    template_dirs = _parse_template_dirs(template_dir)

    ocr = OcrConfig(
        api_key=os.getenv("OCR_API_KEY", "").strip(),
        template_dir=template_dir,
        template_dirs=template_dirs,
        debug=_flag("OCR_DEBUG"),
        required_fields=os.getenv(
            "OCR_REQUIRED_FIELDS", "invoice_number,date,amount,currency_code"
        ),
        auto_generate_templates=_flag("OCR_AUTO_GENERATE_TEMPLATES"),
        template_healing_max_attempts=_int("OCR_TEMPLATE_HEALING_MAX_ATTEMPTS", 3),
        llm_provider=os.getenv("OCR_LLM_PROVIDER", "").strip().lower(),
    )

    openrouter = OpenRouterConfig(
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1/chat/completions",
        ),
        model=os.getenv("OPENROUTER_MODEL", ""),
        template_model=os.getenv("OPENROUTER_TEMPLATE_MODEL", ""),
        timeout=_int("OPENROUTER_TIMEOUT", 30),
        require_zdr=_flag("OPENROUTER_REQUIRE_ZDR"),
        disable_reasoning=_flag("OPENROUTER_DISABLE_REASONING", default=True),
        http_referer=os.getenv("OPENROUTER_HTTP_REFERER", ""),
        app_name=os.getenv("OPENROUTER_APP_NAME", ""),
        pdf_engine=os.getenv("OPENROUTER_PDF_ENGINE", "native"),
        rate_limit_retries=_int("OPENROUTER_RATE_LIMIT_RETRIES", 2),
        rate_limit_backoff_ms=_int("OPENROUTER_RATE_LIMIT_BACKOFF_MS", 2000),
    )

    ollama = OllamaConfig(
        base_url=os.getenv(
            "OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1/chat/completions"
        ),
        model=os.getenv("OLLAMA_MODEL", ""),
        template_model=os.getenv("OLLAMA_TEMPLATE_MODEL", ""),
        timeout=_int("OLLAMA_TIMEOUT", 60),
        max_pdf_pages=_int("OLLAMA_MAX_PDF_PAGES", 4),
    )

    return AppConfig(ocr=ocr, openrouter=openrouter, ollama=ollama)
