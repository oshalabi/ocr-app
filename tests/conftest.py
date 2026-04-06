"""Shared pytest fixtures available to all test modules."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ocr_service.main import ocr_service


@pytest.fixture(scope="session")
def client() -> TestClient:
    """A single TestClient instance reused across the whole test session."""
    return TestClient(app)


@pytest.fixture()
def tmp_template_dir(tmp_path: Path) -> Path:
    """An empty temporary directory suitable for use as a writable template dir."""
    d = tmp_path / "templates"
    d.mkdir()
    return d


@pytest.fixture()
def minimal_pdf() -> bytes:
    """Minimal valid PDF header bytes — enough to pass file-type validation."""
    return b"%PDF-1.7"


@pytest.fixture()
def minimal_png() -> bytes:
    """Minimal PNG magic bytes."""
    return b"png-data"


@pytest.fixture()
def demo_invoice_text(fixtures_dir: Path) -> str:
    return (fixtures_dir / "demo_supplier_invoice.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
