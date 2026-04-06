"""Pydantic response models for the OCR API.

Using explicit models gives callers a machine-readable OpenAPI schema at
/docs and /openapi.json, and catches serialisation bugs at the boundary.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class FieldValue(BaseModel):
    value: str | int | float | None
    confidence: float


class ExtractResponse(BaseModel):
    status: str
    message: str
    fields: dict[str, FieldValue]
    unmapped_fields: dict[str, Any]
    # Each dict has keys derived from the actual column headers found in the invoice
    # (e.g. {"description": "Widget A", "quantity": "2", "unit_price": "10.00", ...})
    line_items: list[dict[str, Any]] = []


class HealthResponse(BaseModel):
    status: str
    templates: int
