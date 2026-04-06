from __future__ import annotations

import os
from typing import Iterable

# Fields the service knows how to map to a canonical public name.
# Any field name NOT in this dict is passed through as-is.
PUBLIC_FIELD_NAMES: dict[str, str] = {
    "invoice_number": "invoice_number",
    "date": "invoice_date",
    "amount": "total_amount",
    "currency_code": "currency_code",
}

# Kept for backwards-compat — callers that import this tuple still work,
# but it is no longer used as a validation whitelist.
DEFAULT_REQUIRED_FIELDS: tuple[str, ...] = (
    "invoice_number",
    "date",
    "amount",
    "currency_code",
)

# Deprecated alias
SUPPORTED_REQUIRED_FIELDS = DEFAULT_REQUIRED_FIELDS


def normalize_required_fields(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """Parse and deduplicate a required-fields value from a request or env var.

    Accepts a comma-separated string, an iterable of strings, or None
    (falls back to DEFAULT_REQUIRED_FIELDS).  Any non-empty field name is
    accepted — there is no whitelist.
    """
    raw_values: list[str] = []

    if value is None:
        raw_values = list(DEFAULT_REQUIRED_FIELDS)
    elif isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",")]
    else:
        raw_values = [str(item).strip() for item in value]

    normalized_values: list[str] = []
    seen: set[str] = set()

    for raw_value in raw_values:
        if not raw_value:
            continue
        if raw_value in seen:
            continue
        normalized_values.append(raw_value)
        seen.add(raw_value)

    if not normalized_values:
        raise ValueError("At least one required field must be specified.")

    return tuple(normalized_values)


def default_required_fields_from_env() -> tuple[str, ...]:
    return normalize_required_fields(os.getenv("OCR_REQUIRED_FIELDS"))


def public_field_name(field_name: str) -> str:
    """Return the canonical public name for a field.

    Known fields are remapped (e.g. 'date' → 'invoice_date').
    Unknown/custom fields are returned unchanged.
    """
    return PUBLIC_FIELD_NAMES.get(field_name, field_name)
