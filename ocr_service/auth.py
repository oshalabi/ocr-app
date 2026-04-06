"""Bearer token authentication dependency for FastAPI endpoints."""
from __future__ import annotations

import logging
import os

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ocr_service.logger import get_logger, log_event

logger = get_logger("ocr_service.auth")
_bearer_scheme = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> None:
    """FastAPI dependency — enforces bearer token auth when OCR_API_KEY is set.

    If OCR_API_KEY is empty the check is skipped, making this safe for
    internal-only Docker network deployments where network isolation is enough.
    """
    expected = os.getenv("OCR_API_KEY", "").strip()
    if not expected:
        return

    token = credentials.credentials if credentials else None
    if not token or token != expected:
        log_event(logger, logging.WARNING, "http.auth.rejected")
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
