from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEBUG_TRUTHY_VALUES = {"1", "true", "yes", "on"}
LOGGER_NAMESPACE = "ocr_service"


class OcrLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        action = getattr(record, "action", None)

        if isinstance(action, str) and action:
            payload["action"] = action

        context = getattr(record, "context", None)

        if isinstance(context, dict) and context:
            payload["context"] = {
                key: normalize_log_value(value)
                for key, value in context.items()
            }

        if record.exc_info and is_debug_enabled():
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def is_debug_enabled() -> bool:
    return os.getenv("OCR_DEBUG", "false").strip().lower() in DEBUG_TRUTHY_VALUES


def configure_logging(force: bool = False) -> None:
    logger = logging.getLogger(LOGGER_NAMESPACE)

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(OcrLogFormatter())
        logger.addHandler(handler)

    logger.setLevel(logging.DEBUG if is_debug_enabled() else logging.INFO)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    level: int,
    action: str,
    **context: Any,
) -> None:
    logger.log(
        level,
        action,
        extra={
            "action": action,
            "context": {
                key: value
                for key, value in context.items()
                if value is not None
            },
        },
    )


def normalize_log_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, (list, tuple, set, frozenset)):
        return [normalize_log_value(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): normalize_log_value(item)
            for key, item in value.items()
        }

    return str(value)
