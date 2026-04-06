import json
import logging

from ocr_service.logger import configure_logging, get_logger, is_debug_enabled, log_event


def test_configure_logging_respects_ocr_debug(monkeypatch) -> None:
    monkeypatch.setenv("OCR_DEBUG", "true")
    configure_logging(force=True)

    assert is_debug_enabled() is True
    assert get_logger("ocr_service.test").getEffectiveLevel() == logging.DEBUG

    monkeypatch.setenv("OCR_DEBUG", "false")
    configure_logging(force=True)

    assert is_debug_enabled() is False
    assert get_logger("ocr_service.test").getEffectiveLevel() == logging.INFO


def test_log_event_outputs_structured_json(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OCR_DEBUG", "true")
    configure_logging(force=True)
    capsys.readouterr()

    logger = get_logger("ocr_service.test")
    log_event(
        logger,
        logging.INFO,
        "test.action",
        file_name="invoice.pdf",
        required_fields=("invoice_number", "date"),
    )

    captured = capsys.readouterr().err.strip().splitlines()
    payload = json.loads(captured[-1])

    assert payload["action"] == "test.action"
    assert payload["context"]["file_name"] == "invoice.pdf"
    assert payload["context"]["required_fields"] == ["invoice_number", "date"]
