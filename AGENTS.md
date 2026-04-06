# OCR Notes

- Fix extraction root causes in `ocr_service` before changing supplier templates.
- Treat supplier templates as the last mile only after extractor, template-generator, and self-healing logic are correct.
- For PDF issues, compare `pdftotext` output with `_extract_ocr_text()` output before changing matching logic.
- When template-generator text sources differ, prefer the more complete source for preview and generation instead of blindly trusting the first OCR pass.
