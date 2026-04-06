FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OCR_DEBUG=false \
    OCR_TIMEOUT=5 \
    OCR_REQUIRED_FIELDS="invoice_number,date,amount,currency_code" \
    OCR_TEMPLATE_DIR=/app/templates \
    OCR_TEMPLATE_DIRS=/app/templates \
    OCR_LLM_PROVIDER=none \
    OCR_OPENROUTER_FALLBACK=false \
    OCR_AUTO_GENERATE_TEMPLATES=false \
    OCR_TEMPLATE_HEALING_MAX_ATTEMPTS=3 \
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions \
    OPENROUTER_TIMEOUT=30 \
    OPENROUTER_REQUIRE_ZDR=false \
    OPENROUTER_DISABLE_REASONING=false \
    OPENROUTER_PDF_ENGINE=native \
    OPENROUTER_RATE_LIMIT_RETRIES=2 \
    OPENROUTER_RATE_LIMIT_BACKOFF_MS=2000 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434/v1/chat/completions \
    OLLAMA_TIMEOUT=60 \
    OLLAMA_MAX_PDF_PAGES=4
# OCR_API_KEY — set at runtime to require bearer token auth on /extract.
# Leave unset (or empty) for internal-only deployments where network isolation is sufficient.
# Persistent templates: mount a host volume to /app/templates so generated templates
# survive container restarts, e.g.:
#   docker run -v /data/ocr-templates:/app/templates ...

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-nld \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ocr_service /app/ocr_service
COPY templates /app/templates

EXPOSE 8080

CMD ["uvicorn", "ocr_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
