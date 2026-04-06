# Invoice OCR Service

A standalone FastAPI microservice that extracts structured fields from invoice and receipt images/PDFs. It can be used independently or as a sidecar alongside any application — it ships with built-in integration for [InvoiceShelf](https://github.com/invoiceshelf/invoiceshelf) (called from Laravel via `HttpExpenseOcrService`).

## How it works

1. Laravel uploads a file to `POST /extract` with an optional bearer token.
2. The service runs [invoice2data](https://github.com/invoice-x/invoice2data) template matching against YAML templates.
3. If no template matches **and** an LLM provider is configured, it falls back to an LLM (OpenRouter or Ollama) to extract fields.
4. Optionally the LLM can generate and persist a new YAML template for future runs.

```
Any HTTP client  ──POST /extract──►  FastAPI service
(Laravel, Python,                          │
 curl, etc.)                       invoice2data (templates)
                                           │ (no match)
                                    LLM fallback (optional)
                                           │
                                     JSON response
```

---

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/` | — | Browser OCR playground |
| `GET` | `/health` | — | Health check (`{"status":"ok","templates":N}`) |
| `POST` | `/extract` | Bearer (if `OCR_API_KEY` set) | Extract fields from a file |
| `GET` | `/template-generator` | — | Browser template generator UI |
| `POST` | `/playground/result` | — | Playground form handler |
| `POST` | `/template-generator/result` | — | Generate and save a starter template |

### `POST /extract`

**Form fields**

| Field | Default | Description |
|-------|---------|-------------|
| `file` | required | PDF, JPG, or PNG |
| `country_code` | `NL` | ISO 3166-1 alpha-2 |
| `required_fields` | `invoice_number,date,amount,currency_code` | Comma-separated list of fields to extract. Any field name is accepted — see below. |

**Required fields**

The `required_fields` parameter is open-ended. Pass any field names you need — the LLM and template engine will look for them in the document. Well-known fields have canonical public names in the response:

| Field name | Response key | Description |
|------------|-------------|-------------|
| `invoice_number` | `invoice_number` | Invoice / receipt identifier |
| `date` | `invoice_date` | Issue date |
| `amount` | `total_amount` | Total amount payable |
| `currency_code` | `currency_code` | ISO 4217 currency code |
| `vat_number` | `vat_number` | VAT / BTW registration number |
| `iban` | `iban` | Bank account IBAN |
| `po_number` | `po_number` | Purchase order number |
| *(any name)* | *(same name)* | Custom field — passed through as-is |

Example — request only what you need:

```bash
curl -X POST http://localhost:8080/extract \
  -H "Authorization: Bearer $OCR_API_KEY" \
  -F "file=@invoice.pdf" \
  -F "country_code=NL" \
  -F "required_fields=invoice_number,date,amount,vat_number,iban"
```

**Response**

```json
{
  "status": "success",
  "message": "Invoice fields extracted successfully.",
  "fields": {
    "invoice_number": { "value": "INV-2026-001", "confidence": 0.98 },
    "date":           { "value": "2026-04-06",   "confidence": 0.95 },
    "amount":         { "value": "123.45",        "confidence": 0.97 },
    "currency_code":  { "value": "EUR",           "confidence": 1.0  }
  },
  "unmapped_fields": {}
}
```

---

## Configuration

All configuration is done via environment variables.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_API_KEY` | _(empty)_ | Bearer token required on `/extract`. Leave empty to disable auth (internal Docker network only). |
| `OCR_TEMPLATE_DIR` | `/app/templates` | Writable directory where generated templates are saved. |
| `OCR_TEMPLATE_DIRS` | `/app/templates` | Comma-separated list of directories to load templates from (read order). |
| `OCR_REQUIRED_FIELDS` | `invoice_number,date,amount,currency_code` | Default required fields when the caller does not specify any. |
| `OCR_DEBUG` | `false` | Emit structured JSON debug logs to stderr. |

### LLM fallback

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_LLM_PROVIDER` | _(empty)_ | Explicit provider: `openrouter`, `ollama`, or `none`. Takes precedence over legacy variable. |
| `OCR_OPENROUTER_FALLBACK` | `false` | Legacy flag — enables OpenRouter when `OCR_LLM_PROVIDER` is not set. |
| `OCR_AUTO_GENERATE_TEMPLATES` | `false` | When `true`, a successful LLM extraction saves a new YAML template. |
| `OCR_TEMPLATE_HEALING_MAX_ATTEMPTS` | `3` | How many LLM retries to attempt when a generated template is invalid. |

### OpenRouter

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | required | Your OpenRouter API key. |
| `OPENROUTER_MODEL` | — | Model ID for field extraction (e.g. `qwen/qwen3-plus:free`). |
| `OPENROUTER_TEMPLATE_MODEL` | — | Model ID used for template generation (falls back to `OPENROUTER_MODEL`). |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1/chat/completions` | API base URL. |
| `OPENROUTER_TIMEOUT` | `30` | Request timeout in seconds. |
| `OPENROUTER_REQUIRE_ZDR` | `false` | Require zero-data-retention routing. |
| `OPENROUTER_DISABLE_REASONING` | `true` | Strip chain-of-thought tokens from responses. |
| `OPENROUTER_RATE_LIMIT_RETRIES` | `2` | Number of retries on 429 responses. |
| `OPENROUTER_RATE_LIMIT_BACKOFF_MS` | `2000` | Delay between rate-limit retries in milliseconds. |
| `OPENROUTER_PDF_ENGINE` | `native` | How PDFs are passed to the model (`native` or `text`). |

### Ollama

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL of your Ollama instance. |
| `OLLAMA_MODEL` | — | Model to use (e.g. `llama3`). |

---

## Running locally

### With Docker (recommended)

First, build the Docker image:

```bash
# Mac/Linux
./scripts/build.sh

# Windows (PowerShell)
.\scripts\build.ps1
```

Then create a `.env.local` file based on `.env` and fill in your values.

Run the container using the provided scripts (which automatically load `.env.local` and mount a `templates` directory):

```bash
# Mac/Linux
./scripts/run.sh

# Windows (PowerShell)
.\scripts\run.ps1
```

Open [http://localhost:8080](http://localhost:8080) for the playground.

### Without Docker

```bash
pip install -r requirements-dev.txt

cp .env .env.local   # fill in your values
set -a && source .env.local && set +a   # or use direnv

uvicorn ocr_service.main:app --reload --port 8080
```

---

## Persistent templates

By default, the container stores templates inside `/app/templates`. To survive restarts, mount a host directory or a named Docker volume:

```bash
# Named volume
docker volume create ocr-templates
docker run ... -v ocr-templates:/app/templates ocr-sidecar

# Host directory
docker run ... -v /data/ocr-templates:/app/templates ocr-sidecar
```

Set `OCR_TEMPLATE_DIR` to the same path so newly generated templates are written there:

```
OCR_TEMPLATE_DIR=/app/templates
```

To load templates from multiple directories (e.g. read-only bundled + writable custom):

```
OCR_TEMPLATE_DIRS=/app/templates/bundled,/app/templates/custom
OCR_TEMPLATE_DIR=/app/templates/custom
```

---

## Authentication

When `OCR_API_KEY` is set, the `/extract` endpoint requires:

```
Authorization: Bearer <OCR_API_KEY>
```

Laravel's `HttpExpenseOcrService` sends this header automatically when `OCR_API_KEY` is configured in the Laravel `.env`.

If `OCR_API_KEY` is empty the header is not required — suitable when the service is reachable only within a private Docker network.

---

## Templates

Templates are YAML files read by [invoice2data](https://github.com/invoice-x/invoice2data). They live under `OCR_TEMPLATE_DIR` organised as:

```
templates/
  <country_code>/
    <issuer_slug>/
      template.yml
      template_1.yml   # additional variants
```

### Generating a starter template

1. Open the Template Generator UI at [http://localhost:8080/template-generator](http://localhost:8080/template-generator).
2. Upload a sample invoice and fill in the field labels.
3. The service generates a `template.yml` and saves it into the writable template directory.

Or enable `OCR_AUTO_GENERATE_TEMPLATES=true` to let the service generate templates automatically after each successful LLM extraction.

---

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

Tests use `pytest` with FastAPI's `TestClient` — no running server needed.

---

## Docker

See the [`scripts` directory](./scripts/) for canonical build and run scripts, or use the [GitHub Actions workflow](.github/workflows/ci.yml) which tests, builds, and pushes to GitHub Container Registry on every push to `main`.
