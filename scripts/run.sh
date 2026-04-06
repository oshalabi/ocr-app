#!/usr/bin/env bash
# run.sh — run the OCR sidecar Docker image locally.
#
# Usage:
#   ./scripts/run.sh [additional docker run args...]

set -euo pipefail

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
if [[ -f .env.local ]]; then
  set -a; source .env.local; set +a
fi

IMAGE="${IMAGE:-ocr-sidecar:latest}"
PORT="${PORT:-8080}"
TEMPLATE_DIR="${TEMPLATE_DIR:-$(pwd)/templates}"

# Create templates directory if it doesn't exist
mkdir -p "$TEMPLATE_DIR"

echo "==> Building image (${IMAGE})..."
"$(dirname "$0")/build.sh"


DOCKER_ARGS=(
  "--rm"
  "-it"
  "-p" "${PORT}:8080"
  "-v" "${TEMPLATE_DIR}:/app/templates"
)

if [[ -f .env ]]; then
  DOCKER_ARGS+=("--env-file" ".env")
fi
if [[ -f .env.local ]]; then
  DOCKER_ARGS+=("--env-file" ".env.local")
fi

echo "==> Running ${IMAGE} on port ${PORT}..."
docker run "${DOCKER_ARGS[@]}" "$@" "${IMAGE}"
