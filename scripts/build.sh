#!/usr/bin/env bash
# build.sh — build (and optionally push) the OCR sidecar Docker image.
#
# Usage:
#   ./build.sh                        # build with tag ocr-sidecar:latest
#   ./build.sh 1.2.3                  # build with tag ocr-sidecar:1.2.3 + ocr-sidecar:latest
#   ./build.sh 1.2.3 ghcr.io/you/ocr  # build + push to a registry
#
# Environment variables (override on the command line or export before running):
#   IMAGE   — base image name          (default: ocr-sidecar)
#   PUSH    — set to "1" to push       (default: 0)

set -euo pipefail

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
if [[ -f .env.local ]]; then
  set -a; source .env.local; set +a
fi

VERSION="${1:-latest}"
REGISTRY="${2:-}"
IMAGE="${IMAGE:-ocr-sidecar}"

if [[ -n "$REGISTRY" ]]; then
  FULL_IMAGE="${REGISTRY}/${IMAGE}"
else
  FULL_IMAGE="${IMAGE}"
fi

TAGS=("${FULL_IMAGE}:${VERSION}")
if [[ "$VERSION" != "latest" ]]; then
  TAGS+=("${FULL_IMAGE}:latest")
fi

TAG_ARGS=()
for tag in "${TAGS[@]}"; do
  TAG_ARGS+=("-t" "$tag")
done

echo "==> Building ${TAGS[*]}"
docker build "${TAG_ARGS[@]}" .

if [[ "${PUSH:-0}" == "1" || -n "$REGISTRY" ]]; then
  for tag in "${TAGS[@]}"; do
    echo "==> Pushing $tag"
    docker push "$tag"
  done
fi

echo "==> Done"
