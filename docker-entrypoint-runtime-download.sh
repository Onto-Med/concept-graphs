#!/usr/bin/env bash
set -euo pipefail

cd /rest_api

echo "[concept-graphs] Synchronizing runtime dependencies..."
uv sync --frozen --no-group test

if [[ "${DOWNLOAD_MODELS:-true}" == "true" ]]; then
  echo "[concept-graphs] Downloading runtime NLP/tokenizer models..."
  uv run --no-sync python -m src.scripts.download_models
else
  echo "[concept-graphs] Skipping model download because DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-}"
fi

echo "[concept-graphs] Starting API..."
exec uv run --no-sync waitress-serve --call --port="${PORT:-9007}" main:create_app
