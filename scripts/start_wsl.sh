#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p data
VENV_DIR="/home/$(id -un)/.edgeforge-model-converter-venv"
exec "$VENV_DIR/bin/gunicorn" -b 0.0.0.0:${PORT:-5000} --workers 1 --threads 8 --timeout 3600 --access-logfile - app:app
