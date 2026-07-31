#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p data
DEFAULT_VENV="/home/$(id -un)/.edgeforge-model-converter-venv-py310"
if [[ ! -x "$DEFAULT_VENV/bin/gunicorn" ]]; then
  DEFAULT_VENV="/home/$(id -un)/.edgeforge-model-converter-venv"
fi
VENV_DIR="${VENV_DIR:-$DEFAULT_VENV}"
SOPHON_VENV="${SOPHON_VENV:-/home/$(id -un)/.edgeforge-sophon-compiler-venv-py310}"
if [[ -d "$SOPHON_VENV/bin" ]]; then
  export PATH="$SOPHON_VENV/bin:$PATH"
fi
exec "$VENV_DIR/bin/gunicorn" -b 0.0.0.0:${PORT:-5000} --workers 1 --threads 8 --timeout 3600 --access-logfile - app:app
