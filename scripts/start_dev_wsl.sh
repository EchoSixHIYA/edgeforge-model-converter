#!/usr/bin/env bash
# Local PR testing: use an independent ASCII task path for the VLM compiler.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV_CONFIG="${EDGEFORGE_DEV_CONFIG:-$ROOT/data/dev.env}"
if [[ -f "$DEV_CONFIG" ]]; then
  # This is a trusted, local-only shell config under the ignored data directory.
  source "$DEV_CONFIG"
fi
DEV_STATE="${EDGEFORGE_DEV_STATE:-/home/$(id -un)/.edgeforge-model-converter-dev}"
export EDGEFORGE_DATA_DIR="${EDGEFORGE_DATA_DIR:-$DEV_STATE/tasks}"
mkdir -p "$EDGEFORGE_DATA_DIR"
printf 'EdgeForge dev task directory: %s\n' "$EDGEFORGE_DATA_DIR"
# Pass through explicit model-root and compiler settings. Do not authorize a
# broad filesystem root or install compiler packages just by starting the web app.
exec bash "$ROOT/scripts/start_wsl.sh"
