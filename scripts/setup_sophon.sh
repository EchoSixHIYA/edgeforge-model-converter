#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p vendor data/tasks

SOPHON_DEMO_DIR="${SOPHON_DEMO:-$ROOT/vendor/sophon-demo}"
if [[ ! -d "$SOPHON_DEMO_DIR/.git" ]]; then
  if [[ -e "$SOPHON_DEMO_DIR" ]]; then
    echo "Path exists but is not a git checkout: $SOPHON_DEMO_DIR" >&2
    echo "Move it aside or set SOPHON_DEMO to an existing sophon-demo checkout." >&2
    exit 1
  fi
  git clone --depth 1 https://github.com/sophgo/sophon-demo.git "$SOPHON_DEMO_DIR"
fi

missing=()
for command_name in model_transform.py model_deploy.py; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    missing+=("$command_name")
  fi
done

if ((${#missing[@]})); then
  echo "SOPHGO demo repository is ready at: $SOPHON_DEMO_DIR"
  echo "TPU-MLIR is not active in PATH; missing: ${missing[*]}"
  echo "Start the official TPU-MLIR environment, then re-run this script to verify conversion commands."
else
  echo "TPU-MLIR commands found."
fi

if python3 - <<'PY' >/dev/null 2>&1
import sophon.sail  # noqa: F401
PY
then
  echo "SOPHON-SAIL Python runtime found."
else
  echo "SOPHON-SAIL Python runtime is not installed in the current Python environment."
  echo "This is required only for running .bmodel files through edgeforge_sdk on BM1684X."
fi

echo "SOPHGO setup check completed."
