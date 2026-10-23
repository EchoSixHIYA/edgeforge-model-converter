#!/usr/bin/env bash
# Optional, explicitly invoked installer. Never called automatically by Flask.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${COSMOS_PYTHON:-python3.10}"
VENV="${COSMOS_VENV:-$ROOT/.venv-cosmos-compiler}"
if ! command -v "$PYTHON" >/dev/null; then
  echo 'Install Python 3.10 first (official TPU-MLIR 1.28.1 wheel requirement).' >&2; exit 1
fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"'
"$PYTHON" -c 'import platform; name, version = platform.libc_ver(); assert name == "glibc" and tuple(map(int, version.split(".")[:2])) >= (2, 35), "TPU-MLIR 1.28.1 requires glibc >= 2.35; use Ubuntu 22.04+"'
for tool in cmake make gcc g++; do
  if ! command -v "$tool" >/dev/null; then
    echo "Missing host build tool: $tool. Install cmake and build-essential using your OS package manager." >&2
    exit 1
  fi
done
if ! command -v parallel >/dev/null || ! parallel --version | head -1 | grep -q 'GNU parallel'; then
  echo 'Install GNU parallel using your OS package manager before continuing.' >&2; exit 1
fi
if [[ ! -x "$VENV/bin/python" ]]; then "$PYTHON" -m venv "$VENV"; fi
"$VENV/bin/python" -m pip install --upgrade pip 'setuptools==75.8.0' wheel
# Keep separate from RKNN/CV requirements. Do not use tpu_mlir[torch], which
# requests an older Torch version incompatible with Transformers 5.3.
"$VENV/bin/python" -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/python" -m pip install tpu_mlir==1.28.1 transformers==5.3.0 'setuptools==75.8.0'
"$VENV/bin/python" -m pip check
"$VENV/bin/python" "$ROOT/server/vlm_worker.py" --probe
printf '\nCompiler installed; configure the SERVER shell with:\nexport EDGEFORGE_LLM_PYTHON=%q\n' "$VENV/bin/python"
echo 'This validates imports/prerequisites only, not model compilation or board inference.'
