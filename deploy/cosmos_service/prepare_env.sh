#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export TMPDIR="$ROOT/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$ROOT/logs"
PYTHON="$ROOT/.venv/bin/python"
"$PYTHON" -m pip install --upgrade 'pip<27' 'setuptools>=64' wheel
# Keep the verified CPU wheels already present on this board. Do not replace
# them with another architecture/runtime build just to change the index.
"$PYTHON" -c 'import importlib.metadata as m; assert m.version("torch").startswith("2.9.0") and m.version("torchvision").startswith("0.24.0"), "Prepare the matching CPU torch 2.9.0 / torchvision 0.24.0 wheels first"'
"$PYTHON" -m pip install cmake==3.31.6 transformers==4.57.1 av==16.0.1 pybind11==2.13.6 'Pillow>=10,<12' -r "$ROOT/requirements.txt"
echo PREPARE_ENV_OK
