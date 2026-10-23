#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VIRTUAL_ENV="$ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PIP_CACHE_DIR="$ROOT/.cache/pip" TMPDIR="$ROOT/tmp"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export SOPHON_SDK_ROOT="${SOPHON_SDK_ROOT:-/opt/sophon/libsophon-current}"
export BUILD_JOBS=2
"$VIRTUAL_ENV/bin/python" -m pip install "$ROOT/sdk[vlm]"
PYTHON_BIN="$VIRTUAL_ENV/bin/python" bash "$ROOT/sdk/runtime/qwen3_vl/build.sh"
"$VIRTUAL_ENV/bin/python" -m pip check
"$VIRTUAL_ENV/bin/python" -m pip freeze > "$ROOT/logs/installed-packages.txt"
echo INSTALL_SDK_OK
