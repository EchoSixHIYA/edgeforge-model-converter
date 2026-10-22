#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON_BIN:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3,10), "VLM runtime requires Python >=3.10"'
if [[ ! -f "${SOPHON_SDK_ROOT:-/opt/sophon/libsophon-current}/include/bmruntime_interface.h" ]]; then
  echo 'Missing libsophon development headers. Install the official matching libsophon runtime/dev SDK and board/PCIe driver; set SOPHON_SDK_ROOT if not under /opt/sophon/libsophon-current.' >&2
  exit 2
fi
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$("$PYTHON" -c 'import sys; print(sys.executable)')" \
  -DSOPHON_SDK_ROOT="${SOPHON_SDK_ROOT:-/opt/sophon/libsophon-current}"
cmake --build "$ROOT/build" --parallel "${BUILD_JOBS:-2}"
cmake --install "$ROOT/build"
"$PYTHON" -c 'import edgeforge_qwen3_vl as r; assert r.EDGEFORGE_ABI == 1; print("Native extension built (not a hardware inference test).")'
