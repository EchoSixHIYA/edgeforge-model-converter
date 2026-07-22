#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE="$(uname -m)"
case "$MACHINE" in
  aarch64|arm64) ;;
  *)
    echo "错误：随包 RKNN-Toolkit-Lite2 仅适用于 aarch64 Linux 板端，当前平台为 $MACHINE。" >&2
    exit 2
    ;;
esac

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
  echo "使用当前虚拟环境：${VIRTUAL_ENV}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  VENV_DIR="${EDGEFORGE_VENV:-${ROOT}/.venv}"
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "创建虚拟环境：${VENV_DIR}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  PYTHON="${VENV_DIR}/bin/python"
fi

PY_TAG="$($PYTHON -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
case "$PY_TAG" in
  cp38|cp39|cp310|cp311|cp312) ;;
  *)
    echo "错误：本 SDK 支持板端 CPython 3.8-3.12，当前为 $($PYTHON -V 2>&1)。" >&2
    exit 3
    ;;
esac

PACKAGES="${ROOT}/runtime/rknn-toolkit-lite2/packages"
shopt -s nullglob
WHEELS=("${PACKAGES}"/rknn_toolkit_lite2-*-${PY_TAG}-${PY_TAG}-*aarch64*.whl)
if [[ ${#WHEELS[@]} -ne 1 ]]; then
  echo "错误：没有找到 ${PY_TAG}/aarch64 对应的 RKNN-Toolkit-Lite2 wheel。" >&2
  exit 4
fi
WHEEL="${WHEELS[0]}"

if command -v md5sum >/dev/null 2>&1; then
  EXPECTED="$(grep "  $(basename "$WHEEL")$" "${PACKAGES}/packages.md5sum")"
  (cd "$PACKAGES" && printf '%s\n' "$EXPECTED" | md5sum -c -)
fi

echo "安装随包 RKNN-Toolkit-Lite2：$(basename "$WHEEL")"
"$PYTHON" -m pip install --disable-pip-version-check "$WHEEL"
echo "安装 EdgeForge SDK 及其 Python 依赖"
"$PYTHON" -m pip install --disable-pip-version-check "$ROOT"

echo
echo "安装完成。SDK 以后无需随模型重复安装，只需复制新转换包的 model/ 目录。"
echo "运行示例："
echo "  $PYTHON ${ROOT}/example.py"
