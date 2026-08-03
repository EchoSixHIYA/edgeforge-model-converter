#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
  echo "Using active virtual environment: ${VIRTUAL_ENV}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  VENV_DIR="${EDGEFORGE_VENV:-${ROOT}/.venv}"
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Creating virtual environment: ${VENV_DIR}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  PYTHON="${VENV_DIR}/bin/python"
fi

PY_TAG="$($PYTHON -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
case "$PY_TAG" in
  cp38|cp39|cp310|cp311|cp312) ;;
  *)
    echo "Error: EdgeForge SDK supports CPython 3.8-3.12, current is $($PYTHON -V 2>&1)." >&2
    exit 3
    ;;
esac

shopt -s nullglob
RKNN_MODELS=("${ROOT}"/model/*.rknn "${ROOT}"/edgeforge_sdk/*.rknn)
BMODELS=("${ROOT}"/model/*.bmodel "${ROOT}"/edgeforge_sdk/*.bmodel)

if ((${#RKNN_MODELS[@]})); then
  MACHINE="$(uname -m)"
  case "$MACHINE" in
    aarch64|arm64) ;;
    *)
      echo "Error: packaged RKNNLite wheels support aarch64 Linux boards only; current platform is $MACHINE." >&2
      exit 2
      ;;
  esac
  PACKAGES="${ROOT}/runtime/rknn-toolkit-lite2/packages"
  WHEELS=("${PACKAGES}"/rknn_toolkit_lite2-*-${PY_TAG}-${PY_TAG}-*aarch64*.whl)
  if [[ ${#WHEELS[@]} -ne 1 ]]; then
    echo "Error: no RKNN-Toolkit-Lite2 wheel found for ${PY_TAG}/aarch64." >&2
    exit 4
  fi
  WHEEL="${WHEELS[0]}"
  if command -v md5sum >/dev/null 2>&1 && [[ -f "${PACKAGES}/packages.md5sum" ]]; then
    EXPECTED="$(grep "  $(basename "$WHEEL")$" "${PACKAGES}/packages.md5sum")"
    (cd "$PACKAGES" && printf '%s\n' "$EXPECTED" | md5sum -c -)
  fi
  echo "Installing packaged RKNN-Toolkit-Lite2: $(basename "$WHEEL")"
  "$PYTHON" -m pip install --disable-pip-version-check "$WHEEL"
fi

echo "Installing EdgeForge SDK and Python dependencies"
"$PYTHON" -m pip install --disable-pip-version-check "$ROOT"

if ((${#BMODELS[@]})); then
  if "$PYTHON" - <<'PY' >/dev/null 2>&1
import sophon.sail  # noqa: F401
PY
  then
    echo "SOPHON-SAIL Python runtime found."
  else
    echo "Warning: this package contains .bmodel files, but sophon.sail is not installed in this Python environment." >&2
    echo "Install the official SOPHON-SAIL runtime on the BM1684/BM1684X device before running inference." >&2
  fi
fi

echo
echo "Installation complete. Reuse this SDK for later model bundles by copying their model/ directory."
echo "Run example:"
echo "  $PYTHON ${ROOT}/example.py"
