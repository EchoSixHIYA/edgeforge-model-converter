#!/usr/bin/env bash
set -euo pipefail

ROOTS=("$@")
if [[ ${#ROOTS[@]} -eq 0 ]]; then
  ROOTS=(/home/linaro /data /opt)
fi

echo "[board-find] roots: ${ROOTS[*]}"

echo "[board-find] bmodel"
find "${ROOTS[@]}" -type f -name '*.bmodel' -printf '%s %p\n' 2>/dev/null \
  | sort -n \
  | head -100 || true

echo "[board-find] onnx/mlir/cvimodel"
find "${ROOTS[@]}" -type f \( -name '*.onnx' -o -name '*.mlir' -o -name '*.cvimodel' \) \
  -printf '%s %p\n' 2>/dev/null \
  | sort -n \
  | head -100 || true

echo "[board-find] tpu-mlir tools"
command -v model_transform.py || true
command -v run_calibration.py || true
command -v model_deploy.py || true
find /opt /home/linaro -maxdepth 6 -type f \
  \( -name model_transform.py -o -name run_calibration.py -o -name model_deploy.py \) \
  2>/dev/null \
  | sort \
  | head -100 || true
