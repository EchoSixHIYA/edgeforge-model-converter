#!/usr/bin/env bash
set -euo pipefail

SEARCH_ROOT="${1:-/home/linaro/3588_edge}"

echo "[board-probe] user=$(whoami)"
echo "[board-probe] host=$(hostname)"
echo "[board-probe] kernel=$(uname -a)"
echo "[board-probe] path=$PATH"
echo "[board-probe] python=$(command -v python3 || true)"
python3 --version || true

echo "[board-probe] runtime tools"
command -v bmrt_test || true
command -v rknn_benchmark || true

echo "[board-probe] possible runtime env scripts"
find "$SEARCH_ROOT" /opt /usr/local "$HOME" -maxdepth 5 \
  \( -name bmrt_test -o -name envsetup.sh -o -name envsetup_cmodel.sh -o \
     -name set_env.sh -o -name sophon-*.sh \) 2>/dev/null \
  | sort \
  | head -100 || true

echo "[board-probe] bmodels under $SEARCH_ROOT"
find "$SEARCH_ROOT" -type f -name '*.bmodel' -printf '%s %p\n' 2>/dev/null \
  | sort -n \
  | head -50 || true

echo "[board-probe] rknn under $SEARCH_ROOT"
find "$SEARCH_ROOT" -type f -name '*.rknn' -printf '%s %p\n' 2>/dev/null \
  | sort -n \
  | head -20 || true

echo "[board-probe] python runtime imports"
python3 - <<'PY'
for module in ("sophon.sail", "rknnlite.api"):
    try:
        imported = __import__(module, fromlist=["*"])
        print(f"{module}: ok {getattr(imported, '__file__', '')}")
    except Exception as exc:
        print(f"{module}: missing {exc!r}")
PY
