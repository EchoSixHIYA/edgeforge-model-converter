#!/usr/bin/env bash
set -euo pipefail

SEARCH_ROOT="${1:-/home/linaro/3588_edge}"
LOOPNUM="${LOOPNUM:-1}"
VENV_DIR="${VENV_DIR:-$SEARCH_ROOT/.edgeforge-board-smoke-venv}"

log() {
  printf '[board-sophon-smoke] %s\n' "$*"
}

fail() {
  printf '[board-sophon-smoke] ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ ! -d "$SEARCH_ROOT" ]]; then
  fail "search root does not exist: $SEARCH_ROOT"
fi

log "board: $(uname -a)"
log "search root: $SEARCH_ROOT"

if [[ ! -d "$VENV_DIR" ]]; then
  log "creating project virtualenv: $VENV_DIR"
  if ! python3 -m venv --system-site-packages "$VENV_DIR"; then
    log "venv ensurepip is unavailable; recreating without pip"
    python3 -m venv --clear --system-site-packages --without-pip "$VENV_DIR"
  fi
fi
PYTHON="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  fail "virtualenv python was not created: $PYTHON"
fi
log "python: $("$PYTHON" --version 2>&1)"
USER_SITE="$(python3 -m site --user-site 2>/dev/null || true)"
if [[ -n "$USER_SITE" && -d "$USER_SITE" ]]; then
  export PYTHONPATH="$USER_SITE${PYTHONPATH:+:$PYTHONPATH}"
  log "python user site: $USER_SITE"
fi

if command -v bmrt_test >/dev/null 2>&1; then
  BMRT_TEST="$(command -v bmrt_test)"
else
  BMRT_TEST="$(find /opt/sophon "$SEARCH_ROOT" -maxdepth 5 -type f -name bmrt_test 2>/dev/null | head -n 1 || true)"
  if [[ -z "$BMRT_TEST" ]]; then
    fail "bmrt_test is not in PATH and was not found under /opt/sophon or $SEARCH_ROOT"
  fi
fi
log "bmrt_test: $BMRT_TEST"

if "$PYTHON" - <<'PY' >/tmp/edgeforge_sail_check.txt 2>&1
import sophon.sail as sail
print(getattr(sail, "__file__", "sophon.sail"))
PY
then
  log "sophon.sail: $(cat /tmp/edgeforge_sail_check.txt)"
else
  cat /tmp/edgeforge_sail_check.txt >&2 || true
  fail "python3 cannot import sophon.sail"
fi

BMODEL="$(
  find "$SEARCH_ROOT" -type f -name '*.bmodel' -printf '%s\t%p\n' 2>/dev/null \
    | sort -n \
    | head -n 1 \
    | cut -f2-
)"

if [[ -z "$BMODEL" ]]; then
  fail "no .bmodel found under $SEARCH_ROOT"
fi

log "selected smallest bmodel: $BMODEL"
log "size: $(du -h "$BMODEL" | awk '{print $1}')"

"$BMRT_TEST" --bmodel "$BMODEL" --loopnum "$LOOPNUM"

log "bmrt_test passed"

log "running SOPHON-SAIL Engine smoke"
"$PYTHON" - "$BMODEL" <<'PY'
import sys
import numpy as np
import sophon.sail as sail

bmodel = sys.argv[1]
engine = sail.Engine(bmodel, 0, sail.IOMode.SYSIO)
graph = engine.get_graph_names()[0]
inputs = engine.get_input_names(graph)
outputs = engine.get_output_names(graph)
feed = {}

def dtype_for(value):
    text = str(value).upper()
    if "INT8" in text:
        return np.int8
    if "UINT8" in text:
        return np.uint8
    if "FLOAT16" in text or "FP16" in text:
        return np.float16
    return np.float32

for name in inputs:
    shape = tuple(int(x) for x in engine.get_input_shape(graph, name))
    try:
        dtype = dtype_for(engine.get_input_dtype(graph, name))
    except Exception:
        dtype = np.float32
    feed[name] = np.zeros(shape, dtype=dtype)

result = engine.process(graph, feed)
if isinstance(result, dict):
    result_items = [(name, result[name]) for name in outputs]
else:
    result_items = list(zip(outputs, result))

print(f"[board-sophon-smoke] sail graph: {graph}")
for name, value in feed.items():
    print(f"[board-sophon-smoke] sail input {name}: shape={value.shape} dtype={value.dtype}")
for name, value in result_items:
    array = np.asarray(value)
    print(f"[board-sophon-smoke] sail output {name}: shape={array.shape} dtype={array.dtype}")
PY

log "SOPHON-SAIL Engine smoke passed"
