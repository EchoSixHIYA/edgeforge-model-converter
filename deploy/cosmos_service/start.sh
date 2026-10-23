#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
ulimit -c 0
mkdir -p logs tmp
if [[ -f logs/gunicorn.pid ]] && kill -0 "$(< logs/gunicorn.pid)" 2>/dev/null; then
  echo "PID file already points to a running process; check logs/gunicorn.pid before starting again."
  exit 1
fi
export PATH="$ROOT/.venv/bin:$PATH"
export LD_LIBRARY_PATH="${SOPHON_SDK_ROOT:-/opt/sophon/libsophon-current}/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$ROOT/tmp"
exec .venv/bin/gunicorn 'app:create_app()' --bind "0.0.0.0:${PORT:-8080}" \
  --workers 1 --worker-class gthread --threads 4 --timeout 1800 --graceful-timeout 20 \
  --pid logs/gunicorn.pid --access-logfile logs/access.log --error-logfile logs/server.log \
  --capture-output --daemon
