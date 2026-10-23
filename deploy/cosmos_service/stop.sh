#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$ROOT/.venv/bin/python" - "$ROOT" <<'PY'
import os, signal, sys, time
from pathlib import Path
root = Path(sys.argv[1]).resolve()
pidfile = root / 'logs/gunicorn.pid'
if not pidfile.exists():
    print('No PID file; service is not running.')
    raise SystemExit()
pid = int(pidfile.read_text())
proc = Path('/proc') / str(pid)
if not proc.exists():
    print('Stale PID file; no running process was stopped.')
    raise SystemExit()
if (proc / 'cwd').resolve() != root or b'gunicorn' not in (proc / 'cmdline').read_bytes():
    raise SystemExit('PID identity mismatch; refusing to stop an unrelated process.')
os.kill(pid, signal.SIGTERM)
print('Sent SIGTERM to the verified Cosmos service master:', pid)
for _ in range(60):
    if not proc.exists():
        print('Cosmos service stopped.')
        break
    time.sleep(0.5)
else:
    raise SystemExit('Service is still shutting down; check logs/server.log before restarting.')
PY
