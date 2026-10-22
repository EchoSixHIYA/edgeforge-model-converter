#!/usr/bin/env python3
"""Local-only Cosmos preflight/compile using the same pipeline and SDK as Web."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.pipeline import Pipeline
from server.tasks import TaskManager
from server.vlm import model_roots, parameters, preflight, resolve_model_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Complete local HF directory; no network downloads")
    parser.add_argument("--allow-root", action="append", help="Explicit allowed model root; defaults to EDGEFORGE_MODEL_ROOTS")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "cosmos-cli"), help="CLI-only task directory, separate from running Web service")
    parser.add_argument("--precision", choices=("w4bf16", "w8bf16", "bf16"), default="w4bf16")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=147456)
    parser.add_argument("--jobs", type=int, choices=(1,), default=1)
    parser.add_argument("--preflight-only", action="store_true", help="Read configs/weight headers/dependencies; NEVER exports or compiles")
    parser.add_argument("--acknowledge-experimental", action="store_true")
    args = parser.parse_args(argv)
    try:
        params = parameters({"precision": args.precision, "seq_length": args.seq_length,
                             "max_pixels": args.max_pixels, "jobs": args.jobs})
        roots = model_roots(args.allow_root if args.allow_root else None)
        source = resolve_model_dir(args.model_dir, roots)
        report = preflight(ROOT, str(source), params, roots, args.output_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if args.preflight_only:
            return 0 if report["ready"] else 2
        if not args.acknowledge_experimental:
            parser.error("Compilation requires --acknowledge-experimental; no real board acceptance is claimed")
        if not report["ready"]:
            return 2
        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        # flock prevents two CLI queues compiling at once in the same directory.
        import fcntl
        with (output / ".queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manager = TaskManager(output, Pipeline(ROOT))
            task = manager.create_vlm({}, source, params, roots)
            print(f"Task {task['id']}: {output / task['id']}", flush=True)
            previous = None
            try:
                while True:
                    task = manager.get(task["id"])
                    message = (task["status"], task["progress"], task["message"])
                    if message != previous:
                        print(f"[{message[0]} {message[1]}%] {message[2]}", flush=True)
                        previous = message
                    if task["status"] in {"completed", "failed", "canceled"}:
                        break
                    time.sleep(0.5)
            except KeyboardInterrupt:
                manager.cancel(task["id"])
                print("Cancellation requested; waiting for compiler process-group termination...", flush=True)
            finally:
                manager.executor.shutdown(wait=True)
            task = manager.get(task["id"])
            if task["status"] == "completed":
                print(json.dumps(manager.tasks[task["id"]]["artifacts"], ensure_ascii=False, indent=2))
                return 0
            return 130 if task["status"] == "canceled" else 1
    except (ValueError, OSError) as exc:
        print(f"Cosmos conversion blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
