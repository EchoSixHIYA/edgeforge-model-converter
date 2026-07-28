"""Validate an EdgeForge SOPHGO SDK package or model bundle.

The script performs offline package checks everywhere.  If bmrt_test is
available, or --require-bmrt is passed, it also verifies that each .bmodel can be
loaded by the SOPHGO runtime tools.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def _extract_if_needed(path):
    path = Path(path)
    if path.is_dir():
        return path
    if path.suffix.lower() != ".zip":
        raise SystemExit(f"expected SDK zip or extracted directory, got: {path}")
    dest = Path(tempfile.mkdtemp(prefix="edgeforge-bmodel-smoke-"))
    with zipfile.ZipFile(path) as archive:
        root = dest.resolve()
        for info in archive.infolist():
            target = (dest / info.filename).resolve()
            if root != target and root not in target.parents:
                raise SystemExit(f"unsafe zip member: {info.filename}")
            archive.extract(info, dest)
    return dest


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to read JSON {path}: {exc}") from exc


def _bmodels(root):
    model_dir = root / "model"
    if not model_dir.is_dir():
        model_dir = root
    return sorted(model_dir.glob("*.bmodel"))


def _run_bmrt(model, loops):
    command = ["bmrt_test", "--bmodel", str(model)]
    if loops is not None:
        command.extend(["--loopnum", str(int(loops))])
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-4000:]
        raise SystemExit(f"bmrt_test failed for {model} with code {result.returncode}:\n{tail}")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test an EdgeForge SOPHGO BModel package")
    parser.add_argument("package", help="SDK zip, extracted SDK directory, or model/ bundle")
    parser.add_argument("--require-bmrt", action="store_true",
                        help="fail if bmrt_test is not available")
    parser.add_argument("--loopnum", type=int, default=1,
                        help="loop count passed to bmrt_test when available")
    args = parser.parse_args()

    root = _extract_if_needed(args.package)
    config_path = root / "model" / "config.json"
    if not config_path.is_file():
        config_path = root / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"config.json not found under {root}")
    config = _load_json(config_path)
    if config.get("backend") != "bmodel" or config.get("vendor") != "sophon":
        raise SystemExit(
            f"expected SOPHGO bmodel config, got vendor={config.get('vendor')} backend={config.get('backend')}")

    models = _bmodels(root)
    if not models:
        raise SystemExit(f"no .bmodel files found under {root}")
    if config.get("model_files"):
        missing = [
            name for name in config["model_files"].values()
            if not (config_path.parent / name).is_file()
        ]
        if missing:
            raise SystemExit(f"config model_files missing from bundle: {missing}")

    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        if manifest.get("backend") != "bmodel" or manifest.get("vendor") != "sophon":
            raise SystemExit("manifest does not describe a SOPHGO BModel package")

    bmrt = shutil.which("bmrt_test")
    if bmrt:
        for model in models:
            _run_bmrt(model, args.loopnum)
        runtime = f"bmrt_test OK ({len(models)} model(s))"
    elif args.require_bmrt:
        raise SystemExit("bmrt_test not found in PATH")
    else:
        runtime = "bmrt_test skipped (not found)"

    print(json.dumps({
        "status": "ok",
        "architecture": config.get("architecture"),
        "platform": config.get("platform"),
        "precision": config.get("precision"),
        "bmodels": [str(model) for model in models],
        "runtime": runtime,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
