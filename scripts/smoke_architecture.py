"""Run a real original-weight -> friendly ONNX -> RKNN -> SDK smoke test."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.pipeline import Pipeline


parser = argparse.ArgumentParser()
parser.add_argument("architecture")
parser.add_argument("weight")
parser.add_argument("--platform", default="rk3588")
args = parser.parse_args()

work = Path(tempfile.mkdtemp(prefix=f"edgeforge-{args.architecture}-"))
source = work / Path(args.weight).name
source.write_bytes(Path(args.weight).read_bytes())
task = {
    "vendor": "rockchip", "architecture": args.architecture, "backend": "rknn",
    "platform": args.platform, "precision": "fp", "source_path": str(source),
    "calibration_path": None, "work_dir": str(work), "acceptance_run": True,
}
artifacts = Pipeline(ROOT).run(task, lambda p, m: print(f"{p:3d}% {m}", flush=True))
print(json.dumps({k: {"path": v, "bytes": Path(v).stat().st_size} for k, v in artifacts.items()}, ensure_ascii=False))
