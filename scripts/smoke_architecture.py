"""Run a real original-weight -> ONNX -> compiled model -> SDK smoke test."""
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
parser.add_argument("--vendor", choices=("rockchip", "sophon"), default="rockchip")
parser.add_argument("--platform")
parser.add_argument("--precision", choices=("fp", "f32", "f16", "bf16", "i8"))
parser.add_argument("--input-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
args = parser.parse_args()

work = Path(tempfile.mkdtemp(prefix=f"edgeforge-{args.architecture}-"))
source = work / Path(args.weight).name
source.write_bytes(Path(args.weight).read_bytes())
task = {
    "vendor": args.vendor, "architecture": args.architecture,
    "backend": "bmodel" if args.vendor == "sophon" else "rknn",
    "platform": args.platform or ("bm1684x" if args.vendor == "sophon" else "rk3588"),
    "precision": args.precision or (("f32" if args.platform == "bm1684" else "f16")
                                    if args.vendor == "sophon" else "fp"),
    "source_path": str(source),
    "calibration_path": None, "work_dir": str(work), "acceptance_run": True,
}
if args.input_size is not None:
    task["input_size"] = args.input_size
artifacts = Pipeline(ROOT).run(task, lambda p, m: print(f"{p:3d}% {m}", flush=True))
print(json.dumps({k: {"path": v, "bytes": Path(v).stat().st_size} for k, v in artifacts.items()}, ensure_ascii=False))
