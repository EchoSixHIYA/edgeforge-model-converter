"""Isolated entrypoint for Rockchip-patched Ultralytics repositories."""
import argparse
import json
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--format", default="onnx")
parser.add_argument("--opset", type=int, default=12)
parser.add_argument("--imgsz", type=int, nargs="+", default=[640], metavar="PIXELS")
parser.add_argument("--metadata", default=None)
args = parser.parse_args()
if len(args.imgsz) not in (1, 2) or any(value <= 0 for value in args.imgsz):
    parser.error("--imgsz expects a positive square size or height and width")
imgsz = args.imgsz[0] if args.imgsz[0] == args.imgsz[-1] else args.imgsz
sys.path.insert(0, str(Path(args.repo).resolve()))

from ultralytics import YOLO

model = YOLO(str(Path(args.model).resolve()))
model.export(format=args.format, opset=args.opset, imgsz=imgsz, simplify=True)
if args.metadata:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        names = [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    Path(args.metadata).write_text(
        json.dumps({"labels": list(names)}, ensure_ascii=False, indent=2), encoding="utf-8")
