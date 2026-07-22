"""Extract custom YOLOv5 labels and pixel-space anchors in repo isolation."""
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.repo).resolve()))

import torch
from models.experimental import attempt_load

model = attempt_load(str(Path(args.model).resolve()), device=torch.device("cpu"), inplace=True, fuse=False)
names = getattr(model, "names", {})
if isinstance(names, dict):
    names = [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
head = model.model[-1]
anchors = (head.anchors * head.stride.view(-1, 1, 1)).detach().cpu().tolist()
Path(args.output).write_text(
    json.dumps({"labels": list(names), "anchors": anchors}, ensure_ascii=False, indent=2), encoding="utf-8")
