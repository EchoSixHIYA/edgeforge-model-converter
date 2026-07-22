"""Extract YOLOv6 labels without silently treating custom models as COCO."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--official-labels", required=True)
parser.add_argument("--official-sha256", required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.repo).resolve()))

import torch

model_path = Path(args.model)
checkpoint = torch.load(str(model_path), map_location="cpu")
model = checkpoint.get("ema") or checkpoint.get("model") if isinstance(checkpoint, dict) else None

names = checkpoint.get("names") if isinstance(checkpoint, dict) else None
if not names and model is not None:
    names = getattr(model, "names", None)
if isinstance(names, dict):
    names = [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
elif isinstance(names, (list, tuple)):
    names = [str(value) for value in names]
else:
    names = []

digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
source = "checkpoint"
if not names and digest.lower() == args.official_sha256.lower():
    names = [line.strip() for line in Path(args.official_labels).read_text(
        encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    source = "official_coco_fallback"
if not names:
    raise ValueError(
        "YOLOv6 checkpoint does not contain class names. Add a names list/dict to the "
        "checkpoint before upload; custom models must not silently use COCO labels.")

Path(args.output).write_text(json.dumps({
    "labels": names, "labels_source": source, "checkpoint_sha256": digest,
}, ensure_ascii=False, indent=2), encoding="utf-8")
