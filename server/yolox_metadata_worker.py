"""Identify a standard YOLOX checkpoint variant and extract safe label metadata."""
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
parser.add_argument("--official-s-sha256", required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.repo).resolve()))

import torch
from yolox.exp import get_exp

model_path = Path(args.model)
checkpoint = torch.load(str(model_path), map_location="cpu")
state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
if not isinstance(state, dict):
    raise ValueError("YOLOX checkpoint must contain a model state_dict")

matches = []
for variant in ("yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x"):
    candidate = get_exp(None, variant).get_model().state_dict()
    if candidate.keys() == state.keys() and all(candidate[key].shape == state[key].shape for key in state):
        matches.append(variant)
if len(matches) != 1:
    raise ValueError(f"cannot uniquely identify standard YOLOX variant; matches={matches}")

names = checkpoint.get("names") if isinstance(checkpoint, dict) else None
if isinstance(names, dict):
    names = [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
elif isinstance(names, (list, tuple)):
    names = [str(value) for value in names]
else:
    names = []
digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
source = "checkpoint"
if not names and digest.lower() == args.official_s_sha256.lower():
    names = [line.strip() for line in Path(args.official_labels).read_text(
        encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    source = "official_coco_fallback"
if not names:
    raise ValueError(
        "YOLOX checkpoint does not contain class names. Custom checkpoints must include names; "
        "COCO labels are only used for a hash-verified official checkpoint.")

Path(args.output).write_text(json.dumps({
    "labels": names, "labels_source": source, "model_variant": matches[0],
    "checkpoint_sha256": digest,
}, ensure_ascii=False, indent=2), encoding="utf-8")
