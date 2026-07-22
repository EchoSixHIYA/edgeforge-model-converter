"""Export official Pytorch_Retinaface training checkpoints to static ONNX."""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--metadata", required=True)
parser.add_argument("--imgsz", type=int, default=320)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.repo).resolve()))

from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace


checkpoint = torch.load(args.model, map_location="cpu")
state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
state = OrderedDict((key[7:] if key.startswith("module.") else key, value)
                    for key, value in state.items())
keys = set(state)
if any(key.startswith("body.stage1.") for key in keys):
    variant, cfg = "mobile0.25", dict(cfg_mnet)
elif any(key.startswith("body.conv1.") for key in keys):
    variant, cfg = "resnet50", dict(cfg_re50)
else:
    raise ValueError("checkpoint is not a supported Pytorch_Retinaface MobileNet0.25/ResNet50 state dict")
cfg["pretrain"] = False
model = RetinaFace(cfg=cfg, phase="test")
missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    raise ValueError(f"RetinaFace checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
model.eval()
dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)
with torch.no_grad():
    torch.onnx.export(
        model, dummy, args.output, opset_version=12, input_names=["images"],
        output_names=["loc", "conf", "landmarks"], do_constant_folding=True)
Path(args.metadata).write_text(json.dumps({
    "labels": ["face"], "model_variant": variant, "input_size": [args.imgsz, args.imgsz],
}, ensure_ascii=False, indent=2), encoding="utf-8")
