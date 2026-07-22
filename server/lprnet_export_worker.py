"""Export an original LPRNet state dict with the RKNN Model Zoo network definition."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--zoo-script", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--metadata", required=True)
args = parser.parse_args()

spec = importlib.util.spec_from_file_location("rknn_zoo_lprnet_export", args.zoo_script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

checkpoint = torch.load(args.model, map_location="cpu")
state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
if not isinstance(state, dict):
    raise ValueError("LPRNet checkpoint must contain a PyTorch state dict")
state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
model = module.LPRNet(class_num=68, dropout_rate=0)
model.load_state_dict(state, strict=True)
model.eval()
with torch.no_grad():
    torch.onnx.export(
        model, torch.zeros(1, 3, 24, 94), args.output, opset_version=12,
        input_names=["input"], output_names=["output"], do_constant_folding=True)
Path(args.metadata).write_text(json.dumps({
    "model_variant": "lprnet_chinese_68", "input_size": [94, 24],
}, ensure_ascii=False, indent=2), encoding="utf-8")
