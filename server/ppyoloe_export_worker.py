"""Export official PP-YOLOE+ CRN-S PaddleDetection weights to RKNN-friendly ONNX."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import paddle


parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--metadata", required=True)
parser.add_argument("--labels", required=True)
args = parser.parse_args()

repo = Path(args.repo).resolve()
model = Path(args.model).resolve()
output = Path(args.output).resolve()
export_root = output.parent / "paddle_inference"
state = paddle.load(str(model))
stem = state.get("backbone.stem.conv1.conv.weight") if isinstance(state, dict) else None
stem_channels = int(stem.shape[0]) if stem is not None else 0
variants = {16: "s", 24: "m"}
if stem_channels not in variants:
    raise ValueError(
        "PP-YOLOE checkpoint is not an official PP-YOLOE+ CRN-S/CRN-M weight: "
        f"stem channels={stem_channels}")
size = variants[stem_channels]
model_variant = f"ppyoloe_plus_crn_{size}_80e_coco"
config = f"configs/ppyoloe/{model_variant}.yml"

subprocess.run([
    sys.executable, "tools/export_model.py", "-c", config, "-o",
    f"weights={model}", "exclude_nms=True", "trt=True",
    "exclude_post_process=True", "use_gpu=False", "--rknn",
    f"--output_dir={export_root}",
], cwd=repo, check=True)

inference = export_root / model_variant
paddle2onnx = Path(sys.executable).with_name("paddle2onnx")
subprocess.run([
    str(paddle2onnx), "--model_dir", str(inference),
    "--model_filename", "model.pdmodel", "--params_filename", "model.pdiparams",
    "--opset_version", "11", "--save_file", str(output),
], cwd=repo, check=True)

labels = [line.strip() for line in Path(args.labels).read_text(
    encoding="utf-8", errors="ignore").splitlines() if line.strip()]
Path(args.metadata).write_text(json.dumps({
    "labels": labels,
    "model_variant": model_variant,
    "input_size": [640, 640],
}, ensure_ascii=False, indent=2), encoding="utf-8")
