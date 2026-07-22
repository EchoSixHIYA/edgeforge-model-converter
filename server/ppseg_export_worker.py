"""Export an original PaddleSeg PP-LiteSeg checkpoint to static RKNN-friendly ONNX."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import paddle


CITYSCAPES_LABELS = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]
CITYSCAPES_PALETTE = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    [0, 0, 230], [119, 11, 32],
]


parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--metadata", required=True)
args = parser.parse_args()

repo = Path(args.repo).resolve()
sys.path.insert(0, str(repo))
from paddleseg.cvlibs import Config, SegBuilder

checkpoint = paddle.load(args.model)
candidates = {
    "stdc1": "pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml",
    "stdc2": "pp_liteseg_stdc2_cityscapes_1024x512_scale0.5_160k.yml",
}
selected = None
diagnostics = []
for variant, filename in candidates.items():
    config = Config(str(repo / "configs" / "pp_liteseg" / filename),
                    opts=["model.backbone.pretrained=None"])
    model = SegBuilder(config).model
    expected = model.state_dict()
    matches = sum(key in checkpoint and list(checkpoint[key].shape) == list(value.shape)
                  for key, value in expected.items())
    missing = [key for key, value in expected.items()
               if key not in checkpoint or list(checkpoint[key].shape) != list(value.shape)]
    diagnostics.append(f"{variant}: {matches}/{len(expected)}; first mismatches={missing[:5]}")
    if all(key.endswith("._scale") for key in missing):
        for key, value in checkpoint.items():
            if key in expected and list(value.shape) == list(expected[key].shape):
                expected[key] = value
        model.set_state_dict(expected)
        selected = variant, model
        break
if selected is None:
    raise ValueError("checkpoint is not a complete PP-LiteSeg STDC1/STDC2 Cityscapes state dictionary; " +
                     "; ".join(diagnostics))

variant, model = selected
model.eval()
base = Path(args.output).with_suffix("")
paddle.onnx.export(
    model, str(base), input_spec=[paddle.static.InputSpec([1, 3, 512, 512], "float32", "x")],
    opset_version=11,
)
generated = base.with_suffix(".onnx")
output = Path(args.output)
if generated.resolve() != output.resolve():
    shutil.move(str(generated), str(output))
Path(args.metadata).write_text(json.dumps({
    "labels": CITYSCAPES_LABELS, "palette": CITYSCAPES_PALETTE,
    "model_variant": variant, "dataset": "cityscapes",
}, ensure_ascii=False, indent=2), encoding="utf-8")
