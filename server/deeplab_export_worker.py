"""Convert the official DeepLabV3+ TensorFlow frozen graph to internal ONNX."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import onnx
from onnx import helper


LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tv",
]


def pascal_palette(count):
    palette = []
    for value in range(count):
        color = [0, 0, 0]
        index = value
        for shift in reversed(range(8)):
            for channel in range(3):
                color[channel] |= ((index >> channel) & 1) << shift
            index >>= 3
        palette.append(color)
    return palette


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--metadata", required=True)
args = parser.parse_args()

subprocess.run([
    sys.executable, "-m", "tf2onnx.convert", "--graphdef", str(Path(args.model).resolve()),
    "--inputs", "sub_7:0[1,513,513,3]", "--outputs", "logits/semantic/BiasAdd:0",
    "--opset", "12", "--output", str(Path(args.output).resolve()),
], check=True)
model = onnx.load(args.output)
original_input = model.graph.input[0]
new_input = helper.make_tensor_value_info(
    "images", onnx.TensorProto.FLOAT, [1, 3, 513, 513])
model.graph.input.remove(original_input)
model.graph.input.insert(0, new_input)
model.graph.node.insert(0, helper.make_node(
    "Transpose", ["images"], [original_input.name], perm=[0, 2, 3, 1],
    name="edgeforge_nchw_to_nhwc"))
onnx.checker.check_model(model)
onnx.save(model, args.output)
Path(args.metadata).write_text(json.dumps({
    "labels": LABELS, "palette": pascal_palette(len(LABELS)),
    "dataset": "pascal_voc", "logit_resize": [513, 513], "input_layout": "nchw",
}, ensure_ascii=False, indent=2), encoding="utf-8")
