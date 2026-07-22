"""Export an uploaded TensorFlow YAMNet SavedModel ZIP to static 3-second ONNX."""
import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--metadata", required=True)
parser.add_argument("--labels", required=True)
args = parser.parse_args()

archive = Path(args.model).resolve()
extract_root = Path(args.output).resolve().parent / "yamnet_saved_model"
extract_root.mkdir(parents=True, exist_ok=True)
total_size = 0
with zipfile.ZipFile(archive) as source:
    for member in source.infolist():
        target = (extract_root / member.filename).resolve()
        if extract_root != target and extract_root not in target.parents:
            raise ValueError(f"unsafe SavedModel ZIP member: {member.filename}")
        total_size += member.file_size
        if total_size > 1024 * 1024 * 1024:
            raise ValueError("SavedModel ZIP expands beyond 1 GiB")
        if not member.is_dir():
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as incoming, target.open("wb") as outgoing:
                while True:
                    chunk = incoming.read(1024 * 1024)
                    if not chunk:
                        break
                    outgoing.write(chunk)

saved_models = list(extract_root.rglob("saved_model.pb"))
if len(saved_models) != 1:
    raise ValueError(f"SavedModel ZIP must contain exactly one saved_model.pb; found {len(saved_models)}")
saved_model = saved_models[0].parent
if not (saved_model / "variables").is_dir():
    raise ValueError("SavedModel ZIP is missing variables/")

temporary = Path(args.output).with_suffix(".dynamic.onnx")
subprocess.run([
    sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(saved_model),
    "--output", str(temporary), "--opset", "12", "--inputs", "waveform:0[48000]",
], check=True)

model = onnx.load(str(temporary))
original_input = model.graph.input[0]
new_input = helper.make_tensor_value_info("audio", onnx.TensorProto.FLOAT, [1, 48000])
shape_name = "edgeforge_audio_shape"
shape = numpy_helper.from_array(np.asarray([48000], dtype=np.int64), shape_name)
reshaped = "edgeforge_audio_reshaped"
model.graph.initializer.append(shape)
model.graph.input.remove(original_input)
model.graph.input.insert(0, new_input)
model.graph.node.insert(0, helper.make_node("Reshape", ["audio", shape_name], [reshaped]))
model.graph.node.insert(1, helper.make_node("Identity", [reshaped], [original_input.name]))
onnx.save(model, args.output)
temporary.unlink(missing_ok=True)

labels = []
for line in Path(args.labels).read_text(encoding="utf-8", errors="ignore").splitlines():
    index, label = line.strip().split(" ", 1)
    if int(index) != len(labels):
        raise ValueError("YAMNet class map indices are not contiguous")
    labels.append(label)
Path(args.metadata).write_text(json.dumps({
    "labels": labels, "sample_rate": 16000, "sample_count": 48000,
    "chunk_seconds": 3,
}, ensure_ascii=False, indent=2), encoding="utf-8")
