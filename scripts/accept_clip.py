"""Validate the packaged CLIP SDK on the RKNN Zoo dog image and text candidates."""
import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort


parser = argparse.ArgumentParser()
parser.add_argument("sdk_zip")
parser.add_argument("image_onnx")
parser.add_argument("text_onnx")
parser.add_argument("image")
args = parser.parse_args()

stage = Path(tempfile.mkdtemp(prefix="edgeforge-clip-accept-"))
with zipfile.ZipFile(args.sdk_zip) as package:
    for member in package.infolist():
        if member.filename.startswith("edgeforge_sdk/") and not member.filename.endswith(".rknn"):
            package.extract(member, stage)
sys.path.insert(0, str(stage))
from edgeforge_sdk import CLIP

model = CLIP()
sessions = {
    "image": ort.InferenceSession(args.image_onnx, providers=["CPUExecutionProvider"]),
    "text": ort.InferenceSession(args.text_onnx, providers=["CPUExecutionProvider"]),
}


def infer_onnx(inputs, *, component):
    value = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
    if component == "image":
        mean = np.asarray(model.config["mean"], dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.asarray(model.config["std"], dtype=np.float32).reshape(1, 1, 1, 3)
        value = ((value.astype(np.float32) - mean) / std).transpose(0, 3, 1, 2)
    session = sessions[component]
    return session.run(None, {session.get_inputs()[0].name: value})


model.infer_raw = infer_onnx
results = model.predict(args.image, texts=["a photo of a dog", "a photo of a cat"])
if results[0].text != "a photo of a dog" or results[0].score < .98:
    raise SystemExit(f"CLIP acceptance failed: {[item.as_dict() for item in results]}")
print("CLIP_ACCEPT_OK", [item.as_dict() for item in results])
