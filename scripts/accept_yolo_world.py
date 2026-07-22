"""Validate the packaged YoloWorld SDK on the RKNN Zoo bus image."""
import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort


parser = argparse.ArgumentParser()
parser.add_argument("sdk_zip")
parser.add_argument("detector_onnx")
parser.add_argument("text_onnx")
parser.add_argument("image")
args = parser.parse_args()

stage = Path(tempfile.mkdtemp(prefix="edgeforge-yolo-world-accept-"))
with zipfile.ZipFile(args.sdk_zip) as package:
    for member in package.infolist():
        if member.filename.startswith("edgeforge_sdk/") and not member.filename.endswith(".rknn"):
            package.extract(member, stage)
sys.path.insert(0, str(stage))
from edgeforge_sdk import YoloWorld  # noqa: E402

model = YoloWorld()
sessions = {
    "detector": ort.InferenceSession(args.detector_onnx, providers=["CPUExecutionProvider"]),
    "text": ort.InferenceSession(args.text_onnx, providers=["CPUExecutionProvider"]),
}


def infer_onnx(inputs, *, component):
    values = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
    session = sessions[component]
    if component == "detector":
        values[0] = values[0].transpose(0, 3, 1, 2)
    return session.run(None, {item.name: value for item, value in zip(session.get_inputs(), values)})


model.infer_raw = infer_onnx
results = model.predict(args.image, texts=model.config["labels"])
bus = [item for item in results if item.label == "bus"]
persons = [item for item in results if item.label == "person"]
if not bus or bus[0].score < .85 or len(persons) < 4:
    raise SystemExit(f"YOLO-World acceptance failed: {[item.as_dict() for item in results[:10]]}")
print("YOLO_WORLD_ACCEPT_OK", [item.as_dict() for item in results[:10]])
