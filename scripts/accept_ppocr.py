"""Run the packaged PPOCR pipeline on the RKNN Zoo system sample with ONNX Runtime."""
import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort


parser = argparse.ArgumentParser()
parser.add_argument("sdk_zip")
parser.add_argument("det_onnx")
parser.add_argument("rec_onnx")
parser.add_argument("image")
args = parser.parse_args()

stage = Path(tempfile.mkdtemp(prefix="edgeforge-ppocr-accept-"))
zipfile.ZipFile(args.sdk_zip).extractall(stage)
sys.path.insert(0, str(stage))
from edgeforge_sdk import PPOCR

model = PPOCR(stage / "model")
sessions = {
    "det": ort.InferenceSession(args.det_onnx, providers=["CPUExecutionProvider"]),
    "rec": ort.InferenceSession(args.rec_onnx, providers=["CPUExecutionProvider"]),
}


def infer_onnx(inputs, *, component):
    value = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
    if component == "det":
        mean = np.asarray(model.config["mean"], dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.asarray(model.config["std"], dtype=np.float32).reshape(1, 1, 1, 3)
        value = (value.astype(np.float32) - mean) / std
    value = value.transpose(0, 3, 1, 2).astype(np.float32)
    session = sessions[component]
    return session.run(None, {session.get_inputs()[0].name: value})


model.infer_raw = infer_onnx
results = model.predict(args.image, confidence=.5)
texts = [item.text for item in results]
expected = ["纯臻营养护发素", "产品信息/参数", "每瓶22元，1000瓶起订）", "【品名】：纯臻营养护发素"]
missing = [text for text in expected if text not in texts]
if missing:
    raise SystemExit(f"PPOCR acceptance failed; missing {missing}; got {texts}")
print("PPOCR_ACCEPT_OK", len(results), [item.as_dict() for item in results[:3]])
