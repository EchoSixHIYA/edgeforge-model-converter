"""Compare a PP-LiteSeg export with the RKNN Model Zoo reference ONNX."""
import argparse
import hashlib
import json

import cv2
import numpy as np
import onnxruntime as ort


parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("reference_model")
parser.add_argument("image")
parser.add_argument("metadata")
args = parser.parse_args()

image = cv2.imread(args.image)
if image is None:
    raise SystemExit(f"cannot read image: {args.image}")
rgb = cv2.cvtColor(cv2.resize(image, (512, 512)), cv2.COLOR_BGR2RGB)
tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32)
tensor = (tensor - np.asarray([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 3, 1, 1))
tensor /= np.asarray([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 3, 1, 1)


def infer(path):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    logits = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    return logits.argmax(axis=1)[0].astype(np.uint8)


actual = infer(args.model)
reference = infer(args.reference_model)
metadata = json.loads(open(args.metadata, encoding="utf-8").read())
ids, counts = np.unique(actual, return_counts=True)
agreement = float(np.mean(actual == reference))
print(json.dumps({
    "pixel_agreement": agreement,
    "mask_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
    "classes": {metadata["labels"][int(i)]: int(count) for i, count in zip(ids, counts)},
}, ensure_ascii=False))
if agreement < .999:
    raise SystemExit(f"PP-LiteSeg export differs from official reference: {agreement:.6f}")
