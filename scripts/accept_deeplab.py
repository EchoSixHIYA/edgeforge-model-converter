"""Validate DeepLabV3+ ONNX segmentation against the Zoo reference overlay."""
import argparse
import hashlib
import json

import cv2
import numpy as np
import onnxruntime as ort


parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("image")
parser.add_argument("reference")
args = parser.parse_args()


def pascal_palette(count=21):
    palette = []
    for value in range(count):
        color, index = [0, 0, 0], value
        for shift in reversed(range(8)):
            for channel in range(3):
                color[channel] |= ((index >> channel) & 1) << shift
            index >>= 3
        palette.append(color)
    return np.asarray(palette, dtype=np.uint8)


bgr = cv2.imread(args.image)
reference = cv2.imread(args.reference)
if bgr is None or reference is None:
    raise SystemExit("cannot read DeepLab acceptance image")
rgb = cv2.cvtColor(cv2.resize(bgr, (513, 513)), cv2.COLOR_BGR2RGB)
tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32)
tensor = (tensor - 127.5) / 127.5
session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
logits = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
if logits.shape[-1] != 21:
    logits = logits.transpose(1, 2, 0)
logits = cv2.resize(logits, (513, 513), interpolation=cv2.INTER_LINEAR)
mask = logits.argmax(axis=-1).astype(np.uint8)
overlay = (rgb.astype(np.float32) * .5 + pascal_palette()[mask] * .5).astype(np.uint8)
overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
ids, counts = np.unique(mask, return_counts=True)
mae = float(np.abs(overlay.astype(np.int16) - reference.astype(np.int16)).mean())
print(json.dumps({
    "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
    "class_pixels": {int(i): int(count) for i, count in zip(ids, counts)},
    "reference_mae": mae,
    "exact_pixel_ratio": float(np.mean(np.all(overlay == reference, axis=2))),
}, ensure_ascii=False))
if mae > 1.0:
    raise SystemExit(f"DeepLab overlay differs from Zoo reference: MAE={mae:.6f}")
