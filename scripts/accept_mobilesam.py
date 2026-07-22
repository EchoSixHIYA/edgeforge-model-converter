"""Validate the packaged MobileSAM SDK against the RKNN Zoo sample output."""
import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


parser = argparse.ArgumentParser()
parser.add_argument("sdk_zip")
parser.add_argument("encoder")
parser.add_argument("decoder")
parser.add_argument("image")
parser.add_argument("expected")
args = parser.parse_args()

stage = Path(tempfile.mkdtemp(prefix="edgeforge-mobilesam-accept-"))
zipfile.ZipFile(args.sdk_zip).extractall(stage)
sys.path.insert(0, str(stage))
from edgeforge_sdk import MobileSAM

model = MobileSAM(stage / "model")
tensor, context = model.preprocess(args.image)
mean = np.asarray(model.config["mean"], dtype=np.float32).reshape(1, 1, 1, 3)
std = np.asarray(model.config["std"], dtype=np.float32).reshape(1, 1, 1, 3)
encoder_input = ((tensor.astype(np.float32) - mean) / std).transpose(0, 3, 1, 2)
encoder = ort.InferenceSession(args.encoder, providers=["CPUExecutionProvider"])
embedding = encoder.run(None, {encoder.get_inputs()[0].name: encoder_input})[0]

coords = np.asarray([[[190.0, 70.0], [460.0, 280.0]]], dtype=np.float32)
original_coords = coords.copy()
coords[..., 0] *= context["coordinate_scale"][0]
coords[..., 1] *= context["coordinate_scale"][1]
labels = np.asarray([[2.0, 3.0]], dtype=np.float32)
decoder_values = [
    embedding, coords, labels, np.zeros((1, 1, 112, 112), dtype=np.float32),
    np.zeros(1, dtype=np.float32),
]
decoder = ort.InferenceSession(args.decoder, providers=["CPUExecutionProvider"])
outputs = decoder.run(None, {item.name: value for item, value in zip(decoder.get_inputs(), decoder_values)})
context.update({"point_coords": original_coords[0], "point_labels": labels[0]})
result = model.postprocess(outputs, context)

source = cv2.imread(args.image)
expected_image = cv2.imread(args.expected)
blend = cv2.add(source * .5, np.full_like(source, (144, 144, 30)) * .5)
blend_error = np.abs(expected_image.astype(np.float32) - blend.astype(np.float32)).mean(axis=2)
source_error = np.abs(expected_image.astype(np.float32) - source.astype(np.float32)).mean(axis=2)
expected_mask = (blend_error < 3.0) & (source_error > 5.0)
intersection = np.count_nonzero(result.mask & expected_mask)
union = np.count_nonzero(result.mask | expected_mask)
iou = intersection / union
if iou < .90:
    raise SystemExit(f"MobileSAM acceptance failed: mask IoU {iou:.6f} < 0.90")
print("MOBILESAM_ACCEPT_OK", result.as_dict(), "official_overlay_iou", round(iou, 6))
