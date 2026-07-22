"""Verify a generated SDK's public pre/postprocess API with ONNX Runtime."""
import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument("sdk_zip")
parser.add_argument("onnx")
parser.add_argument("image")
parser.add_argument("--expect-label", action="append", default=[],
                    help="Require this label in structured results; may be repeated")
parser.add_argument("--min-results", type=int, default=0,
                    help="Require at least this many structured results")
parser.add_argument("--confidence", type=float, default=0.25)
args = parser.parse_args()

dest = Path(tempfile.mkdtemp(prefix="edgeforge-sdk-call-"))
zipfile.ZipFile(args.sdk_zip).extractall(dest)
sys.path.insert(0, str(dest))
import json
import edgeforge_sdk

config = json.loads((dest / "model" / "config.json").read_text(encoding="utf-8"))
class_names = {
    "yolov5": "YoloV5", "yolov5_seg": "YoloV5Seg", "yolov6": "YoloV6", "yolov7": "YoloV7",
    "yolov8": "YoloV8", "yolov8_pose": "YoloV8Pose", "yolov8_obb": "YoloV8Obb",
    "yolov8_seg": "YoloV8Seg",
    "yolov10": "YoloV10",
    "yolo11": "Yolo11", "mobilenet": "MobileNetV2", "resnet": "ResNet50",
    "retinaface": "RetinaFace",
    "ppyoloe": "PPYoloE",
    "yolox": "YoloX",
}
model = getattr(edgeforge_sdk, class_names[config["architecture"]])(dest / "model")
tensor, context = model.preprocess(args.image)
session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
mean = np.asarray(model.config.get("mean", [0, 0, 0]), dtype=np.float32).reshape(1, 3, 1, 1)
std = np.asarray(model.config.get("std", [255, 255, 255]), dtype=np.float32).reshape(1, 3, 1, 1)
input_tensor = (tensor.transpose(0, 3, 1, 2).astype(np.float32) - mean) / std
outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
detections = model.postprocess(outputs, context, confidence=args.confidence)
if len(detections) < args.min_results:
    raise SystemExit(f"SDK validation failed: expected at least {args.min_results} results, got {len(detections)}")
actual_labels = {item.label.strip() for item in detections}
missing_labels = set(args.expect_label) - actual_labels
if missing_labels:
    raise SystemExit(
        f"SDK validation failed: missing labels {sorted(missing_labels)}; got {sorted(actual_labels)}")
print("SDK_CALL_OK", len(detections), detections[0].as_dict() if detections else None)
