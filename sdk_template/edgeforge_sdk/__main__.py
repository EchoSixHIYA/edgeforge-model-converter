import argparse
import json

from . import DeepLabV3Plus, LPRNet, MobileNetV2, PPLiteSeg, PPYoloE, ResNet50, RetinaFace, YAMNet, Yolo11, YoloV5, YoloV5Seg, YoloV6, YoloV7, YoloV8, YoloV8Obb, YoloV8Pose, YoloV8Seg, YoloV10, YoloX

MODELS = {item.__name__: item for item in (YoloV5, YoloV5Seg, YoloV6, YoloV7, YoloV8, YoloV8Obb, YoloV8Pose, YoloV8Seg, YoloV10, Yolo11, YoloX, DeepLabV3Plus, LPRNet, MobileNetV2, PPLiteSeg, PPYoloE, ResNet50, RetinaFace, YAMNet)}

parser = argparse.ArgumentParser(description="EdgeForge model inference")
parser.add_argument("model_class", choices=sorted(MODELS))
parser.add_argument("model")
parser.add_argument("source")
parser.add_argument("--output", default="result.jpg")
parser.add_argument("--confidence", type=float, default=.25)
args = parser.parse_args()

with MODELS[args.model_class](args.model) as model:
    detections = model.predict(args.source, confidence=args.confidence)
    if model.config.get("modality") == "image":
        model.draw(args.source, detections, args.output)
payload = detections.as_dict() if hasattr(detections, "as_dict") else [item.as_dict() for item in detections]
print(json.dumps(payload, ensure_ascii=False, indent=2))
