import argparse
import json

from . import MobileNetV2, PPYoloE, ResNet50, RetinaFace, Yolo11, YoloV5, YoloV5Seg, YoloV6, YoloV7, YoloV8, YoloV8Obb, YoloV8Pose, YoloV8Seg, YoloV10, YoloX

MODELS = {item.__name__: item for item in (YoloV5, YoloV5Seg, YoloV6, YoloV7, YoloV8, YoloV8Obb, YoloV8Pose, YoloV8Seg, YoloV10, Yolo11, YoloX, MobileNetV2, PPYoloE, ResNet50, RetinaFace)}

parser = argparse.ArgumentParser(description="EdgeForge model inference")
parser.add_argument("model_class", choices=sorted(MODELS))
parser.add_argument("model")
parser.add_argument("image")
parser.add_argument("--output", default="result.jpg")
parser.add_argument("--confidence", type=float, default=.25)
args = parser.parse_args()

with MODELS[args.model_class](args.model) as model:
    detections = model.predict(args.image, confidence=args.confidence)
    model.draw(args.image, detections, args.output)
print(json.dumps([item.as_dict() for item in detections], ensure_ascii=False, indent=2))
