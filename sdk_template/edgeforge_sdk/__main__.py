import argparse
import json

from . import CLIP, DeepLabV3Plus, LPRNet, MobileNetV2, MobileSAM, PPOCR, PPLiteSeg, PPYoloE, ResNet50, RetinaFace, YAMNet, Yolo11, YoloV5, YoloV5Seg, YoloV6, YoloV7, YoloV8, YoloV8Obb, YoloV8Pose, YoloV8Seg, YoloV10, YoloX

MODELS = {item.__name__: item for item in (YoloV5, YoloV5Seg, YoloV6, YoloV7, YoloV8, YoloV8Obb, YoloV8Pose, YoloV8Seg, YoloV10, Yolo11, YoloX, CLIP, DeepLabV3Plus, LPRNet, MobileNetV2, MobileSAM, PPOCR, PPLiteSeg, PPYoloE, ResNet50, RetinaFace, YAMNet)}

parser = argparse.ArgumentParser(description="EdgeForge model inference")
parser.add_argument("model_class", choices=sorted(MODELS))
parser.add_argument("model")
parser.add_argument("source")
parser.add_argument("--output", default="result.jpg")
parser.add_argument("--confidence", type=float, default=.25)
parser.add_argument("--point", action="append", default=[], metavar="X,Y,LABEL",
                    help="MobileSAM prompt; pass exactly twice")
parser.add_argument("--text", action="append", default=[],
                    help="CLIP text candidate; may be repeated")
args = parser.parse_args()

with MODELS[args.model_class](args.model) as model:
    if args.model_class == "CLIP":
        if not args.text:
            parser.error("CLIP requires at least one --text candidate")
        detections = model.predict(args.source, texts=args.text)
    elif args.model_class == "MobileSAM":
        if len(args.point) != 2:
            parser.error("MobileSAM requires exactly two --point X,Y,LABEL arguments")
        points = [tuple(float(value) for value in item.split(",")) for item in args.point]
        detections = model.predict(
            args.source, point_coords=[item[:2] for item in points],
            point_labels=[int(item[2]) for item in points])
    else:
        detections = model.predict(args.source, confidence=args.confidence)
    if model.config.get("modality") in {"image", "image+prompt"}:
        model.draw(args.source, detections, args.output)
payload = detections.as_dict() if hasattr(detections, "as_dict") else [item.as_dict() for item in detections]
print(json.dumps(payload, ensure_ascii=False, indent=2))
