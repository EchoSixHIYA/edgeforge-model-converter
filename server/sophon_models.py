SOPHON_PLATFORMS = ["bm1684x"]
SOPHON_DEFAULT_PRECISIONS = ("f32", "f16", "bf16", "i8")


SOPHON_MODELS = {
    "yolov5": {
        "sample": "YOLOv5", "bmodel_name": "yolov5",
        "input_size": [640, 640], "weight_suffixes": (".pt",),
        "notes": "SOPHGO path uses standard ONNX, TPU-MLIR, and BModel for BM1684X.",
    },
    "yolov7": {
        "sample": "YOLOv7", "bmodel_name": "yolov7",
        "input_size": [640, 640], "weight_suffixes": (".pt",),
        "notes": "SOPHGO path follows sophon-demo YOLOv7 TPU-MLIR scripts.",
    },
    "yolov8": {
        "sample": "YOLOv8_plus_det", "bmodel_name": "yolov8",
        "input_size": [640, 640], "weight_suffixes": (".pt",),
        "notes": "SOPHGO path maps to sophon-demo YOLOv8_plus_det.",
    },
    "yolov8_seg": {
        "sample": "YOLOv8_plus_seg", "bmodel_name": "yolov8s-seg",
        "input_size": [640, 640], "weight_suffixes": (".pt",),
        "notes": "SOPHGO path maps to sophon-demo YOLOv8_plus_seg.",
    },
    "yolov8_pose": {
        "sample": "YOLOv8_pose", "bmodel_name": "yolov8s-pose",
        "input_size": [640, 640], "weight_suffixes": (".pt",),
        "notes": "SOPHGO path follows sophon-demo YOLOv8_pose TPU-MLIR scripts.",
    },
    "yolov8_obb": {
        "sample": "YOLOv8_plus_obb", "bmodel_name": "yolov8s-obb",
        "input_size": [1024, 1024], "weight_suffixes": (".pt",),
        "notes": "SOPHGO path maps to sophon-demo YOLOv8_plus_obb with 1024 input.",
    },
    "yolov8_cls": {
        "sample": "YOLOv8_plus_cls", "bmodel_name": "yolov8s-cls",
        "input_size": [224, 224], "weight_suffixes": (".pt",),
        "notes": "SOPHGO path maps to sophon-demo YOLOv8_plus_cls.",
    },
    "yolov9": {
        "sample": "YOLOv9_det", "bmodel_name": "yolov9",
        "input_size": [640, 640], "weight_suffixes": (".pt",),
        "notes": "SOPHGO path maps to sophon-demo YOLOv9_det.",
    },
    "yolov9_seg": {
        "sample": "YOLOv9_seg", "bmodel_name": "yolov9-seg",
        "input_size": [640, 640], "weight_suffixes": (".pt",),
        "notes": "SOPHGO path maps to sophon-demo YOLOv9_seg.",
    },
    "yolov10": {
        "sample": "YOLOv10", "bmodel_name": "yolov10",
        "input_size": [640, 640], "weight_suffixes": (".pt",),
        "notes": "SOPHGO path follows sophon-demo YOLOv10 TPU-MLIR scripts.",
    },
    "yolo11": {
        "sample": "YOLOv11_det", "bmodel_name": "yolo11",
        "input_size": [640, 640], "weight_suffixes": (".pt",),
        "notes": "SOPHGO path maps to sophon-demo YOLOv11_det.",
    },
    "yolo11_obb": {
        "sample": "YOLOv11_obb", "bmodel_name": "yolo11-obb",
        "input_size": [1024, 1024], "weight_suffixes": (".pt",),
        "notes": "SOPHGO path maps to sophon-demo YOLOv11_obb.",
    },
    "yolov12": {
        "sample": "YOLOv12_det", "bmodel_name": "yolov12",
        "input_size": [640, 640], "weight_suffixes": (".pt",),
        "notes": "SOPHGO path maps to sophon-demo YOLOv12_det.",
    },
    "yolox": {
        "sample": "YOLOX", "bmodel_name": "yolox",
        "input_size": [640, 640], "weight_suffixes": (".pth",),
        "notes": "SOPHGO path follows sophon-demo YOLOX TPU-MLIR scripts.",
    },
    "yolo_world": {
        "sample": "YOLO_world_v2", "bmodel_name": "yoloworld_v2",
        "input_size": [640, 640], "weight_suffixes": (".pth",),
        "precisions": ("f32", "f16", "bf16"),
        "notes": (
            "sophon-demo YOLO_world_v2 currently publishes FP32/FP16 MLIR scripts; "
            "INT8 is held back until official calibration flow is mirrored."
        ),
    },
}


def sophon_model(architecture):
    return SOPHON_MODELS.get(architecture)


def sophon_precisions(architecture):
    cfg = SOPHON_MODELS[architecture]
    return tuple(cfg.get("precisions", SOPHON_DEFAULT_PRECISIONS))
