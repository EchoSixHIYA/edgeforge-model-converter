"""Versioned RKNN Model Zoo catalog and integration capability metadata.

The catalog is intentionally broader than the set of models that may be
submitted.  ``catalog_ready`` means the official example is visible in the
workbench; ``available`` means the complete EdgeForge contract (exporter,
structured SDK post-processing, and real acceptance) has been met.
"""

from .sophon_models import SOPHON_PLATFORMS, SOPHON_MODELS


ROCKCHIP_PLATFORMS = [
    "rk3562", "rk3566", "rk3568", "rk3576", "rk3588", "rv1126b",
    "rv1103", "rv1106", "rv1109", "rv1126", "rk1808", "rk3399pro",
]
PLATFORMS = ROCKCHIP_PLATFORMS + SOPHON_PLATFORMS

VENDORS = [
    {"id": "rockchip", "name": "Rockchip", "backend": "rknn", "format": "RKNN",
     "platforms": ROCKCHIP_PLATFORMS, "available": True},
    {"id": "sophon", "name": "Sophon", "backend": "bmodel", "format": "BModel",
     "platforms": SOPHON_PLATFORMS, "available": True},
]
VENDOR_BY_ID = {item["id"]: item for item in VENDORS}

RKNN_MODEL_ZOO_BASELINE = "vendored-rknn_model_zoo"
_RK2 = ["rk3562", "rk3566", "rk3568", "rk3576", "rk3588", "rv1126b"]
_MCU = ["rv1103", "rv1106"]
_LEGACY = ["rv1109", "rv1126", "rk1808", "rk3399pro"]
_ALL = _RK2 + _MCU + _LEGACY
_STANDARD = _RK2 + _LEGACY


def _item(key, name, category, *, example=None, source="upstream", platforms=None,
          input_size=None, mean=(0, 0, 0), std=(255, 255, 255), precisions=("fp", "i8"),
          color_order="rgb",
          notes="", weight_formats=(), exporter_ready=False, postprocess_ready=False,
          verified=False, blocker="", sophon_platforms=None, sophon_precisions=None,
          sophon_verified=False, sophon_notes="", sophon_input_size=None):
    available = exporter_ready and postprocess_ready and verified
    if available:
        status = "available"
        status_label = "可转换"
    elif exporter_ready and postprocess_ready:
        status = "verification_required"
        status_label = "待真实验收"
    elif exporter_ready:
        status = "sdk_required"
        status_label = "待 SDK 后处理"
    else:
        status = "exporter_required"
        status_label = "待专用导出器"
    vendor_support = {
        "rockchip": {
            "backend": "rknn", "format": "RKNN", "platforms": list(platforms or _STANDARD),
            "precisions": list(precisions), "available": available, "verified": verified,
            "status": status, "blocker": blocker or ("" if available else status_label),
            "notes": notes,
        }
    }
    if sophon_platforms:
        sophon_cfg = SOPHON_MODELS.get(key, {})
        sophon_available = exporter_ready and postprocess_ready
        vendor_support["sophon"] = {
            "backend": "bmodel", "format": "BModel", "platforms": list(sophon_platforms),
            "precisions": list(sophon_precisions or sophon_cfg.get("precisions", ("f16", "i8"))),
            "input_size": list(sophon_input_size or sophon_cfg.get("input_size") or input_size or [640, 640]),
            "available": sophon_available, "verified": bool(sophon_verified),
            "status": "verification_required" if not sophon_verified else "available",
            "blocker": "" if sophon_available else "SOPHGO exporter/backend is not registered",
            "notes": sophon_notes or sophon_cfg.get("notes", ""),
        }
    return {
        "id": key, "name": name, "category": category, "example": example or key,
        "source": source, "platforms": list(platforms or _STANDARD),
        "input_size": input_size, "mean": list(mean), "std": list(std),
        "color_order": color_order,
        "precisions": list(precisions), "notes": notes, "weight_formats": list(weight_formats),
        "catalog_ready": True, "exporter_ready": exporter_ready,
        "postprocess_ready": postprocess_ready, "verified": verified,
        "available": available, "export_ready": available,
        "status": status, "status_label": status_label,
        "blocker": blocker or ("" if available else status_label),
        "vendor_support": vendor_support,
    }


# This is the complete example set in the vendored official RKNN Model Zoo.
# PPOCR is one official example containing both detection and recognition, so it
# remains one workbench entry and documents both precision profiles in notes.
CATALOG = [
    _item("mobilenet", "MobileNet V2", "图像分类", source="pth", platforms=_ALL,
          input_size=[224, 224], mean=(123.675, 116.28, 103.53),
          std=(58.395, 57.12, 57.375), color_order="bgr", weight_formats=(".pth",),
          exporter_ready=True, postprocess_ready=True, verified=True,
          notes="接受 TorchVision MobileNet V2 .pth 状态字典；已完成 RK3588 FP 构建，官方 bell 图 Top-1 为类别 494 chime。"),
    _item("resnet", "ResNet-50", "图像分类", source="pth", input_size=[224, 224],
          mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375),
          color_order="rgb", weight_formats=(".pth",), exporter_ready=True,
          postprocess_ready=True, verified=True,
          notes="接受 TorchVision ResNet-50 .pth 状态字典；已完成 RK3588 FP 构建，官方 dog 图 Top-1 为类别 155 Shih-Tzu。"),
    _item("yolov5", "YOLOv5", "目标检测", source="pt", platforms=_ALL, input_size=[640, 640],
          weight_formats=(".pt",), exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/yolov5 export.py --rknpu；从原始权重提取标签和 anchors；已完成 RK3588 FP 与官方 bus 图验收。",
          sophon_platforms=SOPHON_PLATFORMS),
    _item("yolov6", "YOLOv6", "目标检测", source="pt", input_size=[640, 640],
          weight_formats=(".pt",), exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/yolov6 的 RKNN 专用 9 路导出；显式 YoloV6 支持 4 通道距离解码；已完成 RK3588 FP 与官方 bus 图验收。"),
    _item("yolov7", "YOLOv7", "目标检测", source="pt", input_size=[640, 640],
          weight_formats=(".pt",), exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/yolov7 export.py --rknpu；从 checkpoint 提取标签和专属 anchors；已完成 RK3588 FP 与官方 bus 图验收。",
          sophon_platforms=SOPHON_PLATFORMS),
    _item("yolov8", "YOLOv8", "目标检测", source="pt", input_size=[640, 640],
          weight_formats=(".pt",), exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/ultralytics_yolov8 的 format=rknn 友好导出；已完成 RK3588 FP 构建和官方 bus 图 SDK 对比验收。",
          sophon_platforms=SOPHON_PLATFORMS),
    _item("yolov8_obb", "YOLOv8 OBB", "旋转框检测", example="yolov8_obb", source="pt",
          input_size=[640, 640], precisions=("fp", "i8"), weight_formats=(".pt",),
          exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/ultralytics_yolov8 的 RKNN 友好导出；显式 YoloV8Obb 返回中心/宽高/弧度角/四顶点，并执行旋转 NMS；已完成 RK3588 FP 验收。",
          sophon_platforms=SOPHON_PLATFORMS),
    _item("yolov10", "YOLOv10", "目标检测", source="pt", platforms=_ALL, input_size=[640, 640],
          weight_formats=(".pt",), exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/yolov10 的 RKNN 友好导出；已完成 PT→ONNX→RKNN→SDK 基准验收。",
          sophon_platforms=SOPHON_PLATFORMS),
    _item("yolo11", "YOLO11", "目标检测", source="pt", platforms=_ALL, input_size=[640, 640],
          weight_formats=(".pt",), exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/ultralytics_yolo11 的 format=rknn 友好导出；已完成 RK3588 FP 构建和官方 bus 图 SDK 对比验收。",
          sophon_platforms=SOPHON_PLATFORMS),
    _item("yolox", "YOLOX", "目标检测", source="pth", input_size=[640, 640], std=(1, 1, 1),
          weight_formats=(".pth",), exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/YOLOX export_onnx.py --rknpu；按官方契约保留 0–255 RGB 输入；已完成 RK3588 FP 与官方 bus 图验收。",
          sophon_platforms=SOPHON_PLATFORMS),
    _item("ppyoloe", "PP-YOLOE", "目标检测", source="pdparams", input_size=[640, 640],
          weight_formats=(".pdparams",), exporter_ready=True, postprocess_ready=True,
          verified=True, notes="接受 PaddleDetection PP-YOLOE+ CRN-S/CRN-M 官方 .pdparams；从权重结构识别变体，使用 RK 官方 9 路优化输出与独立 Paddle 2.5.2 导出环境；两个变体均已完成 RK3588 FP 与官方 bus 图验收。"),
    _item("yolo_world", "YOLO-World V2", "开放词汇检测", source="pth", platforms=_RK2,
          input_size=[640, 640], precisions=("fp", "i8"), weight_formats=(".pth",),
          exporter_ready=True, postprocess_ready=True, verified=True,
          notes="接受 YOLO-World V2-S 官方 .pth；从同一 checkpoint 导出检测器与 CLIP 文本塔；显式 YoloWorld 接收最多 80 个自定义类别文本。",
          sophon_platforms=SOPHON_PLATFORMS),
    _item("yolov8_pose", "YOLOv8 Pose", "人体姿态", example="yolov8_pose", source="pt",
          platforms=_RK2, input_size=[640, 640], precisions=("fp", "i8"), weight_formats=(".pt",),
          exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/ultralytics_yolov8 的 RKNN 友好导出；显式 YoloV8Pose 返回人体框与 17×3 关键点；已完成 RK3588 FP 验收。",
          sophon_platforms=SOPHON_PLATFORMS),
    _item("deeplabv3", "DeepLabV3+", "图像分割", source="tensorflow_pb",
          input_size=[513, 513], mean=(127.5, 127.5, 127.5),
          std=(127.5, 127.5, 127.5), precisions=("fp", "i8"),
          weight_formats=(".pb",), exporter_ready=True, postprocess_ready=True,
          verified=True,
          notes="接受 Zoo 官方 DeepLabV3+ MobileNetV2 TensorFlow 冻结图 .pb；裁取官方输入/输出节点并返回 Pascal VOC 语义掩码；RK3588 FP 与官方 test_image.jpg 已通过。"),
    _item("yolov5_seg", "YOLOv5 Seg", "实例分割", example="yolov5_seg", source="pt",
          input_size=[640, 640], precisions=("fp", "i8"), weight_formats=(".pt",),
          exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/yolov5 的 RKNN 分割导出；从原始权重提取 anchors；显式 YoloV5Seg 返回框与原图尺寸实例掩码；已完成 RK3588 FP 验收。"),
    _item("yolov8_seg", "YOLOv8 Seg", "实例分割", example="yolov8_seg", source="pt",
          input_size=[640, 640], precisions=("fp", "i8"), weight_formats=(".pt",),
          exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/ultralytics_yolov8 的 13 路 RKNN 友好导出；显式 YoloV8Seg 返回框与原图尺寸实例掩码；已完成 RK3588 FP 验收。",
          sophon_platforms=SOPHON_PLATFORMS),
    _item("ppseg", "PP-LiteSeg", "图像分割", source="pdparams", input_size=[512, 512],
          mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375),
          weight_formats=(".pdparams",), exporter_ready=True, postprocess_ready=True,
          verified=True,
          notes="接受 PaddleSeg PP-LiteSeg STDC1/STDC2 Cityscapes 原始 .pdparams；显式 PPLiteSeg 返回语义掩码；RK3588 FP 与官方 test.png 已通过。"),
    _item("mobilesam", "MobileSAM", "交互分割", example="mobilesam", source="pt",
          platforms=_RK2, input_size=[448, 448], precisions=("fp",),
          mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375),
          weight_formats=(".pt",), exporter_ready=True, postprocess_ready=True,
          verified=True,
          notes="接受 MobileSAM 官方 mobile_sam.pt；一次生成 encoder/decoder 两个 RKNN；显式 MobileSAM 支持点与框提示并返回 InteractiveMask；已完成 RK3588 FP 官方图片验收。"),
    _item("retinaface", "RetinaFace", "人脸检测", example="RetinaFace", source="pth",
          input_size=[320, 320], mean=(104, 117, 123), std=(1, 1, 1), color_order="bgr",
          precisions=("fp", "i8"), weight_formats=(".pth",),
          exporter_ready=True, postprocess_ready=True, verified=True,
          notes="接受 Pytorch_Retinaface MobileNet0.25/ResNet50 .pth 状态字典；显式 RetinaFace 返回人脸框与 5 个关键点；已完成 RK3588 FP 与官方测试图验收。"),
    _item("lprnet", "LPRNet", "车牌识别", example="LPRNet", source="pth",
          platforms=_ALL, input_size=[94, 24], mean=(127.5, 127.5, 127.5),
          std=(127.5, 127.5, 127.5), color_order="bgr", weight_formats=(".pth",),
          exporter_ready=True, postprocess_ready=True, verified=True,
          notes="接受 sirius-ai/LPRNet_Pytorch 原始 .pth 状态字典；显式 LPRNet 返回 TextRecognition；已完成 RK3588 FP 与官方 test 图验收。"),
    _item("ppocr", "PPOCR Det / Rec", "OCR", example="PPOCR/PPOCR-System",
          source="paddle_inference_zip", input_size=[480, 480], precisions=("fp",),
          mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375),
          weight_formats=(".zip",), exporter_ready=True, postprocess_ready=True,
          verified=True,
          notes="接受含官方 PP-OCRv4 det/rec Paddle 推理 tar 的 ZIP；生成双 RKNN；显式 PPOCR 执行 DB 检测、透视裁剪和 CTC 中文解码；已完成 RK3588 FP 验收。"),
    _item("lite_transformer", "Lite Transformer", "机器翻译", precisions=("fp",),
          blocker="编码器/解码器多模型、词表与自回归解码尚未接入"),
    _item("clip", "CLIP", "图文匹配", example="clip", source="pytorch_bin",
          platforms=_RK2, input_size=[224, 224], precisions=("fp",),
          mean=(122.7709383, 116.7460125, 104.09373615),
          std=(68.5005327, 66.6321579, 70.32316305), weight_formats=(".bin",),
          exporter_ready=True, postprocess_ready=True, verified=True,
          notes="接受 openai/clip-vit-base-patch32 官方 pytorch_model.bin；生成 image/text 双 RKNN；显式 CLIP 内置 tokenizer 与相似度 Softmax；已完成 RK3588 FP 验收。"),
    _item("wav2vec2", "Wav2Vec2", "语音识别", platforms=_RK2, precisions=("fp",),
          blocker="音频原始权重导出、特征处理与 CTC 解码尚未接入"),
    _item("whisper", "Whisper", "语音识别", platforms=_RK2, precisions=("fp",),
          blocker="编码器/解码器多模型、mel 特征与 tokenizer 尚未接入"),
    _item("zipformer", "Zipformer", "语音识别", platforms=_RK2, precisions=("fp",),
          blocker="encoder/decoder/joiner 多模型与流式解码尚未接入"),
    _item("yamnet", "YAMNet", "声音分类", source="savedmodel_zip", platforms=_STANDARD,
          input_size=[48000], mean=(0,), std=(1,), precisions=("fp",),
          weight_formats=(".zip",), exporter_ready=True, postprocess_ready=True,
          verified=True, notes="接受包含 saved_model.pb/variables 的 TensorFlow SavedModel ZIP；显式 YAMNet 处理 3 秒 16 kHz 波形；RK3588 FP 与官方 test.wav Top-1 Animal 已通过。"),
    _item("mms_tts", "MMS TTS", "语音合成", platforms=_RK2, precisions=("fp",),
          blocker="编码器/解码器多模型、tokenizer 与波形生成尚未接入"),
]

_TASK_CONTRACTS = {
    "mobilenet": ("image", "classification", "Classification"),
    "resnet": ("image", "classification", "Classification"),
    "yolov5": ("image", "object_detection", "Detection"),
    "yolov6": ("image", "object_detection", "Detection"),
    "yolov7": ("image", "object_detection", "Detection"),
    "yolov8": ("image", "object_detection", "Detection"),
    "yolov8_obb": ("image", "oriented_object_detection", "OrientedDetection"),
    "yolov10": ("image", "object_detection", "Detection"),
    "yolo11": ("image", "object_detection", "Detection"),
    "yolox": ("image", "object_detection", "Detection"),
    "ppyoloe": ("image", "object_detection", "Detection"),
    "yolo_world": ("image+text", "open_vocabulary_detection", "Detection"),
    "yolov8_pose": ("image", "pose_estimation", "Pose"),
    "deeplabv3": ("image", "semantic_segmentation", "SegmentationMask"),
    "yolov5_seg": ("image", "instance_segmentation", "InstanceMask"),
    "yolov8_seg": ("image", "instance_segmentation", "InstanceMask"),
    "ppseg": ("image", "semantic_segmentation", "SegmentationMask"),
    "mobilesam": ("image+prompt", "interactive_segmentation", "InteractiveMask"),
    "retinaface": ("image", "face_detection", "FaceDetection"),
    "lprnet": ("image", "text_recognition", "TextRecognition"),
    "ppocr": ("image", "ocr_pipeline", "OCRResult"),
    "lite_transformer": ("text", "translation", "Translation"),
    "clip": ("image+text", "image_text_matching", "Similarity"),
    "wav2vec2": ("audio", "speech_recognition", "Transcription"),
    "whisper": ("audio", "speech_recognition", "Transcription"),
    "zipformer": ("audio", "streaming_speech_recognition", "Transcription"),
    "yamnet": ("audio", "audio_classification", "AudioClassification"),
    "mms_tts": ("text", "text_to_speech", "AudioWaveform"),
}

for _model in CATALOG:
    _model["modality"], _model["task_type"], _model["result_type"] = _TASK_CONTRACTS[_model["id"]]

BY_ID = {item["id"]: item for item in CATALOG}
