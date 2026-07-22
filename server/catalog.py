"""Versioned RKNN Model Zoo catalog and integration capability metadata.

The catalog is intentionally broader than the set of models that may be
submitted.  ``catalog_ready`` means the official example is visible in the
workbench; ``available`` means the complete EdgeForge contract (exporter,
structured SDK post-processing, and real acceptance) has been met.
"""

PLATFORMS = [
    "rk3562", "rk3566", "rk3568", "rk3576", "rk3588", "rv1126b",
    "rv1103", "rv1106", "rv1109", "rv1126", "rk1808", "rk3399pro",
]

VENDORS = [
    {"id": "rockchip", "name": "Rockchip", "backend": "rknn", "format": "RKNN", "available": True},
    {"id": "sophon", "name": "Sophon", "backend": "bmodel", "format": "BModel", "available": False},
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
          verified=False, blocker=""):
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
          notes="使用 airockchip/yolov5 export.py --rknpu；从原始权重提取标签和 anchors；已完成 RK3588 FP 与官方 bus 图验收。"),
    _item("yolov6", "YOLOv6", "目标检测", source="pt", input_size=[640, 640],
          weight_formats=(".pt",), exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/yolov6 的 RKNN 专用 9 路导出；显式 YoloV6 支持 4 通道距离解码；已完成 RK3588 FP 与官方 bus 图验收。"),
    _item("yolov7", "YOLOv7", "目标检测", source="pt", input_size=[640, 640],
          weight_formats=(".pt",), exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/yolov7 export.py --rknpu；从 checkpoint 提取标签和专属 anchors；已完成 RK3588 FP 与官方 bus 图验收。"),
    _item("yolov8", "YOLOv8", "目标检测", source="pt", input_size=[640, 640],
          weight_formats=(".pt",), exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/ultralytics_yolov8 的 format=rknn 友好导出；已完成 RK3588 FP 构建和官方 bus 图 SDK 对比验收。"),
    _item("yolov8_obb", "YOLOv8 OBB", "旋转框检测", example="yolov8_obb", source="pt",
          input_size=[640, 640], precisions=("fp", "i8"), weight_formats=(".pt",),
          exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/ultralytics_yolov8 的 RKNN 友好导出；显式 YoloV8Obb 返回中心/宽高/弧度角/四顶点，并执行旋转 NMS；已完成 RK3588 FP 验收。"),
    _item("yolov10", "YOLOv10", "目标检测", source="pt", platforms=_ALL, input_size=[640, 640],
          weight_formats=(".pt",), exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/yolov10 的 RKNN 友好导出；已完成 PT→ONNX→RKNN→SDK 基准验收。"),
    _item("yolo11", "YOLO11", "目标检测", source="pt", platforms=_ALL, input_size=[640, 640],
          weight_formats=(".pt",), exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/ultralytics_yolo11 的 format=rknn 友好导出；已完成 RK3588 FP 构建和官方 bus 图 SDK 对比验收。"),
    _item("yolox", "YOLOX", "目标检测", source="pth", input_size=[640, 640], std=(1, 1, 1),
          weight_formats=(".pth",), exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/YOLOX export_onnx.py --rknpu；按官方契约保留 0–255 RGB 输入；已完成 RK3588 FP 与官方 bus 图验收。"),
    _item("ppyoloe", "PP-YOLOE", "目标检测", input_size=[640, 640],
          blocker="需接入 PaddleDetection 原始权重导出、检测后处理和真实验收"),
    _item("yolo_world", "YOLO-World V2", "开放词汇检测", platforms=_RK2, input_size=[640, 640],
          blocker="双模型/文本提示输入尚未接入，需一并打包 tokenizer 与后处理"),
    _item("yolov8_pose", "YOLOv8 Pose", "人体姿态", example="yolov8_pose", source="pt",
          platforms=_RK2, input_size=[640, 640], precisions=("fp", "i8"), weight_formats=(".pt",),
          exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/ultralytics_yolov8 的 RKNN 友好导出；显式 YoloV8Pose 返回人体框与 17×3 关键点；已完成 RK3588 FP 验收。"),
    _item("deeplabv3", "DeepLabV3+", "图像分割", input_size=[513, 513],
          blocker="需接入原始 TensorFlow 权重导出、语义掩码后处理和真实验收"),
    _item("yolov5_seg", "YOLOv5 Seg", "实例分割", example="yolov5_seg", source="pt",
          input_size=[640, 640], precisions=("fp", "i8"), weight_formats=(".pt",),
          exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/yolov5 的 RKNN 分割导出；从原始权重提取 anchors；显式 YoloV5Seg 返回框与原图尺寸实例掩码；已完成 RK3588 FP 验收。"),
    _item("yolov8_seg", "YOLOv8 Seg", "实例分割", example="yolov8_seg", source="pt",
          input_size=[640, 640], precisions=("fp", "i8"), weight_formats=(".pt",),
          exporter_ready=True, postprocess_ready=True, verified=True,
          notes="使用 airockchip/ultralytics_yolov8 的 13 路 RKNN 友好导出；显式 YoloV8Seg 返回框与原图尺寸实例掩码；已完成 RK3588 FP 验收。"),
    _item("ppseg", "PP-LiteSeg", "图像分割", input_size=[512, 512],
          blocker="需接入 PaddleSeg 原始权重导出、语义掩码后处理和真实验收"),
    _item("mobilesam", "MobileSAM", "交互分割", platforms=_RK2, input_size=[448, 448], precisions=("fp",),
          blocker="编码器/解码器多模型任务、提示输入与掩码后处理尚未接入"),
    _item("retinaface", "RetinaFace", "人脸检测", example="RetinaFace", input_size=[320, 320],
          precisions=("i8",), blocker="需接入原始权重导出、人脸框/关键点后处理和真实验收"),
    _item("lprnet", "LPRNet", "车牌识别", example="LPRNet", platforms=_ALL, input_size=[94, 24],
          blocker="需接入原始权重导出、字符表 CTC 解码和真实验收"),
    _item("ppocr", "PPOCR Det / Rec", "OCR", example="PPOCR", input_size=[480, 480],
          notes="官方示例同时包含检测（FP16/INT8）与识别（FP16）模型。",
          blocker="双模型上传、PaddleOCR 导出、DB 后处理与文字解码尚未接入"),
    _item("lite_transformer", "Lite Transformer", "机器翻译", precisions=("fp",),
          blocker="编码器/解码器多模型、词表与自回归解码尚未接入"),
    _item("clip", "CLIP", "图文匹配", platforms=_RK2, precisions=("fp",),
          blocker="图像/文本双模型、tokenizer 与相似度后处理尚未接入"),
    _item("wav2vec2", "Wav2Vec2", "语音识别", platforms=_RK2, precisions=("fp",),
          blocker="音频原始权重导出、特征处理与 CTC 解码尚未接入"),
    _item("whisper", "Whisper", "语音识别", platforms=_RK2, precisions=("fp",),
          blocker="编码器/解码器多模型、mel 特征与 tokenizer 尚未接入"),
    _item("zipformer", "Zipformer", "语音识别", platforms=_RK2, precisions=("fp",),
          blocker="encoder/decoder/joiner 多模型与流式解码尚未接入"),
    _item("yamnet", "YAMNet", "声音分类", platforms=_RK2, precisions=("fp",),
          blocker="音频原始权重导出、波形预处理与标签后处理尚未接入"),
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
    "mobilesam": ("image+prompt", "interactive_segmentation", "SegmentationMask"),
    "retinaface": ("image", "face_detection", "FaceDetection"),
    "lprnet": ("image", "text_recognition", "TextRecognition"),
    "ppocr": ("image", "ocr_pipeline", "OCRResult"),
    "lite_transformer": ("text", "translation", "Translation"),
    "clip": ("image+text", "image_text_matching", "Similarity"),
    "wav2vec2": ("audio", "speech_recognition", "Transcription"),
    "whisper": ("audio", "speech_recognition", "Transcription"),
    "zipformer": ("audio", "streaming_speech_recognition", "Transcription"),
    "yamnet": ("audio", "audio_classification", "Classification"),
    "mms_tts": ("text", "text_to_speech", "AudioWaveform"),
}

for _model in CATALOG:
    _model["modality"], _model["task_type"], _model["result_type"] = _TASK_CONTRACTS[_model["id"]]

BY_ID = {item["id"]: item for item in CATALOG}
