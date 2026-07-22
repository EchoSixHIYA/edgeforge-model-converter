import json
import wave
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    box: tuple
    score: float
    class_id: int
    label: str

    def as_dict(self):
        return {"box": list(self.box), "score": self.score, "class_id": self.class_id, "label": self.label}


@dataclass(frozen=True)
class Pose:
    box: tuple
    score: float
    class_id: int
    label: str
    keypoints: tuple

    def as_dict(self):
        return {"box": list(self.box), "score": self.score, "class_id": self.class_id,
                "label": self.label, "keypoints": [list(point) for point in self.keypoints]}


@dataclass(frozen=True)
class OrientedDetection:
    center: tuple
    size: tuple
    angle: float
    points: tuple
    score: float
    class_id: int
    label: str

    def as_dict(self):
        return {"center": list(self.center), "size": list(self.size), "angle": self.angle,
                "points": [list(point) for point in self.points], "score": self.score,
                "class_id": self.class_id, "label": self.label}


@dataclass(frozen=True)
class InstanceMask:
    box: tuple
    score: float
    class_id: int
    label: str
    mask: np.ndarray

    def as_dict(self):
        return {"box": list(self.box), "score": self.score, "class_id": self.class_id,
                "label": self.label, "mask_shape": list(self.mask.shape),
                "mask_area": int(np.count_nonzero(self.mask))}


@dataclass(frozen=True)
class FaceDetection:
    box: tuple
    score: float
    landmarks: tuple
    label: str = "face"
    class_id: int = 0

    def as_dict(self):
        return {"box": list(self.box), "score": self.score,
                "landmarks": [list(point) for point in self.landmarks],
                "label": self.label, "class_id": self.class_id}


@dataclass(frozen=True)
class TextRecognition:
    text: str
    score: float
    token_ids: tuple

    @property
    def label(self):
        return self.text

    def as_dict(self):
        return {"text": self.text, "score": self.score,
                "token_ids": list(self.token_ids)}


@dataclass(frozen=True)
class Classification:
    class_id: int
    label: str
    score: float
    rank: int

    def as_dict(self):
        return {"class_id": self.class_id, "label": self.label,
                "score": self.score, "rank": self.rank}


@dataclass(frozen=True)
class AudioClassification:
    class_id: int
    label: str
    score: float
    rank: int

    def as_dict(self):
        return {"class_id": self.class_id, "label": self.label,
                "score": self.score, "rank": self.rank}


@dataclass(frozen=True)
class SegmentationMask:
    mask: np.ndarray
    labels: tuple

    @property
    def class_ids(self):
        return tuple(int(value) for value in np.unique(self.mask))

    def as_dict(self):
        ids, counts = np.unique(self.mask, return_counts=True)
        return {
            "mask_shape": list(self.mask.shape),
            "classes": [{"class_id": int(class_id),
                         "label": self.labels[int(class_id)] if int(class_id) < len(self.labels) else str(int(class_id)),
                         "pixels": int(count)}
                        for class_id, count in zip(ids, counts)],
        }


@dataclass(frozen=True)
class InteractiveMask:
    mask: np.ndarray
    score: float
    point_coords: tuple
    point_labels: tuple

    def as_dict(self):
        return {
            "mask_shape": list(self.mask.shape), "mask_area": int(np.count_nonzero(self.mask)),
            "score": self.score, "point_coords": [list(point) for point in self.point_coords],
            "point_labels": list(self.point_labels),
        }


class _ModelBase:
    """Stable inference API for an EdgeForge-generated RKNN model."""

    architecture = None

    def __init__(self, model_path=None, *, config_path=None, target=None, device_id=None, core_mask=None):
        packaged = Path(__file__).parent
        self.model_path, resolved_config = _resolve_model_assets(model_path, config_path, packaged)
        self.config_path = resolved_config
        self.config = json.loads(resolved_config.read_text(encoding="utf-8"))
        required = {"architecture", "modality", "task_type", "input_size"}
        missing = required - set(self.config)
        if missing:
            raise ValueError(f"model config is missing required fields: {sorted(missing)}")
        if self.architecture and self.config["architecture"] != self.architecture:
            raise ValueError(
                f"{type(self).__name__} requires architecture={self.architecture}, "
                f"but model bundle declares {self.config['architecture']}")
        self.target, self.device_id, self.core_mask = target, device_id, core_mask
        self.engine = None
        self._lite = False

    def load(self):
        """Load the RKNN file and initialize its runtime, returning ``self``.

        The operation is idempotent.  RKNNLite is selected on-device; the full
        Toolkit2 path requires ``target`` and a connected Rockchip device.
        """
        if self.engine is not None:
            return self
        try:
            from rknnlite.api import RKNNLite
            self.engine, self._lite = RKNNLite(), True
        except ImportError:
            from rknn.api import RKNN
            self.engine = RKNN()
        ret = self.engine.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        if self._lite:
            kwargs = {"core_mask": self.core_mask} if self.core_mask is not None else {}
        else:
            if not self.target:
                raise RuntimeError("完整 RKNN-Toolkit2 加载已编译 RKNN 时必须连接设备并传 target；板端请使用 RKNNLite")
            kwargs = {"target": self.target}
            if self.device_id:
                kwargs["target"] = self.target or self.config["platform"]
                kwargs["device_id"] = self.device_id
        ret = self.engine.init_runtime(**kwargs)
        if ret != 0:
            self.engine.release(); self.engine = None
            raise RuntimeError(f"init_runtime failed: {ret}")
        return self

    def close(self):
        if self.engine is not None:
            self.engine.release(); self.engine = None

    def __enter__(self):
        return self.load()

    def __exit__(self, *_):
        self.close()

    def infer_raw(self, inputs):
        """Run NPU inference and return untouched output tensors in node order.

        A single ndarray is accepted for single-input models.  Pass a list or
        tuple in model-input order for multi-input models.
        """
        self.load()
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        normalized = []
        for value in inputs:
            # RKNNLite requires an explicit batch dimension.  Accept the
            # convenient HWC form as well so callers of infer_raw() do not hit
            # the opaque "need 4dims input" runtime error.
            if isinstance(value, np.ndarray) and value.ndim == 3:
                value = np.expand_dims(value, axis=0)
            normalized.append(value)
        shapes = [list(value.shape) if isinstance(value, np.ndarray) else None
                  for value in normalized]
        try:
            outputs = self.engine.inference(inputs=normalized)
        except Exception as exc:
            raise RuntimeError(
                f"RKNN inference failed for input shapes {shapes}: {exc}. "
                "Check that the board's librknnrt/driver supports the RKNN "
                "Toolkit version used to compile this model."
            ) from exc
        if outputs is None:
            raise RuntimeError(
                f"RKNN inference returned no outputs for input shapes {shapes}. "
                "Check the preceding RKNN runtime log and align the board's "
                "librknnrt/driver with the model compiler version."
            )
        return outputs

    def predict(self, source, *, confidence=0.25, iou=0.45, top_k=5):
        """Preprocess a modality-appropriate input and return structured results."""
        tensor, context = self.preprocess(source)
        outputs = self.infer_raw(tensor)
        return self.postprocess(outputs, context, confidence=confidence, iou=iou, top_k=top_k)

    def preprocess(self, source):
        """Return model input data and task-specific context."""
        modality = self.config.get("modality", "image")
        if modality == "audio":
            return _preprocess_audio(
                source, int(self.config.get("sample_rate", 16000)),
                int(self.config.get("sample_count", self.config["input_size"][0])))
        if modality != "image":
            raise NotImplementedError(
                f"{self.config.get('task_type')} 尚未注册 {modality} 模态的 preprocess()")
        original = _read_image(source)
        if self.config.get("task_type") in {"classification", "text_recognition", "semantic_segmentation"}:
            width, height = self.config["input_size"]
            tensor = cv2.resize(original, (width, height))
            if self.config.get("color_order", "rgb").lower() == "rgb":
                tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
            return np.expand_dims(tensor, axis=0), {"original_shape": list(original.shape[:2])}
        tensor, transform = _letterbox(original, tuple(self.config["input_size"]))
        if self.config.get("color_order", "rgb").lower() == "rgb":
            tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
        return np.expand_dims(tensor, axis=0), {
            "original_shape": list(original.shape[:2]), "transform": list(transform)}

    def postprocess(self, outputs, context, *, confidence=0.25, iou=0.45, top_k=5):
        """Turn raw NPU outputs into results using matching preprocess context.

        ``context`` must come from ``preprocess`` for the same image and output
        batch; it is used to reverse letterboxing into original-image coords.
        """
        architecture = self.config["architecture"]
        task_type = self.config.get("task_type", "object_detection")
        if task_type == "classification":
            return _postprocess_classification(outputs, self.config.get("labels", []), top_k)
        if task_type == "audio_classification" and architecture == "yamnet":
            return _postprocess_yamnet(outputs, self.config.get("labels", []), top_k)
        if task_type == "semantic_segmentation" and architecture == "ppseg":
            return _postprocess_semantic_mask(
                outputs, self.config.get("labels", []), tuple(context["original_shape"]))
        if task_type == "semantic_segmentation" and architecture == "deeplabv3":
            return _postprocess_semantic_mask(
                outputs, self.config.get("labels", []), tuple(context["original_shape"]),
                tuple(self.config.get("logit_resize", self.config["input_size"])))
        if task_type == "text_recognition" and architecture == "lprnet":
            return _postprocess_lprnet(outputs)
        if task_type == "face_detection" and architecture == "retinaface":
            boxes, scores, landmarks = _postprocess_retinaface(
                outputs, tuple(self.config["input_size"]), confidence, iou)
            boxes = _restore_boxes(boxes, tuple(context["transform"]),
                                   tuple(context["original_shape"]))
            landmarks = _restore_face_landmarks(
                landmarks, tuple(context["transform"]), tuple(context["original_shape"]))
            return [FaceDetection(tuple(float(x) for x in box), float(score),
                                  tuple(tuple(float(x) for x in point) for point in points))
                    for box, score, points in zip(boxes, scores, landmarks)]
        if task_type == "pose_estimation" and architecture == "yolov8_pose":
            boxes, scores, keypoints = _postprocess_yolov8_pose(
                outputs, tuple(self.config["input_size"]), confidence, iou)
            boxes = _restore_boxes(boxes, tuple(context["transform"]), tuple(context["original_shape"]))
            keypoints = _restore_keypoints(
                keypoints, tuple(context["transform"]), tuple(context["original_shape"]))
            labels = self.config.get("labels", [])
            label = labels[0] if labels else "person"
            return [Pose(tuple(float(x) for x in box), float(score), 0, label,
                         tuple(tuple(float(x) for x in point) for point in points))
                    for box, score, points in zip(boxes, scores, keypoints)]
        if task_type == "oriented_object_detection" and architecture == "yolov8_obb":
            rects, classes, scores = _postprocess_yolov8_obb(
                outputs, tuple(self.config["input_size"]),
                len(self.config.get("labels", [])) or 15, confidence, iou)
            rects = _restore_oriented(rects, tuple(context["transform"]),
                                      tuple(context["original_shape"]))
            labels = self.config.get("labels", [])
            results = []
            for rect, cls, score in zip(rects, classes, scores):
                center, size, angle = tuple(rect[:2]), tuple(rect[2:4]), float(rect[4])
                points = cv2.boxPoints((center, size, angle * 180.0 / np.pi))
                results.append(OrientedDetection(
                    tuple(float(x) for x in center), tuple(float(x) for x in size), angle,
                    tuple(tuple(float(x) for x in point) for point in points), float(score), int(cls),
                    labels[int(cls)] if int(cls) < len(labels) else str(int(cls))))
            return results
        if task_type == "instance_segmentation" and architecture in {"yolov5_seg", "yolov8_seg"}:
            labels = self.config.get("labels", [])
            if architecture == "yolov5_seg":
                anchors = self.config.get("anchors")
                if not anchors:
                    raise ValueError("YOLOv5 Seg SDK config is missing model-specific anchors")
                boxes, classes, scores, masks = _postprocess_yolov5_seg(
                    outputs, tuple(self.config["input_size"]), len(labels) or 80,
                    anchors, confidence, iou)
            else:
                boxes, classes, scores, masks = _postprocess_yolov8_seg(
                    outputs, tuple(self.config["input_size"]), len(labels) or 80,
                    confidence, iou)
            masks = _restore_masks(masks, tuple(context["transform"]),
                                   tuple(context["original_shape"]))
            boxes = _restore_boxes(boxes, tuple(context["transform"]),
                                   tuple(context["original_shape"]))
            return [InstanceMask(tuple(float(x) for x in box), float(score), int(cls),
                                 labels[int(cls)] if int(cls) < len(labels) else str(int(cls)), mask)
                    for box, cls, score, mask in zip(boxes, classes, scores, masks)]
        if task_type != "object_detection":
            raise NotImplementedError(
                f"{task_type} 必须使用自己的结构化结果和后处理器，不能套用 Detection")
        supported = {"yolov5", "yolov6", "yolov7", "yolov8", "yolov10", "yolo11", "yolox", "ppyoloe"}
        if architecture not in supported:
            raise NotImplementedError(
                f"{architecture} 暂只提供 infer_raw()；该任务类型的结构化后处理器尚未注册")
        labels = self.config.get("labels", [])
        nc = len(labels) or _infer_classes(outputs)
        if architecture == "yolox":
            boxes, classes, scores = _postprocess_yolox(
                outputs, tuple(self.config["input_size"]), nc, confidence, iou)
        elif architecture in {"yolov5", "yolov7"}:
            anchors = self.config.get("anchors")
            if not anchors:
                raise ValueError("YOLOv5 SDK config is missing model-specific anchors")
            boxes, classes, scores = _postprocess_yolov5(
                outputs, tuple(self.config["input_size"]), nc, anchors, confidence, iou)
        elif architecture == "yolov10":
            boxes, classes, scores = _postprocess_dfl(outputs, tuple(self.config["input_size"]), nc, confidence, iou, nms=False)
        else:
            boxes, classes, scores = _postprocess_dfl(outputs, tuple(self.config["input_size"]), nc, confidence, iou, nms=True)
        boxes = _restore_boxes(boxes, tuple(context["transform"]), tuple(context["original_shape"]))
        return [Detection(tuple(float(x) for x in box), float(score), int(cls),
                          labels[int(cls)] if int(cls) < len(labels) else str(int(cls)))
                for box, cls, score in zip(boxes, classes, scores)]

    def draw(self, image, results, output=None):
        if self.config.get("modality", "image") != "image":
            raise NotImplementedError("draw() 仅适用于图像模型；音频分类结果请读取 label/score")
        canvas = _read_image(image).copy()
        if isinstance(results, SegmentationMask):
            palette = np.asarray(self.config.get("palette", []), dtype=np.uint8)
            if len(palette) <= int(results.mask.max(initial=0)):
                raise ValueError("semantic segmentation palette does not cover every class ID")
            colors_bgr = palette[results.mask][..., ::-1]
            canvas = cv2.addWeighted(canvas, .45, colors_bgr, .55, 0)
            if output:
                cv2.imwrite(str(output), canvas)
            return canvas
        for index, result in enumerate(results):
            if isinstance(result, OrientedDetection):
                polygon = np.asarray(result.points, dtype=np.int32)
                cv2.polylines(canvas, [polygon], True, (42, 108, 82), 2)
                origin = tuple(polygon[0])
            elif isinstance(result, FaceDetection):
                x1, y1, x2, y2 = map(int, result.box)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (42, 108, 82), 2)
                for x, y in result.landmarks:
                    cv2.circle(canvas, (int(x), int(y)), 3, (58, 155, 220), -1)
                origin = (x1, max(16, y1 - 6))
            elif isinstance(result, InstanceMask):
                overlay = np.zeros_like(canvas)
                overlay[result.mask] = (58, 155, 220)
                canvas = cv2.addWeighted(canvas, 1.0, overlay, .45, 0)
                x1, y1, x2, y2 = map(int, result.box)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (42, 108, 82), 2)
                origin = (x1, max(16, y1 - 6))
            elif isinstance(result, (Detection, Pose)):
                x1, y1, x2, y2 = map(int, result.box)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (42, 108, 82), 2)
                origin = (x1, max(16, y1 - 6))
                if isinstance(result, Pose):
                    for x, y, score in result.keypoints:
                        if score >= .25:
                            cv2.circle(canvas, (int(x), int(y)), 3, (58, 155, 220), -1)
            else:
                origin = (12, 28 + index * 24)
            cv2.putText(canvas, f"{result.label} {result.score:.2f}", origin,
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (42, 108, 82), 2)
        if output:
            cv2.imwrite(str(output), canvas)
        return canvas


def _read_image(value):
    if isinstance(value, np.ndarray):
        return value
    image = cv2.imread(str(value))
    if image is None:
        raise ValueError(f"cannot read image: {value}")
    return image


def _preprocess_audio(source, sample_rate, sample_count):
    """Decode PCM WAV/array input into one fixed-length float32 waveform."""
    source_rate = sample_rate
    if isinstance(source, tuple) and len(source) == 2:
        waveform, source_rate = source
        waveform = np.asarray(waveform, dtype=np.float32)
    elif isinstance(source, np.ndarray):
        waveform = np.asarray(source, dtype=np.float32)
    else:
        path = Path(source)
        try:
            with wave.open(str(path), "rb") as audio:
                source_rate = audio.getframerate()
                channels = audio.getnchannels()
                width = audio.getsampwidth()
                frames = audio.readframes(audio.getnframes())
        except (OSError, wave.Error) as exc:
            raise ValueError(f"cannot read PCM WAV audio: {source}") from exc
        if width == 1:
            waveform = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 2:
            waveform = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 3:
            raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
            values = (raw[:, 0].astype(np.int32) |
                      (raw[:, 1].astype(np.int32) << 8) |
                      (raw[:, 2].astype(np.int32) << 16))
            values = np.where(values & 0x800000, values - 0x1000000, values)
            waveform = values.astype(np.float32) / 8388608.0
        elif width == 4:
            waveform = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported WAV sample width: {width} bytes")
        if channels > 1:
            waveform = waveform.reshape(-1, channels).mean(axis=1)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if source_rate <= 0:
        raise ValueError(f"invalid audio sample rate: {source_rate}")
    if source_rate != sample_rate and waveform.size:
        output_count = max(1, int(round(waveform.size * sample_rate / source_rate)))
        old_positions = np.arange(waveform.size, dtype=np.float64)
        new_positions = np.arange(output_count, dtype=np.float64) * source_rate / sample_rate
        waveform = np.interp(new_positions, old_positions, waveform).astype(np.float32)
    actual_count = min(waveform.size, sample_count)
    fixed = np.zeros(sample_count, dtype=np.float32)
    fixed[:actual_count] = waveform[:actual_count]
    return np.expand_dims(fixed, axis=0), {
        "source_sample_rate": int(source_rate), "sample_rate": int(sample_rate),
        "source_sample_count": int(waveform.size), "sample_count": int(sample_count),
    }


def _resolve_model_assets(model_path, config_path, packaged):
    if model_path is None:
        model = packaged / "model.rknn"
        config = Path(config_path) if config_path else packaged / "config.json"
    else:
        requested = Path(model_path)
        if requested.is_dir():
            models = sorted(requested.glob("*.rknn"))
            if not models:
                raise ValueError(f"model directory contains no .rknn file: {requested}")
            model = models[0] if len(models) == 1 else requested
            config = Path(config_path) if config_path else requested / "config.json"
        else:
            model = requested
            if config_path:
                config = Path(config_path)
            elif requested.with_suffix(".json").is_file():
                config = requested.with_suffix(".json")
            else:
                config = requested.parent / "config.json"
    if not config.is_file():
        raise ValueError(
            f"model config not found: {config}; keep config.json with the RKNN model "
            "or pass config_path explicitly")
    return model, config


def _postprocess_classification(outputs, labels, top_k):
    if len(outputs) != 1:
        raise ValueError(f"classification expected one output tensor, got {len(outputs)}")
    logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    if not logits.size:
        return []
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    count = max(0, min(int(top_k), logits.size))
    indices = np.argsort(probabilities)[::-1][:count]
    return [Classification(int(class_id),
                           labels[int(class_id)] if int(class_id) < len(labels) else str(int(class_id)),
                           float(probabilities[class_id]), rank)
            for rank, class_id in enumerate(indices, start=1)]


def _postprocess_yamnet(outputs, labels, top_k):
    if not labels:
        raise ValueError("YAMNet SDK config is missing class labels")
    candidates = []
    for output in outputs:
        value = np.asarray(output, dtype=np.float32)
        if value.ndim >= 2 and value.shape[-1] == len(labels):
            candidates.append(value.reshape(-1, len(labels)))
    if len(candidates) != 1:
        shapes = [list(np.asarray(output).shape) for output in outputs]
        raise ValueError(
            f"YAMNet expected exactly one score tensor ending in {len(labels)} classes; got {shapes}")
    scores = candidates[0].mean(axis=0)
    count = max(0, min(int(top_k), scores.size))
    indices = np.argsort(scores)[::-1][:count]
    return [AudioClassification(int(class_id), labels[int(class_id)],
                                float(scores[class_id]), rank)
            for rank, class_id in enumerate(indices, start=1)]


def _postprocess_semantic_mask(outputs, labels, original_shape, logit_resize=None):
    if len(outputs) != 1:
        raise ValueError(f"semantic segmentation expected one output tensor, got {len(outputs)}")
    logits = np.asarray(outputs[0], dtype=np.float32)
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError(f"semantic segmentation expected [1,C,H,W] or [1,H,W,C], got {logits.shape}")
    class_count = len(labels)
    if class_count and logits.shape[1] == class_count:
        pixels = logits[0].transpose(1, 2, 0)
    elif class_count and logits.shape[-1] == class_count:
        pixels = logits[0]
    else:
        raise ValueError(f"semantic output has no axis matching {class_count} configured classes: {logits.shape}")
    if logit_resize and pixels.shape[:2] != (logit_resize[1], logit_resize[0]):
        pixels = cv2.resize(pixels, tuple(logit_resize), interpolation=cv2.INTER_LINEAR)
    mask = pixels.argmax(axis=-1)
    height, width = original_shape
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
    return SegmentationMask(mask.astype(np.int32), tuple(labels))


def _letterbox(image, size):
    width, height = size
    h, w = image.shape[:2]
    ratio = min(width / w, height / h)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    left, top = (width - nw) // 2, (height - nh) // 2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
    return canvas, (ratio, left, top)


def _nchw(value):
    value = np.asarray(value)
    if value.ndim == 4:
        # RKNN/ONNX may return NCHW or NHWC. Spatial axes are square in these
        # YOLO heads, so use their equality instead of comparing magnitudes
        # (80 classes and an 80x80 feature map are otherwise ambiguous).
        if value.shape[1] == value.shape[2] and value.shape[2] != value.shape[3]:
            value = value.transpose(0, 3, 1, 2)
    return value


def _dfl(position):
    position = _nchw(position)
    n, c, h, w = position.shape
    bins = c // 4
    value = position.reshape(n, 4, bins, h, w)
    value = np.exp(value - value.max(axis=2, keepdims=True))
    value /= value.sum(axis=2, keepdims=True)
    return (value * np.arange(bins, dtype=np.float32).reshape(1, 1, bins, 1, 1)).sum(axis=2)


def _decode(position, size):
    position = _nchw(position)
    if position.shape[1] != 4:
        position = _dfl(position)
    h, w = position.shape[2:]
    col, row = np.meshgrid(np.arange(w), np.arange(h))
    grid = np.stack((col, row), axis=0).reshape(1, 2, h, w)
    stride = np.array([size[0] / w, size[1] / h]).reshape(1, 2, 1, 1)
    xy1 = (grid + .5 - position[:, :2]) * stride
    xy2 = (grid + .5 + position[:, 2:]) * stride
    return np.concatenate((xy1, xy2), axis=1).transpose(0, 2, 3, 1).reshape(-1, 4)


def _infer_classes(outputs):
    channels = []
    for output in outputs:
        value = _nchw(output)
        if value.ndim == 4:
            channels.append(value.shape[1])
    candidates = [x for x in channels if x not in {1, 64}]
    if not candidates:
        raise ValueError("cannot infer class count; labels are required")
    return min(candidates)


def _postprocess_dfl(outputs, size, nc, confidence, iou, nms):
    branches, step = 3, len(outputs) // 3
    boxes, probs = [], []
    for i in range(branches):
        boxes.append(_decode(outputs[i * step], size))
        score = _nchw(outputs[i * step + 1]).transpose(0, 2, 3, 1).reshape(-1, nc)
        probs.append(score)
    boxes, probs = np.concatenate(boxes), np.concatenate(probs)
    classes = probs.argmax(axis=1)
    scores = probs[np.arange(len(probs)), classes]
    keep = np.where(scores >= confidence)[0]
    boxes, classes, scores = boxes[keep], classes[keep], scores[keep]
    if not len(scores):
        return boxes, classes, scores
    if nms:
        selected = []
        for cls in np.unique(classes):
            ids = np.where(classes == cls)[0]
            xywh = boxes[ids].copy()
            xywh[:, 2:] -= xywh[:, :2]
            local = cv2.dnn.NMSBoxes(xywh.tolist(), scores[ids].tolist(), confidence, iou)
            if len(local): selected.extend(ids[np.asarray(local).reshape(-1)].tolist())
        keep = np.asarray(selected, dtype=np.int64)
    else:
        keep = np.argsort(scores)[::-1][:300]
    return boxes[keep], classes[keep], scores[keep]


def _decode_yolov5(value, anchors, size, nc):
    value = _nchw(value)
    if value.ndim != 4 or value.shape[0] != 1:
        raise ValueError(f"unexpected YOLOv5 output shape: {value.shape}")
    _, channels, height, width = value.shape
    anchor_count = len(anchors)
    expected = anchor_count * (nc + 5)
    if channels != expected:
        raise ValueError(
            f"YOLOv5 output has {channels} channels; expected {expected} "
            f"for {anchor_count} anchors and {nc} classes")
    position = value.reshape(1, anchor_count, nc + 5, height, width)[0]
    col, row = np.meshgrid(np.arange(width), np.arange(height))
    grid = np.stack((col, row), axis=0).reshape(1, 2, height, width)
    stride = np.array([size[0] / width, size[1] / height], dtype=np.float32).reshape(1, 2, 1, 1)
    anchor_array = np.asarray(anchors, dtype=np.float32).reshape(anchor_count, 2, 1, 1)
    center = (position[:, :2] * 2.0 - 0.5 + grid) * stride
    extent = np.square(position[:, 2:4] * 2.0) * anchor_array
    boxes = np.concatenate((center - extent / 2.0, center + extent / 2.0), axis=1)
    boxes = boxes.transpose(0, 2, 3, 1).reshape(-1, 4)
    objectness = position[:, 4:5].transpose(0, 2, 3, 1).reshape(-1)
    class_probs = position[:, 5:].transpose(0, 2, 3, 1).reshape(-1, nc)
    return boxes, objectness, class_probs


def _postprocess_yolov5(outputs, size, nc, anchors, confidence, iou):
    if len(outputs) != len(anchors):
        raise ValueError(f"YOLOv5 expected {len(anchors)} output branches, got {len(outputs)}")
    decoded = [_decode_yolov5(output, branch_anchors, size, nc)
               for output, branch_anchors in zip(outputs, anchors)]
    boxes = np.concatenate([item[0] for item in decoded])
    objectness = np.concatenate([item[1] for item in decoded])
    probabilities = np.concatenate([item[2] for item in decoded])
    classes = probabilities.argmax(axis=1)
    scores = probabilities[np.arange(len(probabilities)), classes] * objectness
    keep = np.where(scores >= confidence)[0]
    boxes, classes, scores = boxes[keep], classes[keep], scores[keep]
    if not len(scores):
        return boxes, classes, scores
    selected = []
    for class_id in np.unique(classes):
        ids = np.where(classes == class_id)[0]
        xywh = boxes[ids].copy()
        xywh[:, 2:] -= xywh[:, :2]
        local = cv2.dnn.NMSBoxes(xywh.tolist(), scores[ids].tolist(), confidence, iou)
        if len(local):
            selected.extend(ids[np.asarray(local).reshape(-1)].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    return boxes[selected], classes[selected], scores[selected]


def _postprocess_yolox(outputs, size, nc, confidence, iou):
    boxes, probabilities = [], []
    for output in outputs:
        value = _nchw(output)
        if value.ndim != 4 or value.shape[1] != nc + 5:
            raise ValueError(f"unexpected YOLOX output shape: {value.shape}")
        _, _, height, width = value.shape
        col, row = np.meshgrid(np.arange(width), np.arange(height))
        stride = np.array([size[0] / width, size[1] / height], dtype=np.float32).reshape(1, 2, 1, 1)
        center = (value[:, :2] + np.stack((col, row), axis=0).reshape(1, 2, height, width)) * stride
        extent = np.exp(value[:, 2:4]) * stride
        branch_boxes = np.concatenate((center - extent / 2, center + extent / 2), axis=1)
        boxes.append(branch_boxes.transpose(0, 2, 3, 1).reshape(-1, 4))
        objectness = value[:, 4:5].transpose(0, 2, 3, 1).reshape(-1, 1)
        classes = value[:, 5:].transpose(0, 2, 3, 1).reshape(-1, nc)
        probabilities.append(objectness * classes)
    boxes, probabilities = np.concatenate(boxes), np.concatenate(probabilities)
    classes = probabilities.argmax(axis=1)
    scores = probabilities[np.arange(len(probabilities)), classes]
    keep = np.where(scores >= confidence)[0]
    boxes, classes, scores = boxes[keep], classes[keep], scores[keep]
    if not len(scores):
        return boxes, classes, scores
    xywh = boxes.copy()
    xywh[:, 2:] -= xywh[:, :2]
    selected = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), confidence, iou)
    selected = np.asarray(selected).reshape(-1).astype(np.int64) if len(selected) else np.empty(0, dtype=np.int64)
    return boxes[selected], classes[selected], scores[selected]


def _postprocess_yolov8_pose(outputs, size, confidence, iou):
    if len(outputs) != 4:
        raise ValueError(f"YOLOv8 Pose expected 4 output tensors, got {len(outputs)}")
    boxes, scores, source_ids = [], [], []
    offset = 0
    for output in outputs[:3]:
        value = _nchw(output)
        if value.ndim != 4 or value.shape[1] != 65:
            raise ValueError(f"unexpected YOLOv8 Pose detection shape: {value.shape}")
        count = value.shape[2] * value.shape[3]
        boxes.append(_decode(value[:, :64], size))
        logits = value[:, 64:65].transpose(0, 2, 3, 1).reshape(-1)
        scores.append(1.0 / (1.0 + np.exp(-logits)))
        source_ids.append(np.arange(offset, offset + count, dtype=np.int64))
        offset += count
    boxes, scores, source_ids = np.concatenate(boxes), np.concatenate(scores), np.concatenate(source_ids)
    keep = np.where(scores >= confidence)[0]
    boxes, scores, source_ids = boxes[keep], scores[keep], source_ids[keep]
    keypoints = np.asarray(outputs[3], dtype=np.float32)
    if keypoints.shape != (1, 17, 3, offset):
        raise ValueError(f"unexpected YOLOv8 Pose keypoint shape: {keypoints.shape}")
    if not len(scores):
        return boxes, scores, np.empty((0, 17, 3), dtype=np.float32)
    xywh = boxes.copy()
    xywh[:, 2:] -= xywh[:, :2]
    selected = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), confidence, iou)
    selected = np.asarray(selected).reshape(-1).astype(np.int64) if len(selected) else np.empty(0, dtype=np.int64)
    # Advanced indexing already moves the selected detection axis first:
    # (selected, 17 keypoints, x/y/confidence).
    points = keypoints[0, :, :, source_ids[selected]]
    return boxes[selected], scores[selected], points


def _postprocess_yolov8_obb(outputs, size, nc, confidence, iou):
    if len(outputs) != 4:
        raise ValueError(f"YOLOv8 OBB expected 4 output tensors, got {len(outputs)}")
    angle_feature = np.asarray(outputs[3], dtype=np.float32).reshape(-1)
    expected = sum(size_ * size_ for size_ in (80, 40, 20))
    if angle_feature.size != expected:
        raise ValueError(f"unexpected YOLOv8 OBB angle shape: {np.asarray(outputs[3]).shape}")
    rects, probabilities, source_ids = [], [], []
    offset = 0
    for output in outputs[:3]:
        value = _nchw(output)
        if value.ndim != 4 or value.shape[1] != 64 + nc:
            raise ValueError(f"unexpected YOLOv8 OBB detection shape: {value.shape}")
        position = _dfl(value[:, :64])
        _, _, height, width = position.shape
        count = height * width
        col, row = np.meshgrid(np.arange(width), np.arange(height))
        ids = np.arange(offset, offset + count, dtype=np.int64)
        angles = (angle_feature[ids] - .25) * np.pi
        left, top, right, bottom = (position[0, i].reshape(-1) for i in range(4))
        dx, dy = (right - left) / 2.0, (bottom - top) / 2.0
        stride_x, stride_y = size[0] / width, size[1] / height
        center_x = (dx * np.cos(angles) - dy * np.sin(angles) + col.reshape(-1) + .5) * stride_x
        center_y = (dx * np.sin(angles) + dy * np.cos(angles) + row.reshape(-1) + .5) * stride_y
        rects.append(np.stack((center_x, center_y, (left + right) * stride_x,
                               (top + bottom) * stride_y, angles), axis=1))
        logits = value[:, 64:].transpose(0, 2, 3, 1).reshape(-1, nc)
        probabilities.append(1.0 / (1.0 + np.exp(-logits)))
        source_ids.append(ids)
        offset += count
    rects, probabilities = np.concatenate(rects), np.concatenate(probabilities)
    classes = probabilities.argmax(axis=1)
    scores = probabilities[np.arange(len(probabilities)), classes]
    keep = np.where(scores >= confidence)[0]
    rects, classes, scores = rects[keep], classes[keep], scores[keep]
    if not len(scores):
        return rects, classes, scores
    selected = []
    for class_id in np.unique(classes):
        ids = np.where(classes == class_id)[0]
        rotated = [((float(x), float(y)), (float(w), float(h)), float(a * 180.0 / np.pi))
                   for x, y, w, h, a in rects[ids]]
        local = cv2.dnn.NMSBoxesRotated(rotated, scores[ids].tolist(), confidence, iou)
        if len(local):
            selected.extend(ids[np.asarray(local).reshape(-1)].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    return rects[selected], classes[selected], scores[selected]


def _postprocess_yolov8_seg(outputs, size, nc, confidence, iou):
    if len(outputs) != 13:
        raise ValueError(f"YOLOv8 Seg expected 13 output tensors, got {len(outputs)}")
    boxes, probabilities, coefficients = [], [], []
    for branch in range(3):
        base = branch * 4
        boxes.append(_decode(outputs[base], size))
        score = _nchw(outputs[base + 1])
        coeff = _nchw(outputs[base + 3])
        if score.shape[1] != nc or coeff.shape[1] != 32:
            raise ValueError(
                f"unexpected YOLOv8 Seg branch shapes: {score.shape}, {coeff.shape}")
        probabilities.append(score.transpose(0, 2, 3, 1).reshape(-1, nc))
        coefficients.append(coeff.transpose(0, 2, 3, 1).reshape(-1, 32))
    boxes = np.concatenate(boxes)
    probabilities = np.concatenate(probabilities)
    coefficients = np.concatenate(coefficients)
    classes = probabilities.argmax(axis=1)
    scores = probabilities[np.arange(len(probabilities)), classes]
    keep = np.where(scores >= confidence)[0]
    boxes, classes, scores, coefficients = (
        boxes[keep], classes[keep], scores[keep], coefficients[keep])
    if not len(scores):
        return boxes, classes, scores, np.empty((0, size[1], size[0]), dtype=bool)
    selected = []
    for class_id in np.unique(classes):
        ids = np.where(classes == class_id)[0]
        xywh = boxes[ids].copy()
        xywh[:, 2:] -= xywh[:, :2]
        local = cv2.dnn.NMSBoxes(xywh.tolist(), scores[ids].tolist(), confidence, iou)
        if len(local):
            selected.extend(ids[np.asarray(local).reshape(-1)].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    selected = selected[np.argsort(scores[selected])[::-1]][:300]
    boxes, classes, scores, coefficients = (
        boxes[selected], classes[selected], scores[selected], coefficients[selected])
    proto = _nchw(outputs[-1]).astype(np.float32)
    if proto.shape[0] != 1 or proto.shape[1] != 32:
        raise ValueError(f"unexpected YOLOv8 Seg prototype shape: {proto.shape}")
    logits = coefficients @ proto[0].reshape(32, -1)
    masks = 1.0 / (1.0 + np.exp(-logits))
    masks = masks.reshape(-1, proto.shape[2], proto.shape[3])
    resized = np.stack([cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR) for mask in masks])
    col = np.arange(size[0])[None, :]
    row = np.arange(size[1])[:, None]
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        resized[index] *= ((col >= x1) & (col < x2) & (row >= y1) & (row < y2))
    return boxes, classes, scores, resized > .5


def _postprocess_yolov5_seg(outputs, size, nc, anchors, confidence, iou):
    if len(outputs) != 7:
        raise ValueError(f"YOLOv5 Seg expected 7 output tensors, got {len(outputs)}")
    if len(anchors) != 3:
        raise ValueError(f"YOLOv5 Seg expected 3 anchor branches, got {len(anchors)}")
    decoded, coefficients = [], []
    for branch in range(3):
        detected = _decode_yolov5(outputs[branch * 2], anchors[branch], size, nc)
        decoded.append(detected)
        coeff = _nchw(outputs[branch * 2 + 1])
        anchor_count = len(anchors[branch])
        if coeff.shape[1] != anchor_count * 32:
            raise ValueError(f"unexpected YOLOv5 Seg coefficient shape: {coeff.shape}")
        _, _, height, width = coeff.shape
        coefficients.append(coeff.reshape(1, anchor_count, 32, height, width)[0]
                            .transpose(0, 2, 3, 1).reshape(-1, 32))
    boxes = np.concatenate([item[0] for item in decoded])
    objectness = np.concatenate([item[1] for item in decoded])
    probabilities = np.concatenate([item[2] for item in decoded])
    coefficients = np.concatenate(coefficients) * objectness[:, None]
    classes = probabilities.argmax(axis=1)
    scores = probabilities[np.arange(len(probabilities)), classes] * objectness
    keep = np.where(scores >= confidence)[0]
    boxes, classes, scores, coefficients = (
        boxes[keep], classes[keep], scores[keep], coefficients[keep])
    if not len(scores):
        return boxes, classes, scores, np.empty((0, size[1], size[0]), dtype=bool)
    selected = []
    for class_id in np.unique(classes):
        ids = np.where(classes == class_id)[0]
        xywh = boxes[ids].copy()
        xywh[:, 2:] -= xywh[:, :2]
        local = cv2.dnn.NMSBoxes(xywh.tolist(), scores[ids].tolist(), confidence, iou)
        if len(local):
            selected.extend(ids[np.asarray(local).reshape(-1)].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    selected = selected[np.argsort(scores[selected])[::-1]][:300]
    boxes, classes, scores, coefficients = (
        boxes[selected], classes[selected], scores[selected], coefficients[selected])
    proto = _nchw(outputs[-1]).astype(np.float32)
    if proto.shape[0] != 1 or proto.shape[1] != 32:
        raise ValueError(f"unexpected YOLOv5 Seg prototype shape: {proto.shape}")
    logits = coefficients @ proto[0].reshape(32, -1)
    masks = (1.0 / (1.0 + np.exp(-logits))).reshape(-1, proto.shape[2], proto.shape[3])
    resized = np.stack([cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR) for mask in masks])
    col = np.arange(size[0])[None, :]
    row = np.arange(size[1])[:, None]
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        resized[index] *= ((col >= x1) & (col < x2) & (row >= y1) & (row < y2))
    return boxes, classes, scores, resized > .5


def _retinaface_priors(size):
    width, height = size
    priors = []
    for min_sizes, step in zip(((16, 32), (64, 128), (256, 512)), (8, 16, 32)):
        feature_h = int(np.ceil(height / step))
        feature_w = int(np.ceil(width / step))
        for row in range(feature_h):
            for col in range(feature_w):
                for minimum in min_sizes:
                    priors.append(((col + .5) * step / width, (row + .5) * step / height,
                                   minimum / width, minimum / height))
    return np.asarray(priors, dtype=np.float32)


def _postprocess_retinaface(outputs, size, confidence, iou):
    if len(outputs) != 3:
        raise ValueError(f"RetinaFace expected 3 output tensors, got {len(outputs)}")
    loc = np.asarray(outputs[0], dtype=np.float32).reshape(-1, 4)
    probabilities = np.asarray(outputs[1], dtype=np.float32).reshape(-1, 2)
    raw_landmarks = np.asarray(outputs[2], dtype=np.float32).reshape(-1, 10)
    priors = _retinaface_priors(size)
    if len(loc) != len(priors) or len(probabilities) != len(priors) or len(raw_landmarks) != len(priors):
        raise ValueError(
            f"RetinaFace output/prior mismatch: {len(loc)}, {len(probabilities)}, "
            f"{len(raw_landmarks)} vs {len(priors)}")
    center = priors[:, :2] + loc[:, :2] * .1 * priors[:, 2:]
    extent = priors[:, 2:] * np.exp(loc[:, 2:] * .2)
    boxes = np.concatenate((center - extent / 2, center + extent / 2), axis=1)
    scale = np.asarray((size[0], size[1], size[0], size[1]), dtype=np.float32)
    boxes *= scale
    points = []
    for index in range(5):
        points.append(priors[:, :2] + raw_landmarks[:, index * 2:index * 2 + 2] * .1 * priors[:, 2:])
    landmarks = np.stack(points, axis=1)
    landmarks[:, :, 0] *= size[0]
    landmarks[:, :, 1] *= size[1]
    scores = probabilities[:, 1]
    keep = np.where(scores >= confidence)[0]
    boxes, scores, landmarks = boxes[keep], scores[keep], landmarks[keep]
    if not len(scores):
        return boxes, scores, landmarks
    xywh = boxes.copy()
    xywh[:, 2:] -= xywh[:, :2]
    selected = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), confidence, iou)
    selected = np.asarray(selected).reshape(-1).astype(np.int64) if len(selected) else np.empty(0, dtype=np.int64)
    selected = selected[np.argsort(scores[selected])[::-1]]
    return boxes[selected], scores[selected], landmarks[selected]


_LPRNET_CHARS = (
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
    "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "I", "O", "-",
)


def _postprocess_lprnet(outputs):
    if len(outputs) != 1:
        raise ValueError(f"LPRNet expected one output tensor, got {len(outputs)}")
    logits = np.asarray(outputs[0], dtype=np.float32)
    if logits.ndim != 3 or logits.shape[1] != len(_LPRNET_CHARS):
        raise ValueError(f"unexpected LPRNet output shape: {logits.shape}")
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    results = []
    blank = len(_LPRNET_CHARS) - 1
    for batch_logits, batch_probabilities in zip(logits, probabilities):
        best = batch_logits.argmax(axis=0)
        token_ids, token_scores = [], []
        previous = blank
        for timestep, token in enumerate(best):
            token = int(token)
            if token != previous and token != blank:
                token_ids.append(token)
                token_scores.append(float(batch_probabilities[token, timestep]))
            previous = token
        text = "".join(_LPRNET_CHARS[token] for token in token_ids)
        score = float(np.mean(token_scores)) if token_scores else 0.0
        results.append(TextRecognition(text, score, tuple(token_ids)))
    return results


def _restore_boxes(boxes, transform, shape):
    ratio, left, top = transform
    result = boxes.copy()
    if len(result):
        result[:, [0, 2]] = (result[:, [0, 2]] - left) / ratio
        result[:, [1, 3]] = (result[:, [1, 3]] - top) / ratio
        result[:, [0, 2]] = result[:, [0, 2]].clip(0, shape[1] - 1)
        result[:, [1, 3]] = result[:, [1, 3]].clip(0, shape[0] - 1)
    return result


def _restore_keypoints(keypoints, transform, shape):
    ratio, left, top = transform
    result = keypoints.copy()
    if len(result):
        result[:, :, 0] = ((result[:, :, 0] - left) / ratio).clip(0, shape[1] - 1)
        result[:, :, 1] = ((result[:, :, 1] - top) / ratio).clip(0, shape[0] - 1)
    return result


def _restore_oriented(rects, transform, shape):
    ratio, left, top = transform
    result = rects.copy()
    if len(result):
        result[:, 0] = ((result[:, 0] - left) / ratio).clip(0, shape[1] - 1)
        result[:, 1] = ((result[:, 1] - top) / ratio).clip(0, shape[0] - 1)
        result[:, 2:4] /= ratio
    return result


def _restore_masks(masks, transform, shape):
    ratio, left, top = transform
    scaled_width = int(round(shape[1] * ratio))
    scaled_height = int(round(shape[0] * ratio))
    restored = []
    for mask in masks:
        content = mask[top:top + scaled_height, left:left + scaled_width].astype(np.uint8)
        restored.append(cv2.resize(content, (shape[1], shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool))
    return np.stack(restored) if restored else np.empty((0, shape[0], shape[1]), dtype=bool)


def _restore_face_landmarks(landmarks, transform, shape):
    ratio, left, top = transform
    result = landmarks.copy()
    if len(result):
        result[:, :, 0] = ((result[:, :, 0] - left) / ratio).clip(0, shape[1] - 1)
        result[:, :, 1] = ((result[:, :, 1] - top) / ratio).clip(0, shape[0] - 1)
    return result


class YoloV5(_ModelBase):
    architecture = "yolov5"


class YoloV6(_ModelBase):
    architecture = "yolov6"


class YoloV7(_ModelBase):
    architecture = "yolov7"


class YoloV8(_ModelBase):
    architecture = "yolov8"


class YoloV8Pose(_ModelBase):
    architecture = "yolov8_pose"


class YoloV8Obb(_ModelBase):
    architecture = "yolov8_obb"


class YoloV8Seg(_ModelBase):
    architecture = "yolov8_seg"


class YoloV5Seg(_ModelBase):
    architecture = "yolov5_seg"


class YoloX(_ModelBase):
    architecture = "yolox"


class YoloV10(_ModelBase):
    architecture = "yolov10"


class Yolo11(_ModelBase):
    architecture = "yolo11"


class MobileNetV2(_ModelBase):
    architecture = "mobilenet"


class ResNet50(_ModelBase):
    architecture = "resnet"


class RetinaFace(_ModelBase):
    architecture = "retinaface"


class PPYoloE(_ModelBase):
    architecture = "ppyoloe"


class LPRNet(_ModelBase):
    architecture = "lprnet"


class YAMNet(_ModelBase):
    architecture = "yamnet"


class PPLiteSeg(_ModelBase):
    architecture = "ppseg"


class DeepLabV3Plus(_ModelBase):
    architecture = "deeplabv3"


class MobileSAM(_ModelBase):
    """MobileSAM image encoder plus prompt decoder.

    ``point_labels`` follows SAM: 1=foreground point, 0=background point,
    and a box is represented by labels 2 (top-left) and 3 (bottom-right).
    """

    architecture = "mobilesam"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engines = {}

    def _component_paths(self):
        files = self.config.get("model_files")
        if set(files or {}) != {"encoder", "decoder"}:
            raise ValueError("MobileSAM config must declare encoder and decoder model_files")
        base = self.model_path if self.model_path.is_dir() else self.config_path.parent
        paths = {name: base / filename for name, filename in files.items()}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise ValueError(f"MobileSAM model components not found: {missing}")
        return paths

    def load(self):
        if self.engines:
            return self
        for name, path in self._component_paths().items():
            try:
                from rknnlite.api import RKNNLite
                engine, lite = RKNNLite(), True
            except ImportError:
                from rknn.api import RKNN
                engine, lite = RKNN(), False
            ret = engine.load_rknn(str(path))
            if ret != 0:
                self.close()
                raise RuntimeError(f"load_rknn failed for MobileSAM {name}: {ret}")
            if lite:
                runtime = {"core_mask": self.core_mask} if self.core_mask is not None else {}
            else:
                if not self.target:
                    engine.release(); self.close()
                    raise RuntimeError("Toolkit2 runtime requires target and a connected Rockchip device")
                runtime = {"target": self.target}
                if self.device_id:
                    runtime["device_id"] = self.device_id
            ret = engine.init_runtime(**runtime)
            if ret != 0:
                engine.release(); self.close()
                raise RuntimeError(f"init_runtime failed for MobileSAM {name}: {ret}")
            self.engines[name] = engine
        return self

    def close(self):
        for engine in self.engines.values():
            engine.release()
        self.engines.clear()

    def infer_raw(self, inputs, *, component):
        """Run one raw component; ``component`` is ``encoder`` or ``decoder``."""
        if component not in {"encoder", "decoder"}:
            raise ValueError("MobileSAM component must be 'encoder' or 'decoder'")
        self.load()
        values = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        kwargs = {"inputs": values}
        if component == "decoder":
            kwargs["data_format"] = "nchw"
        outputs = self.engines[component].inference(**kwargs)
        if outputs is None:
            raise RuntimeError(f"MobileSAM {component} inference returned no outputs")
        return outputs

    def preprocess(self, source):
        original = _read_image(source)
        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        target = int(self.config["input_size"][0])
        height, width = rgb.shape[:2]
        scale = target / max(height, width)
        resized_width, resized_height = int(width * scale + .5), int(height * scale + .5)
        resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        tensor = np.zeros((target, target, 3), dtype=np.uint8)
        tensor[:resized_height, :resized_width] = resized
        return np.expand_dims(tensor, 0), {
            "original_shape": (height, width), "input_shape": (resized_height, resized_width),
            "coordinate_scale": (resized_width / width, resized_height / height),
        }

    def predict(self, source, *, point_coords, point_labels, mask_input=None, has_mask_input=0.0):
        tensor, context = self.preprocess(source)
        coords = np.asarray(point_coords, dtype=np.float32).reshape(1, -1, 2)
        labels = np.asarray(point_labels, dtype=np.float32).reshape(1, -1)
        expected = int(self.config.get("prompt_count", 2))
        if coords.shape[1] != expected or labels.shape[1] != expected:
            raise ValueError(f"MobileSAM requires exactly {expected} prompt entries")
        original_coords = coords.copy()
        coords[..., 0] *= context["coordinate_scale"][0]
        coords[..., 1] *= context["coordinate_scale"][1]
        embeddings = self.infer_raw(tensor, component="encoder")[0]
        if mask_input is None:
            mask_input = np.zeros((1, 1, 112, 112), dtype=np.float32)
        decoder_inputs = [
            np.asarray(embeddings, dtype=np.float32), coords, labels,
            np.asarray(mask_input, dtype=np.float32),
            np.asarray([has_mask_input], dtype=np.float32),
        ]
        outputs = self.infer_raw(decoder_inputs, component="decoder")
        context.update({"point_coords": original_coords[0], "point_labels": labels[0]})
        return self.postprocess(outputs, context)

    def postprocess(self, outputs, context, **_):
        if len(outputs) != 2:
            raise ValueError(f"MobileSAM decoder expected 2 outputs, got {len(outputs)}")
        scores = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        masks = np.asarray(outputs[1], dtype=np.float32)
        if masks.ndim != 4 or masks.shape[0] != 1 or masks.shape[1] != scores.size:
            raise ValueError(f"unexpected MobileSAM output shapes: {scores.shape}, {masks.shape}")
        target = int(self.config["input_size"][0])
        resized_height, resized_width = context["input_shape"]
        height, width = context["original_shape"]
        index = int(scores.argmax())
        mask = cv2.resize(masks[0, index], (target, target), interpolation=cv2.INTER_LINEAR)
        mask = mask[:resized_height, :resized_width]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR) > 0
        return InteractiveMask(
            mask, float(scores[index]),
            tuple(tuple(float(value) for value in point) for point in context["point_coords"]),
            tuple(int(value) for value in context["point_labels"]),
        )

    def draw(self, image, result, output=None):
        canvas = _read_image(image).copy()
        overlay = np.zeros_like(canvas)
        overlay[result.mask] = (30, 144, 255)
        canvas = cv2.addWeighted(canvas, 1.0, overlay, .5, 0)
        for (x, y), label in zip(result.point_coords, result.point_labels):
            if label in (0, 1):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(canvas, (int(x), int(y)), 5, color, -1)
        box_points = [point for point, label in zip(result.point_coords, result.point_labels) if label in (2, 3)]
        if len(box_points) == 2:
            cv2.rectangle(canvas, tuple(map(int, box_points[0])), tuple(map(int, box_points[1])), (0, 255, 0), 2)
        if output:
            cv2.imwrite(str(output), canvas)
        return canvas
