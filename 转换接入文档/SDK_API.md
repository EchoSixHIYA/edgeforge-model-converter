# EdgeForge SDK API 说明

本文说明转换工作台的通用 Python SDK 公共接口。SDK 只需安装一次；每次转换得到的 `model/` 是可独立复制的模型包，包含 `.rknn + config.json`。SDK 只提供显式模型类，不提供自动识别入口。每种类拥有自己的预处理和后处理契约，不会把 YOLOv10 的逻辑套给 YOLOv8、分类或未来的分割/语音模型。

```python
from edgeforge_sdk import YoloV10

# 必须显式选择模型类；模型包架构不匹配时立即报错。
with YoloV10("models/yolov10") as model:
    results = model.predict("test.jpg")
```

已提供的显式类包括 `YoloV5`、`YoloV5Seg`、`YoloV6`、`YoloV7`、`YoloV8`、`YoloV8Seg`、`YoloV8Pose`、`YoloV8Obb`、`YoloV10`、`Yolo11`、`YoloX`、`PPYoloE`、`PPLiteSeg`、`DeepLabV3Plus`、`MobileNetV2`、`ResNet50`、`RetinaFace`、`LPRNet` 和 `YAMNet`，后续接入模型会继续注册到同一 SDK。切换模型只需替换模型目录，不需要重新安装。只复制裸 `.rknn` 不足以推断自定义标签、颜色表、anchors、颜色顺序等信息，因此必须保留同目录 `config.json`；也可用 `config_path=` 显式指定。

## 1. 安装与运行环境

SDK ZIP 已内置 RKNN-Toolkit-Lite2 2.3.2 的六个 aarch64 wheel，覆盖 CPython 3.7–3.12；一键安装脚本启用当前 SDK 支持的 CPython 3.8–3.12。在 RK356x/RK3576/RK3588/RV1126B 等 64 位 Linux 板端解压后执行：

```bash
bash install.sh
.venv/bin/python example.py
```

脚本会复用已激活的虚拟环境；否则在 SDK 根目录创建 `.venv`，按 Python 版本选择 wheel、校验 MD5，并安装 SDK、NumPy 和 OpenCV。可用 `EDGEFORGE_VENV=/指定目录 bash install.sh` 指定环境。随包 wheel 不替换板卡固件中的 `librknnrt.so` 或 NPU 驱动。

板端使用 `rknnlite.api.RKNNLite`，无需指定 `target`。x86 Linux 调试机使用完整 RKNN-Toolkit2，必须连接真实 Rockchip 设备并传入 `target`；已编译的 RKNN 不能在 x86 模拟器中直接运行。

## 2. 创建与释放

```python
from edgeforge_sdk import YoloV10

model = YoloV10(
    model_path="model/",   # 推荐：包含一个 .rknn 与 config.json 的模型目录
    config_path=None,      # 配置不在模型目录时显式指定
    target="rk3588",      # 板端 RKNNLite 忽略；x86 调试机必填
    device_id=None,        # 多设备时使用 RKNN-Toolkit2 识别的设备 ID
    core_mask=None,        # 板端可传 RKNNLite.NPU_CORE_0 等常量
)
```

推荐使用上下文管理器，它会在退出时释放 NPU 运行时：

```python
with YoloV10("model") as model:
    result = model.predict("test.jpg")
```

也可以手动调用 `close()`。同一个实例不应被多个线程同时调用；线程池场景请为每个工作线程创建独立实例。

## 3. `load()`

```python
model.load() -> YoloV10
```

用途：加载 RKNN 文件并初始化运行时。`predict()` 和 `infer_raw()` 会自动调用它，只有需要预热、提前检查设备连接或把初始化耗时移出请求路径时才需显式调用。

行为：

- 首次调用加载模型并初始化运行时，成功后返回当前实例，便于链式或上下文调用。
- 重复调用是幂等的；运行时已初始化时不会再次加载。
- 板端优先使用 RKNNLite；未安装 RKNNLite 时尝试完整 RKNN-Toolkit2。
- `close()` 后可再次调用 `load()` 重新初始化。

常见异常：

- `ImportError`：未安装 RKNNLite/RKNN-Toolkit2。
- `RuntimeError: load_rknn failed`：模型文件损坏、路径错误或运行时版本不兼容。
- `RuntimeError: init_runtime failed`：设备未连接、`target/device_id/core_mask` 不正确或驱动/运行时不匹配。
- x86 使用完整 Toolkit2 却未传 `target` 时会明确报错。

```python
model = YoloV10("model", target="rk3588", device_id="设备 ID")
try:
    model.load()
    # 此处运行时已经就绪
finally:
    model.close()
```

## 4. `predict()`

```python
model.predict(source, *, confidence=0.25, iou=0.45, top_k=5)
```

用途：面向业务的同步快捷入口，按顺序执行 `preprocess → infer_raw → postprocess`。

参数：

- `source`：图像模型接收图片路径或 OpenCV BGR `H×W×3` 数组；YAMNet 接收 PCM WAV 路径、float32 波形数组，或 `(波形数组, 采样率)`。
- `confidence`：置信度阈值，范围通常为 0～1；低于阈值的候选被过滤。
- `iou`：需要 NMS 的架构使用的交并比阈值；YOLOv10 的端到端/top-k 输出不执行传统 NMS。
- `top_k`：分类模型返回的最高分结果数量；检测模型忽略该参数。

目标检测返回 `Detection` 列表，每项字段为：

- `box`：原图坐标系 `(x1, y1, x2, y2)` 浮点框，已撤销 letterbox 并裁剪到图像边界。
- `score`：浮点置信度。
- `class_id`：从 0 开始的类别编号。
- `label`：从模型元数据提取的类别名；无标签时为类别编号字符串。
- `as_dict()`：生成可 JSON 序列化的字典。

图像分类返回 `Classification` 列表，每项包含 `class_id`、`label`、Softmax 后的 `score`、从 1 开始的 `rank` 和 `as_dict()`。例如：

YAMNet 返回 `AudioClassification` 列表。SDK 将输入转成单声道、重采样至 16 kHz，并截断或补零为 48,000 个样本（3 秒）；后处理对各时间帧的 521 类分数求平均后返回 Top-K。WAV 支持 8/16/24/32 位整数 PCM，不依赖额外音频库：

```python
from edgeforge_sdk import YAMNet

with YAMNet("model") as model:
    events = model.predict("test.wav", top_k=5)
    print([item.as_dict() for item in events])
```

语义分割使用显式 `PPLiteSeg`，返回单个 `SegmentationMask`，而不是检测框列表。`mask` 是还原至原图尺寸的二维类别 ID 数组，`class_ids` 是图中实际出现的类别，`as_dict()` 返回掩码尺寸和各类像素数；`draw()` 使用随模型打包的 Cityscapes 颜色表生成叠加图：

```python
from edgeforge_sdk import PPLiteSeg

with PPLiteSeg("model") as model:
    segmentation = model.predict("test.png")
    print(segmentation.class_ids)
    print(segmentation.as_dict())
    model.draw("test.png", segmentation, "result.png")
```

DeepLabV3+ 使用相同的 `SegmentationMask` 结果契约，但必须使用自己的显式类。它按官方实现将 `65×65×21` logits 双线性放大至 `513×513` 后再取 argmax，不能复用 PP-LiteSeg 的直接 logits 契约：

```python
from edgeforge_sdk import DeepLabV3Plus

with DeepLabV3Plus("model") as model:
    segmentation = model.predict("test_image.jpg")
    model.draw("test_image.jpg", segmentation, "result.png")
```

MobileSAM 是双 RKNN 模型包，`load()` 会同时初始化 `encoder.rknn` 与 `decoder.rknn`。`predict()` 必须显式传入两个提示；标签 `0/1` 表示背景/前景点，`2/3` 表示框的左上角/右下角。返回单个 `InteractiveMask`，其中 `mask` 已恢复到原图尺寸，另含 `score`、`point_coords` 与 `point_labels`：

```python
from edgeforge_sdk import MobileSAM

with MobileSAM("model") as model:
    result = model.predict(
        "picture.jpg",
        point_coords=[(190, 70), (460, 280)],
        point_labels=[2, 3],
    )
    print(result.as_dict())
    model.draw("picture.jpg", result, "result.jpg")
```

需要自行调度时，MobileSAM 的裸推理入口必须指明组件：`infer_raw(image, component="encoder")` 或 `infer_raw(decoder_inputs, component="decoder")`。解码器输入顺序固定为 `image_embeddings, point_coords, point_labels, mask_input, has_mask_input`；`postprocess()` 接收解码器的 `iou_predictions, low_res_masks` 以及同一次 `preprocess()` 的上下文。一般业务应优先使用 `predict()`，以免坐标缩放或模型间张量顺序出错。

PPOCR 同样是一个双 RKNN 模型包，但使用独立的 `PPOCR` 类。`predict()` 先运行文本检测，再对按阅读顺序排列的四点区域进行透视裁剪和识别，返回 `list[OCRResult]`；每项包含原图四点 `polygon`、`text`、`score` 和 `as_dict()`：

```python
from edgeforge_sdk import PPOCR

with PPOCR("model") as model:
    lines = model.predict("test.jpg", confidence=0.5)
    print([line.as_dict() for line in lines])
    model.draw("test.jpg", lines, "result.jpg")
```

分步调用须使用 `infer_raw(inputs, component="det")` 或 `infer_raw(inputs, component="rec")`，并将同一组件传给 `postprocess(..., component=...)`。检测组件返回四点区域，识别组件返回 `(text, score)`；完整的检测到识别调度应优先交给 `predict()`。

CLIP 使用同一模型目录中的 `image.rknn`、`text.rknn` 和 `tokenizer/`。`predict()` 接收一张图片和一个或多个候选文本，返回按概率降序排列的 `list[Similarity]`；每项包含 `text`、Softmax `score`、`rank` 和 `as_dict()`：

```python
from edgeforge_sdk import CLIP

with CLIP("model") as model:
    matches = model.predict(
        "dog_224x224.jpg",
        texts=["a photo of a dog", "a photo of a cat"],
    )
    print([item.as_dict() for item in matches])
```

### YOLO-World

`YoloWorld` 使用同一模型目录中的 `detector.rknn`、`text.rknn` 和 `tokenizer/`。调用者显式传入 1–80 个类别文本，SDK 自动完成 tokenization、逐条文本编码、80 类特征补齐、双输入检测和三尺度 NMS：

```python
from edgeforge_sdk import YoloWorld

with YoloWorld("model") as model:
    detections = model.predict(
        "bus.jpg", texts=["person", "bus", "stop sign"],
        confidence=0.25, iou=0.45,
    )
```

- `load()`：同时加载并初始化文本编码器和检测器两个 RKNN；重复调用安全。
- `predict()`：推荐入口，返回 `list[Detection]`，坐标已还原到原图，`label` 是调用时传入的文本。
- `infer_raw(inputs, component="text")`：输入 `[1,20]` token IDs，返回归一化的 `[1,512]` 文本特征。
- `infer_raw([image, text_features], component="detector")`：输入预处理图像与 `[1,80,512]` 文本特征，返回三组类别概率和距离框。
- `postprocess(outputs, context, confidence=..., iou=...)`：`context` 必须包含 `preprocess()` 产生的缩放信息和本次 `texts`；执行距离框解码、按类 NMS 和原图坐标恢复。
- `draw()`：绘制开放词汇标签、置信度与边界框。

分步调用使用 `component="image"` 或 `component="text"`。文本输入必须由随包 tokenizer 编码并补齐为 20 token；一般业务应使用 `predict()`，由 SDK 完成 tokenizer、两个编码器、特征归一化、相似度缩放和 Softmax。

人脸检测使用显式 `RetinaFace`，返回 `FaceDetection` 列表。除 `box`、`score`、`label`、`class_id` 外，每项的 `landmarks` 固定包含 5 个原图坐标点，顺序为左眼、右眼、鼻尖、左嘴角、右嘴角；`draw()` 会同时绘制人脸框和关键点。

```python
from edgeforge_sdk import RetinaFace

with RetinaFace("models/retinaface") as model:
    faces = model.predict("test.jpg", confidence=0.5)
    print([face.as_dict() for face in faces])
    model.draw("test.jpg", faces, "result.jpg")
```

```python
with MobileNetV2("model") as model:
    top5 = model.predict("test.jpg", top_k=5)
    print([item.as_dict() for item in top5])
```

```python
with YoloV10("model") as model:
    detections = model.predict("test.jpg", confidence=0.35, iou=0.45)
    payload = [item.as_dict() for item in detections]
    model.draw("test.jpg", detections, "result.jpg")
```

空场景返回空列表。图片不可读取时抛出 `ValueError`；运行时错误由 `load()`/`infer_raw()` 原样抛出；架构未注册结构化后处理时抛出 `NotImplementedError`，绝不返回含义不明的裸数组。

## 5. `infer_raw()`

```python
model.infer_raw(inputs) -> list[numpy.ndarray]
```

用途：只执行 NPU 推理，适合已有自己的预处理/后处理、批处理调度或需要查看原始输出张量的调用方。

参数约定：

- 单输入模型可直接传一个 `numpy.ndarray`，SDK 会包装为单元素列表。
- 多输入模型必须按 `config.json`/模型输入节点顺序传 `list` 或 `tuple`。
- 图像 SDK 的标准输入来自 `preprocess()`：`uint8`、NHWC，并显式包含 batch 维。YAMNet 的标准输入为 `[1,48000]` float32 波形。不要跨模型或跨模态复用预处理结果。
- 直接调用时，颜色、形状、数据类型和量化约定均由调用方负责。

返回值始终是按模型输出节点顺序排列的数组列表，不做布局转换、阈值过滤、解码或坐标还原。输出可能是 NCHW 或 NHWC，也可能是量化整数；调用方不能假设固定数量或固定布局。

```python
with YoloV10("model") as model:
    tensor, context = model.preprocess("test.jpg")
    raw_outputs = model.infer_raw(tensor)
    print([(x.shape, x.dtype) for x in raw_outputs])
```

`infer_raw()` 会自动 `load()`。同一实例上的推理调用不是线程安全的。

## 6. `postprocess()`

```python
model.postprocess(outputs, context, *, confidence=0.25, iou=0.45, top_k=5)
```

用途：将 `infer_raw()` 的输出变成稳定的业务对象，适合把 NPU 推理放到独立队列后再处理结果。

参数：

- `outputs`：当前模型一次推理得到的完整原始输出列表，顺序不可改变。
- `context`：同一张图片由 `preprocess()` 返回的上下文。检测任务还包含 `transform=[ratio,left,top]`，用于把 letterbox 坐标恢复到原图；分类任务不需要坐标变换。
- `confidence`、`iou`、`top_k`：语义与 `predict()` 相同。

必须把同一次 `preprocess()` 的 `context` 与对应输出配对；混用其他图片的上下文会产生错误坐标。不要手工构造或依赖未文档化的上下文字段，未来任务类型可扩展该结构。

```python
with YoloV10("model") as model:
    tensor, context = model.preprocess("test.jpg")
    outputs = model.infer_raw(tensor)

# NPU 已释放后，纯 CPU 后处理仍可执行
detections = model.postprocess(outputs, context, confidence=0.25)
```

后处理会识别官方输出的 NCHW/NHWC 布局。YOLOv8/10/11 执行 DFL/网格解码；YOLOv5/7 使用从原始权重提取并写入 `config.json` 的 anchors 执行 grid/anchor 解码；YOLOv6 解码官方 9 路距离输出；YOLOX 使用自身的 grid/stride 和指数宽高解码，且预处理保留 0–255 像素范围。随后执行阈值过滤、需要时的分类别 NMS，以及 letterbox 坐标还原。输出节点数、通道数、anchors 或上下文不符合本模型契约时抛出 `ValueError`；未注册该架构时抛出 `NotImplementedError`。

## 7. 四个接口的关系

```text
load() ───────────────────────────────┐
                                      v
image → preprocess() → infer_raw() → postprocess() → 结构化结果
          \____________ predict() _______________/
```

- 一般业务：导入对应的显式模型类并使用上下文管理器，例如 `with YoloV10("模型目录")`。
- 服务预热：启动时显式 `load()`，停机时 `close()`。
- 自定义流水线：分别调用 `preprocess()`、`infer_raw()`、`postprocess()`。
- 自定义算法：调用 `infer_raw()`，自行解释裸输出；此时结构和兼容性由调用方负责。

## 8. 兼容性边界

- SDK 与生成它的模型、`config.json` 和 RKNN 运行时版本配套，不应混用其他任务的文件。
- `predict()` 只对已完成结构化后处理和真实验收的架构开放。
- 当前公共结构化结果包括检测/分割/姿态、人脸、车牌文本、图像分类，以及声音分类 `AudioClassification`；其余语音和多模型任务会在各自结果类型和验收完成后开放。
- `reference/` 仅用于核对官方实现，业务代码应只导入 `edgeforge_sdk`。

### 任务类型不能混用

| 模型任务 | 输入契约 | `predict()` 结果契约 | 后处理要点 |
|---|---|---|---|
| 图像分类 | 图片 | `list[Classification]` | Softmax、Top-K、类别标签映射 |
| 目标检测 | 图片 | `list[Detection]` | 框解码、阈值、NMS、坐标还原 |
| 语义分割 | 图片 | `SegmentationMask` | 类别 logits/argmax、最近邻恢复原图尺寸、颜色表绘制 |
| 实例分割 | 图片 | `list[InstanceMask]` | 检测框、prototype mask、掩码还原 |
| 姿态估计 | 图片 | `list[Pose]` | 框与关键点解码、关键点坐标还原 |
| OCR | 图片或图片列表 | `list[OCRResult]` | 文本区域检测、裁剪排序、字符解码 |
| 语音识别 | PCM/音频路径 | `Transcription` | 重采样、特征提取、CTC/自回归解码 |
| 声音分类 | PCM/音频路径 | `list[AudioClassification]` | 单声道化、16 kHz 重采样、3 秒波形窗口、帧分数平均与标签 |
| 文字转语音 | 文本 | `AudioWaveform` | tokenizer、声学模型/解码器、波形输出 |

工作台目录中的 `modality`、`task_type`、`result_type` 是机器可读契约。只有该任务的专用 exporter、预处理、结构化结果、后处理和真实验收全部完成，模型才会进入可提交列表。仅能得到裸张量时可以在研发阶段使用 `infer_raw()`，但不能把它标记成完整 SDK。
