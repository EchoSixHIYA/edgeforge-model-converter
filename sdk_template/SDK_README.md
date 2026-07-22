# EdgeForge 通用模型 SDK

`edgeforge_sdk` 只需安装一次。以后从任意转换包复制 `model/` 目录（其中包含一个 `.rknn` 和 `config.json`），传给对应的显式模型类，无需再次安装。SDK 不提供自动识别入口；`reference/` 仅用于核对上游官方实现。

```python
from edgeforge_sdk import YoloV10

# 必须显式选择；模型架构不匹配时立即报错
model = YoloV10("/opt/models/yolov10")
```

显式类包括 `YoloV5`、`YoloV5Seg`、`YoloV6`、`YoloV7`、`YoloV8`、`YoloV8Seg`、`YoloV8Pose`、`YoloV8Obb`、`YoloV10`、`Yolo11`、`YoloX`、`PPYoloE`、`MobileNetV2`、`ResNet50`、`RetinaFace` 和 `LPRNet`。不同类使用各自的预处理和后处理。裸 RKNN 不包含完整标签、anchors 和任务契约，必须和 `config.json` 一起复制；不同路径可用 `config_path=` 指定。

## 安装

SDK 已附带 RKNN-Toolkit-Lite2 2.3.2 的六个 aarch64 离线 wheel（CPython 3.7–3.12）。一键脚本启用 SDK 支持的 CPython 3.8–3.12。在 RK3588 等 64 位 Linux 板端解压后执行：

```bash
bash install.sh
.venv/bin/python example.py
```

如果已经激活虚拟环境，`install.sh` 会直接安装到当前环境；否则会在 SDK 根目录自动创建 `.venv`。也可用 `EDGEFORGE_VENV=/指定路径 bash install.sh` 指定环境目录。安装器会校验 wheel、按 Python 版本自动选择文件，并安装 SDK 的 NumPy、OpenCV 等依赖。

随包 wheel 只负责 Python 侧 RKNNLite。板卡固件中的 `librknnrt.so` 和 NPU 驱动仍由系统提供，安装脚本不会替换系统库；若模型编译器与板端运行库版本不同，运行时会给出兼容性警告。

板端使用 RKNNLite，无需传 `target`。装有完整 RKNN-Toolkit2 的 x86 调试机必须连接真实板卡并使用对应模型类，例如 `YoloV10("model", target="rk3588", device_id="设备 ID")`；已编译 RKNN 不能直接使用 x86 模拟器。

## 推荐调用

```python
from edgeforge_sdk import YoloV10

with YoloV10("model") as model:
    results = model.predict("test.jpg", confidence=0.25, iou=0.45)
    print([item.as_dict() for item in results])
    model.draw("test.jpg", results, "result.jpg")
```

也可以运行 `python3 -m edgeforge_sdk YoloV10 model test.jpg --output result.jpg`。CLI 同样要求显式指定模型类。

## `load()`

`load()` 加载模型包中的 RKNN 并初始化 NPU，成功后返回当前显式模型实例。重复调用不会重复初始化；`predict()`/`infer_raw()` 会自动调用。需要预热时可显式调用，结束后调用 `close()`。

```python
model = YoloV10(model_path="other.rknn", config_path="other.json", target="rk3588")
model.load()
try:
    ...
finally:
    model.close()
```

## `predict()`

`predict(image, *, confidence=0.25, iou=0.45, top_k=5)` 顺序执行预处理、NPU 推理和结构化后处理。`image` 可为图片路径或 OpenCV BGR `H×W×3` 数组。

目标检测返回 `list[Detection]`。每项包含原图坐标 `box=(x1,y1,x2,y2)`、`score`、`class_id`、`label`，并提供 `as_dict()`。没有目标时返回空列表。未注册结构化后处理的架构会抛出 `NotImplementedError`。

图像分类返回 `list[Classification]`，每项包含 `class_id`、`label`、Softmax 后的 `score`、`rank` 和 `as_dict()`；`top_k` 控制返回数量，`confidence/iou` 仅供检测任务使用。

## `infer_raw()`

`infer_raw(inputs) -> list[numpy.ndarray]` 只运行 NPU，不进行预处理和后处理。单输入可直接传数组；多输入须按模型节点顺序传列表。返回数组的数量、顺序、布局和数据类型与编译模型一致，可能是 NCHW/NHWC 或量化整数。

```python
tensor, context = model.preprocess("test.jpg")
outputs = model.infer_raw(tensor)
```

直接构造输入时，调用方负责颜色、尺寸、dtype 和量化约定。

## `postprocess()`

`postprocess(outputs, context, *, confidence=0.25, iou=0.45, top_k=5)` 将本模型的完整裸输出转换为结构化结果。`context` 必须来自同一张图的 `preprocess()`；检测任务用它还原 letterbox 坐标，分类任务用 `top_k` 返回排序后的类别。

```python
with YoloV10("model") as model:
    tensor, context = model.preprocess("test.jpg")
    outputs = model.infer_raw(tensor)

results = model.postprocess(outputs, context, confidence=0.25)
```

检测后处理包含输出布局归一化、DFL/网格解码、阈值过滤、架构需要时的 NMS 和原图坐标还原。输出结构不符合本模型契约时抛出 `ValueError`。

分类后处理对单输出 logits 执行数值稳定的 Softmax，并映射随包标签；返回 `Classification`，不会套用检测框逻辑。

## 生命周期与并发

推荐使用对应的显式类和上下文管理器，例如 `with YoloV10("模型目录") as model:`，退出时自动释放运行时。同一个模型实例不是线程安全的。

接口关系：`predict(image)` 等价于 `preprocess(image)` → `infer_raw(tensor)` → `postprocess(outputs, context)`。一般业务使用 `predict()`；自定义调度使用分步接口；完全自定义后处理只使用 `infer_raw()`。
