# SOPHGO BM1684X 接入说明

EdgeForge 的 SOPHGO 链路与 RKNN 链路并列存在，不复用 RKNN 后端产物。

当前链路：

```text
原始权重
  -> SOPHGO 友好 ONNX
  -> TPU-MLIR model_transform.py
  -> run_calibration.py (INT8)
  -> model_deploy.py
  -> BM1684X .bmodel
  -> SOPHON-SAIL Python SDK
```

## 已接入的优先模型

首批以 `sophgo/sophon-demo` 中有 TPU-MLIR 脚本的 YOLO 系列为主：

- YOLOv5
- YOLOv7
- YOLOv8 detection
- YOLOv8 Pose
- YOLOv8 Seg
- YOLOv8 OBB，官方 TPU-MLIR 样例使用 1024 输入
- YOLOv8 Cls
- YOLOv9 detection
- YOLOv9 Seg
- YOLOv10
- YOLO11 detection
- YOLO11 OBB，按 OBB 1024 输入接入
- YOLOv12 detection
- YOLOX
- YOLO-World V2

YOLO-World V2 在官方样例中包含 detector 与 `clip_text` 两个 ONNX/BModel 组件，因此 SDK 包会保留多模型结构。官方当前主要脚本覆盖 FP32/FP16，INT8 暂不开放到工作台，直到校准契约完成复刻与验收。

## 精度约定

工作台精度到 TPU-MLIR 的映射：

- `f32` -> `--quantize F32`
- `f16` -> `--quantize F16`
- `bf16` -> `--quantize BF16`
- `i8` -> `--quantize INT8`

INT8 必须上传校准图片 ZIP。服务端会同时生成两种校准输入：`dataset.txt` 继续供 RKNN 使用；SOPHGO 后端把解压后的图片目录传给 `run_calibration.py --dataset`，与 `sophon-demo` 官方脚本保持一致。

## 环境要求

转换机需要先进入官方 TPU-MLIR 环境，并确保以下命令在 `PATH` 中：

- `model_transform.py`
- `run_calibration.py`，仅 INT8 需要
- `model_deploy.py`

准备官方样例仓库并检查当前 shell 是否已进入 TPU-MLIR/SAIL 环境：

```bash
bash scripts/setup_sophon.sh
```

脚本会把 `sophgo/sophon-demo` 克隆到 `vendor/sophon-demo`。也可以通过 `SOPHON_DEMO=/path/to/sophon-demo bash scripts/setup_sophon.sh` 使用已有仓库。TPU-MLIR 和 SOPHON-SAIL 仍以官方安装方式为准，脚本只做仓库准备和命令自检，不把算能编译器装进主服务 venv。

板端运行 `.bmodel` 需要安装 SOPHON-SAIL Python 包。统一 SDK 会根据模型文件后缀自动选择运行时：

- `.rknn` 使用 RKNNLite/RKNN Runtime
- `.bmodel` 使用 `sophon.sail.Engine`

调用入口仍然是显式模型类，例如：

```python
from edgeforge_sdk import YoloV8

with YoloV8("model", device_id=0) as model:
    results = model.predict("test.jpg")
    print([item.as_dict() for item in results])
```

## 验收命令

转换完成后，可先在任意机器做包结构检查：

```bash
python scripts/smoke_bmodel.py path/to/yolov8_bm1684x_f16_sdk.zip
```

在 BM1684X 设备或已进入 SOPHGO Runtime 环境的机器上，要求同时执行 `bmrt_test`：

```bash
python scripts/smoke_bmodel.py path/to/yolov8_bm1684x_f16_sdk.zip --require-bmrt
```

### 板端最小 BModel smoke

如果 BM1684X 板端已经有若干 `.bmodel`，可先用仓库脚本选择体积最小的模型做运行时冒烟：

```bash
bash scripts/board_sophon_smoke.sh /home/linaro/3588_edge
```

脚本会在指定项目目录下创建 `.edgeforge-board-smoke-venv`，优先使用 `--system-site-packages` 复用板端官方安装的 SAIL；如果板端缺少 `ensurepip/python3-venv` 的完整能力，会自动降级为 `--without-pip` 的项目虚拟环境，不污染系统解释器。随后脚本会检查 `bmrt_test`、`sophon.sail`，在指定目录下查找最小 `.bmodel` 并执行：

```bash
bmrt_test --bmodel <smallest.bmodel> --loopnum 1
```

`bmrt_test` 不在 `PATH` 时，脚本会继续在 `/opt/sophon` 和搜索目录下查找。`bmrt_test` 通过后，还会用同一项目虚拟环境执行 SOPHON-SAIL `Engine(...).process(...)`，打印输入/输出 shape，验证 SDK 依赖的高层 Python 运行时。

如果 SSH 已开启，可以从开发机把脚本传到板端后运行；若仍提示找不到 `bmrt_test` 或 `sophon.sail`，需要先安装/进入 SOPHGO Runtime/SAIL 官方环境。

脚本会检查 `manifest.json`、`model/config.json`、`.bmodel` 文件和多组件 `model_files`，并在 `bmrt_test` 存在时加载每个 BModel。

## 版本管理

`vendor/sophon-demo` 不纳入 git。官方仓库只作为外部参考与本地导出依赖，项目版本库仅提交 EdgeForge 的适配层、文档和测试。
