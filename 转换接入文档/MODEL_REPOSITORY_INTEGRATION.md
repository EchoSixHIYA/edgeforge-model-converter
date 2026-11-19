# 模型仓库接入指南

本文面向继续开发 EdgeForge 模型转换站的 Agent。目标不是“把一个 ONNX 转成 RKNN”，而是把某个模型的**原始训练权重**可靠接入完整交付链路：

```text
原始权重
  → 模型专用仓库 exporter
  → 厂商友好型中间模型
  → 芯片厂商编译器
  → 边缘模型
  → 可安装、可调用、带示例的架构 SDK
```

任何缺少专用 exporter、结构化后处理或 SDK 使用示例的接入，都不能标记为“可用”。

## 1. 项目定位

服务运行于 WSL2 Ubuntu，通过 Flask 向 Windows 宿主机提供网站和 API。目前启用的芯片厂家是 Rockchip，输出格式为 RKNN；Sophon/BModel 已预留厂家和后端注册位置。

核心原则：

1. 网站只接收训练流程产生的原始权重，例如 PyTorch `.pt`。
2. 用户不能上传预转换 ONNX 绕过模型专用 exporter。
3. 不同模型必须使用其专门仓库中的 Rockchip/Sophon 优化导出代码。
4. ONNX 等中间模型只在任务私有目录内部流转。
5. 成功任务必须同时交付边缘模型和可调用 SDK。
6. 官方仓库示例只能放入 SDK 的 `reference/`，不能冒充 SDK 本体。

## 2. 目录与职责

```text
模型转换服务器/
├─ app.py                         Flask 入口
├─ server/
│  ├─ web.py                     页面及任务 API、输入校验
│  ├─ catalog.py                 厂家、芯片、架构和能力目录
│  ├─ exporters.py               原始权重 → 友好 ONNX 的架构路由
│  ├─ export_worker.py           隔离加载 Ultralytics 类专用仓库
│  ├─ pipeline.py                导出、校验、编译、SDK 打包流水线
│  ├─ tasks.py                   单任务编译队列、状态持久化、实时进度
│  └─ backends/
│     ├─ base.py                 厂商编译后端抽象与注册表
│     └─ rknn.py                 ONNX → RKNN
├─ sdk_template/
│  ├─ edgeforge_sdk/             下载包中的真实 SDK 本体
│  │  ├─ model.py                统一调用、预处理和结构化后处理
│  │  ├─ __init__.py
│  │  └─ __main__.py             SDK 命令行入口
│  ├─ example.py                 必须随 SDK 下载的最小使用示例
│  ├─ SDK_README.md              SDK 用户文档
│  └─ pyproject.toml             SDK 安装定义
├─ vendor/
│  ├─ rknn_model_zoo/            官方部署示例和参考后处理
│  ├─ yolov10/                   Rockchip YOLOv10 专用仓库
│  └─ exporters/                 其他模型专用导出仓库
├─ scripts/
│  ├─ setup_wsl.sh               环境、工具链和仓库安装
│  ├─ start_wsl.sh               服务启动
│  ├─ smoke_architecture.py      原始权重完整转换冒烟测试
│  └─ smoke_sdk.py               生成后 SDK 调用测试
└─ tests/                        API 和 SDK 回归测试
```

## 3. 完整任务生命周期

### 3.1 网站提交

`POST /api/tasks` 接收：

- `vendor`：芯片厂家，例如 `rockchip`。
- `architecture`：模型架构，例如 `yolov10`。
- `platform`：目标芯片，例如 `rk3588`。
- `precision`：厂家、架构与目标芯片共同支持的精度。已登记的 `i8` 自动使用仓库内置的 COCO 校准样本，不接收用户校准集。
- `model`：原始权重。

后端格式不能由用户随意填写。`web.py` 根据厂家目录自动选择 RKNN/BModel 等编译器。

### 3.2 架构导出

`Pipeline._to_onnx()` 必须调用 `ArchitectureExporters.export()`。公开任务不能走内部 ONNX 旁路。

导出器职责：

```text
输入：原始权重路径、任务目录、架构、厂家、进度回调
输出：任务目录内唯一、可验证的友好 ONNX 路径
失败：抛出包含专用导出器末尾日志的异常
```

### 3.3 厂商编译

中间模型通过 `BackendRegistry` 路由。例如 `RknnBackend` 执行：

1. `rknn.config()`；
2. `rknn.load_onnx()`；
3. `rknn.build()`；
4. `rknn.export_rknn()`。

未来 Sophon 后端应新增 `server/backends/sophon.py`，实现相同的 `ConversionBackend` 接口，不应把 Sophon 命令写进 RKNN 后端或网页任务逻辑。

### 3.4 SDK 组装

`Pipeline._package_sdk()` 将生成的模型、运行配置、类别、SDK 本体和参考代码封装为 ZIP。下载包不是源码仓库快照，而是本次模型的可执行交付物。

## 4. 接入一个新模型仓库

以下步骤必须按顺序完成。

### 步骤一：确定权威仓库和权重格式

从 RKNN Model Zoo 对应示例的 README 开始，找到它指向的模型专用仓库。优先级：

1. 芯片厂家维护的优化分支，如 `airockchip/*`；
2. Model Zoo README 明确指定的提交或导出说明；
3. 原始模型官方仓库。

记录：

- 原始权重扩展名；
- 框架和版本；
- exporter 的真实入口；
- 输入尺寸及动态尺寸限制；
- 输出节点含义；
- 是否移除了 NMS、DFL、decode 等算子；
- 官方 Python/C++ 后处理所在位置；
- 支持的芯片和精度。

不要根据模型名字猜测导出命令。

### 步骤二：安装专用仓库

在 `scripts/setup_wsl.sh` 中加入可重复执行的浅克隆逻辑：

```bash
if [[ ! -d vendor/exporters/example/.git ]]; then
  git clone --depth 1 https://github.com/vendor/example.git vendor/exporters/example
fi
```

依赖必须进入项目虚拟环境。不同仓库若存在同名 Python 包或版本冲突，应使用独立子进程和 `PYTHONPATH` 隔离，不能在 Flask 进程内直接切换已导入模块。

### 步骤三：登记目录能力

在 `server/catalog.py` 中登记：

```python
_item(
    "example",
    "Example Model",
    "目标检测",
    source="pt",
    input_size=[640, 640],
    export_ready=True,
    weight_formats=(".pt",),
    platforms=["rk3588", "rk3568"],
    notes="使用厂商专用 exporter.py。",
)
```

只有满足本文“完成定义”的架构才能设置 `export_ready=True`。仅能导出 ONNX、但没有可调用 SDK 后处理器的架构仍应保持不可提交。

### 步骤四：实现架构导出器

在 `ArchitectureExporters` 中注册架构族和仓库。Ultralytics 类仓库优先通过隔离 worker 调用模型对象：

```python
cmd = [
    sys.executable,
    str(root / "server" / "export_worker.py"),
    "--repo", str(repo),
    "--model", str(source.resolve()),
    "--format", "rknn",
    "--opset", "13",
]
```

`format=rknn` 在部分 Rockchip 仓库中表示“执行 exporter 的 RKNN 友好 ONNX 分支”，并不代表它已经编译出 `.rknn`。必须阅读仓库中的 exporter 实现确认其真实输出。

传统脚本型仓库可以直接执行专用 `export.py`，但参数必须来自该仓库的文档或源码。例如 YOLOv5 的 Rockchip 分支要求 `--rknpu`。

导出完成后：

1. 找到本次生成的 ONNX；
2. 复制到任务目录的 `model.onnx`；
3. 使用 `onnx.checker.check_model()` 校验；
4. 保留 exporter 日志用于错误反馈；
5. 不允许退回通用 PyTorch→ONNX 导出作为静默兜底。

### 步骤五：实现 SDK 后处理器

在 `sdk_template/edgeforge_sdk/model.py` 中为架构注册结构化后处理。应优先对照 Model Zoo 官方 Python 示例，但实现为稳定 SDK API，而不是导入或启动官方脚本。

必须处理：

- NCHW/NHWC 输出布局；
- 多尺度输出顺序；
- sigmoid/softmax 是否已位于图内；
- DFL、anchor、decode；
- NMS 或 YOLOv10 top-k；
- letterbox 坐标还原；
- 自训练类别数量和标签；
- 空结果；
- 量化输出数据类型；
- 分割 mask、姿态 keypoints、OBB 角度等任务特有结果。

统一方法契约通过显式架构类提供，不提供自动架构入口：

```python
from edgeforge_sdk import YoloV10

model = YoloV10("model")
tensor, context = model.preprocess(image)
raw_outputs = model.infer_raw(tensor)
results = model.postprocess(raw_outputs, context)

# 等价快捷入口
results = model.predict(image)
```

不要让 `predict()` 对未实现的任务返回含义不明的数组。未注册时必须明确抛出 `NotImplementedError`，并且该架构不能在网站中标记为完整可用。

### 步骤六：提取模型元数据

自训练模型的类别不能硬编码为 COCO。打包器应优先从导出 ONNX 的 `metadata_props["names"]` 提取标签；只有元数据缺失时才回退到专用仓库的标签文件。

如模型还需要 anchors、词表、tokenizer、字典或归一化配置，应复制到 `edgeforge_sdk/` 并写入 `config.json`，作为包数据随 SDK 安装。

## 5. 下载 SDK 的强制结构

每个成功任务的 ZIP 至少包含：

```text
<architecture>_<platform>_<precision>_sdk.zip
├─ edgeforge_sdk/
│  ├─ __init__.py
│  ├─ __main__.py
│  ├─ model.py
│  ├─ config.json
│  └─ model.rknn              本次生成的边缘模型
├─ example.py                 必须存在且可直接运行
├─ SDK_README.md              安装和 API 使用说明
├─ pyproject.toml             支持 pip install .
├─ manifest.json              构建信息
├─ model/
│  ├─ model.rknn
│  └─ model.onnx              内部中间产物，供排障
└─ reference/
   ├─ python/                 官方参考实现
   ├─ cpp/
   └─ README.md
```

### SDK 示例的最低要求

下载包中的 `example.py` 必须调用 SDK 本体，不能调用 `reference/` 中的官方脚本：

```python
from edgeforge_sdk import YoloV10

with YoloV10("model") as model:
    results = model.predict("test.jpg", confidence=0.25)
    for item in results:
        print(item.as_dict())
    model.draw("test.jpg", results, "result.jpg")
```

`SDK_README.md` 必须同时包含：

1. 板端 RKNN-Toolkit-Lite2 或对应厂商运行时的安装说明；
2. `pip install .`；
3. Python 最小调用示例；
4. CLI 调用示例；
5. 返回对象字段；
6. 多输入模型的输入约定；
7. 连接 x86 调试机和板端运行的区别；
8. 当前明确限制。

## 6. 进度与错误处理

转换任务串行执行，避免多个厂商编译器争用内存。新增阶段时应通过 `progress(percent, message)` 报告状态。

建议区间：

| 阶段 | 进度 |
|---|---:|
| 上传文件检查 | 8% |
| 准备内置样本（仅 INT8） | 15% |
| 专用 exporter | 24%–38% |
| ONNX 校验 | 38% |
| 厂商模型加载 | 48%–56% |
| 编译和优化 | 65%–86% |
| SDK 组装 | 91% |
| 完成 | 100% |

错误信息应说明失败的是“专用导出”“ONNX 校验”“厂商编译”还是“SDK 打包”，并附相关子进程日志末尾。不要只返回“转换失败”。

## 7. 验收测试

### 7.1 基础测试

```bash
/home/$(id -un)/.edgeforge-model-converter-venv/bin/python -m pytest -q
```

至少覆盖：

- 不支持的厂家/架构被拒绝；
- 已转换 ONNX 上传被拒绝；
- 原始权重扩展名校验；
- NCHW/NHWC 输出布局；
- 任务持久化；
- SDK 包必需文件存在。

### 7.2 原始权重完整冒烟测试

```bash
python scripts/smoke_architecture.py yolov10 /path/to/yolov10n.pt
```

该测试必须实际经历：

```text
PT → 专用 exporter → ONNX → RKNN → SDK ZIP
```

不能用手工准备的 ONNX 代替。

### 7.3 SDK 调用测试

```bash
python scripts/smoke_sdk.py \
  /path/to/generated_sdk.zip \
  /path/to/internal_model.onnx \
  /path/to/test.jpg
```

测试应导入生成包对应的显式模型类，并调用其 `preprocess()` 和 `postprocess()`。对于目标检测，结果至少要与官方示例在类别、置信度和坐标上基本一致。

YOLOv10 当前基准示例应识别官方 bus 图片中的 `bus`，不能仅以“没有抛异常”作为通过条件。

板端可用时，还必须使用 SDK 内嵌 `.rknn` 和 RKNNLite 实测 `predict()`。

## 8. 完成定义（Definition of Done）

一个新架构只有同时满足以下条件才能上线：

- [ ] 找到并记录模型专用权威仓库。
- [ ] 网站接收原始权重，不接收预转换 ONNX。
- [ ] 专用 exporter 能生成厂商友好中间模型。
- [ ] 中间模型通过结构校验。
- [ ] 厂商编译器实际生成目标边缘模型。
- [ ] FP 模式完成真实转换测试。
- [ ] SDK 提供该任务类型的结构化 `predict()`。
- [ ] 自训练标签或词表被正确打包。
- [ ] SDK ZIP 包含 `example.py` 和 `SDK_README.md`。
- [ ] `example.py` 调用 `edgeforge_sdk`，而不是官方参考脚本。
- [ ] SDK 调用结果与官方参考输出基本一致。
- [ ] 官方代码仅位于 `reference/`。
- [ ] 自动化测试通过。
- [ ] WSL 服务重启后 Windows 可通过 `http://localhost:5000` 访问。

如果任一项未完成，应保持该架构不可提交，并在目录中标注具体缺口。

## 9. 当前参考接入

YOLOv10 是完整链路的首要参考：

1. 原始 `.pt` 上传；
2. 加载 `airockchip/yolov10`；
3. 调用其 exporter 的 `format=rknn` 分支生成友好 ONNX；
4. RKNN-Toolkit2 编译；
5. 从 ONNX 元数据提取 80 类标签；
6. 生成内嵌模型的可安装 SDK；
7. 使用 SDK 公共前后处理接口验证官方 bus 图片；
8. 输出结构化 `Detection`，并能绘制结果图。

接入新仓库时应复制这套生命周期和验收标准，而不是复制 YOLOv10 的具体输出节点假设。
