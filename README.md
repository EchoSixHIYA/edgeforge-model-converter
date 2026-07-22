# EdgeForge 模型转换服务器

后续接入其他模型仓库前，请先阅读 [模型仓库接入指南](转换接入文档/MODEL_REPOSITORY_INTEGRATION.md)。SDK 调用方请阅读 [SDK API 说明](转换接入文档/SDK_API.md)。

部署在 WSL2 Ubuntu 的 Flask 服务。宿主机浏览器上传训练模型后，服务按架构完成 RKNN 友好 ONNX、RKNN 构建，并把对应 RKNN Model Zoo 官方 Python/C++ 示例与模型一起封装为 SDK。

## 当前能力

- 工作台目录覆盖仓库内固定版 RKNN Model Zoo 的全部 28 个官方示例，包括分类、YOLO 系列、分割、OCR、语音及多模态；每项分别公开专用导出、SDK 后处理和真实验收状态。
- 页面显式选择芯片厂家；当前启用 Rockchip→RKNN，Sophon→BModel 已作为未启用后端展示并预留注册接口。
- YOLOv5/6/7/8/10/11、YOLOX 和 PP-YOLOE（含 S/M）的专用原始权重导出路由已经登记；只有同时完成结构化后处理和真实验收的架构才允许提交。
- 用户入口禁止上传 ONNX；上传训练得到的原始权重后，服务必须先调用所选架构的专用仓库 exporter 生成友好 ONNX，再编译 RKNN。
- INT8/UINT8 支持上传校准图片 ZIP；FP 模式无需校准集。
- Server-Sent Events 实时显示进度；任务和日志持久化；转换任务串行排队。
- 每次成功转换强制生成同一套可安装的 `edgeforge-model-sdk`：内嵌 RKNN、模型标签和可直接运行的演示输入（图像为 `test.jpg`，音频为 `test.wav`）；调用方必须显式选择 `YoloV8`、`PPYoloE`、`YAMNet` 等模型类，再使用统一的 `load/predict/preprocess/infer_raw/postprocess/draw` 接口。官方 Python/C++ 示例保留在 `reference/`，不再作为主调用方案。
- 后端通过 `ConversionBackend` 注册；以后新增 Sophon `.bmodel` 无需改网页任务协议。

## 在 WSL2 安装

Ubuntu 20.04 的 Python 3.8 与许多 RKNN-Toolkit2 官方 wheel 匹配。先从 [RKNN-Toolkit2 官方仓库](https://github.com/airockchip/rknn-toolkit2) 选择与 Python 3.8 / x86_64 匹配的 wheel，放到本目录 `wheels/`。然后在 PowerShell 执行：

```powershell
wsl -d Ubuntu-20.04 -- bash -lc "cd '/mnt/e/边缘部署/模型转换服务器' && bash scripts/setup_wsl.sh"
```

安装脚本会同时安装 CPU 版 PyTorch、各 Rockchip YOLO 导出分支、PP-YOLOE 专用的 Paddle 2.5.2 环境，以及 YAMNet 专用的 TensorFlow 2.13.1 环境。两个隔离环境分别位于 `/home/<用户>/.edgeforge-ppyoloe-export-venv` 和 `/home/<用户>/.edgeforge-yamnet-export-venv`，只负责原始权重导出，不会污染主服务环境。

## 启动并从宿主机访问

```powershell
wsl -d Ubuntu-20.04 -- bash -lc "cd '/mnt/e/边缘部署/模型转换服务器' && bash scripts/start_wsl.sh"
```

打开 <http://localhost:5000>。服务监听 `0.0.0.0`，WSL2 的 localhost 转发开启时宿主机可直接访问。健康检查为 <http://localhost:5000/health>。

生产环境可将 `scripts/start_wsl.sh` 配为 systemd 服务。必须保持 Gunicorn `workers=1`，任务队列当前位于进程内；线程用于网页、进度流和下载，不会并发执行 RKNN 编译。

## 转换约束

RKNN Model Zoo 的“支持模型”表示官方提供了友好 ONNX 与部署示例，不等于任意同名训练权重已经具备自动导出链路。工作台会完整展示 28 个官方示例以及每项的具体缺口，但只允许提交满足“专用 exporter、结构化 SDK 后处理、真实验收”三项条件的模型。当前目录中标记为“可转换”的模型均已完成原始权重、RKNN 构建和统一 SDK 闭环；未完成项仍显示具体 blocker 并拒绝提交。多输入或多模型架构还需要设计一次任务上传多个权重及配套 tokenizer/词表的协议。

量化校准图片应接近真实部署数据分布。上传的 ZIP 会安全解压到任务私有目录并生成绝对路径 `dataset.txt`。

## API

- `GET /health`：环境组件状态
- `GET /api/catalog`：模型、后端和平台目录
- `POST /api/tasks`：multipart 表单，字段 `vendor/architecture/platform/precision/model/calibration`；后端由厂家自动选择
- `GET /api/tasks/:id/events`：SSE 实时进度
- `GET /api/tasks/:id/download/model`：模型产物
- `GET /api/tasks/:id/download/package`：完整 SDK

## 添加其他厂商后端

在 `server/backends/` 实现 `ConversionBackend`，声明输出后缀并在 `Pipeline.__init__` 注册即可。任务、SSE、产物下载和 SDK 清单均与 RKNN 解耦；前端会通过 `/api/catalog` 逐步改为动态展示新后端。

## 测试

```bash
/home/$(id -un)/.edgeforge-model-converter-venv/bin/pytest -q
```

接口测试不需要 RKNN-Toolkit2；真实转换需要官方工具链。
