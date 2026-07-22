# RKNN 官方模型逐项接入计划

本计划以仓库内固定版 `vendor/rknn_model_zoo` 的 28 个官方示例为基线。目录收录不等于上线；每个模型只有完成下述验收门槛后，才把 `server/catalog.py` 中的 `verified` 设为 `True` 并进入工作台可提交列表。

## 单模型完成门槛

每个模型必须留下可复现证据：

1. 确认权威训练仓库、原始权重格式、固定提交和许可证。
2. 用户上传原始权重，专用 exporter 生成 RKNN 友好 ONNX；禁止公开 ONNX 旁路。
3. ONNX 通过结构检查，输出节点、布局、dtype 和模型元数据符合官方示例。
4. FP 模式真实生成目标 RKNN；官方支持 INT8 时，再用真实校准集完成量化构建。
5. SDK 按任务类型实现 `preprocess / infer_raw / postprocess / predict`，返回稳定结构化结果。
6. SDK 输出与 RKNN Model Zoo 官方 Python 示例对比：类别/文本一致，置信度与坐标在合理容差内。
7. SDK ZIP 包含模型、配置、`example.py`、演示输入、`SDK_README.md` 和 `reference/`。
8. 自动化回归通过；板端可用时完成 RKNNLite `predict()` 实测。

`scripts/smoke_architecture.py` 可设置内部 `acceptance_run` 测试未开放模型。网页 API 永远不会接受此字段。

## 统一 SDK 交付契约

- `edgeforge-model-sdk` 只安装一次；版本 2 起内置所有已接入架构的适配器。
- 每次转换包的 `model/` 目录是可移植模型包，包含一个 `.rknn`、`config.json` 和供审计的 ONNX。
- SDK 只提供显式入口，例如 `YoloV10("model/")`、`YoloV8(...)`、`MobileNetV2(...)`；不提供按配置自动选择类的入口，架构不匹配会拒绝加载。
- 新模型接入时是在同一 SDK 中注册新的任务适配器，不要求部署端为每个模型安装独立 SDK。

## 接入顺序

| 批次 | 模型 | 原因与主要工作 |
|---|---|---|
| 0 | YOLOv10 | 已有基准闭环，作为验收参考 |
| 1 | YOLOv8、YOLO11、YOLOv5 | 已有原始权重 exporter；先完成检测 SDK 与真实转换 |
| 2 | MobileNet、ResNet、YAMNet、LPRNet | 单输入单输出；建立分类、音频分类和 CTC 结果契约 |
| 3 | YOLOv6、YOLOv7、YOLOX、PP-YOLOE、RetinaFace | 单模型检测，但各自 decode/anchor/关键点不同 |
| 4 | DeepLabV3+、PP-LiteSeg、YOLOv5-Seg、YOLOv8-Seg | 建立语义/实例分割结果、掩码还原与绘制 |
| 5 | YOLOv8-Pose、YOLOv8-OBB | 建立关键点和旋转框结果、专用 NMS |
| 6 | PPOCR、MobileSAM | 多模型上传、模型间数据流和组合 SDK |
| 7 | YOLO-World、CLIP | 图像+文本输入、tokenizer/词表打包 |
| 8 | Wav2Vec2、Whisper、Zipformer | 音频预处理、CTC/自回归/流式解码、多模型状态 |
| 9 | Lite Transformer、MMS TTS | 文本 tokenizer、多模型自回归和音频输出 |

## 当前状态

| 模型 | exporter | 结构化 SDK | 真实验收 | 工作台 |
|---|---:|---:|---:|---:|
| YOLOv10 | 完成 | 完成 | 完成 | 开放 |
| YOLOv8 Detect | 完成；本轮修正为 `format=rknn` | 完成 | RK3588 FP 与官方 bus 图通过 | 开放 |
| YOLO11 Detect | 完成；同步修正为 `format=rknn` | 完成 | RK3588 FP 与官方 bus 图通过 | 开放 |
| YOLOv5 Detect | 完成，含标签/anchors 侧车 | 完成 | RK3588 FP 与官方 bus 图通过 | 开放 |
| YOLOv6 Detect | 完成，Rockchip 9 路 RKNN 导出 | 完成，4 通道距离解码与显式类 | RK3588 FP 与官方 bus 图通过 | 开放 |
| YOLOv7 Detect | 完成，`export.py --rknpu` | 完成，checkpoint anchors 与显式类 | RK3588 FP 与官方 bus 图通过 | 开放 |
| YOLOX | 完成，`export_onnx.py --rknpu` 与变体匹配 | 完成，专属 grid/stride 解码与显式类 | RK3588 FP 与官方 bus 图通过 | 开放 |
| YOLOv8 Pose | 完成，`format=rknn` 固定 640 输入 | 完成，`Pose` 17×3 关键点与显式类 | RK3588 FP 与官方 bus 图通过 | 开放 |
| YOLOv8 OBB | 完成，`format=rknn` 固定 640 输入 | 完成，旋转解码/NMS 与显式类 | RK3588 FP 与官方港口图通过 | 开放 |
| YOLOv8 Seg | 完成，13 路 `format=rknn` 输出 | 完成，prototype mask 与 `InstanceMask` | RK3588 FP 与官方 bus 图通过 | 开放 |
| YOLOv5 Seg | 完成，7 路 `export.py --rknpu` 输出 | 完成，权重 anchors/prototype mask 与 `InstanceMask` | RK3588 FP 与官方 bus 图通过 | 开放 |
| MobileNet V2 | 完成，接受 `.pth` 状态字典 | 完成，`Classification` Top-K | RK3588 FP；官方 bell 图类别 494 通过 | 开放 |
| ResNet-50 | 完成，接受 `.pth` 状态字典 | 复用分类契约，RGB 预处理 | RK3588 FP；官方 dog 图类别 155 通过 | 开放 |
| RetinaFace | 完成，接受 MobileNet0.25/ResNet50 `.pth` | 完成，`FaceDetection` 人脸框与 5 点关键点 | RK3588 FP；官方 test 图通过 | 开放 |
| PP-YOLOE | 完成，接受 CRN-S/CRN-M `.pdparams` | 完成，9 路 DFL/类别/score-sum 与显式 `PPYoloE` | S/M 均完成 RK3588 FP；官方 bus 图通过 | 开放 |
| 其余 13 项 | 见工作台 `blocker` | 未完成 | 待开始 | 关闭 |

## YOLOv8 当前执行清单

- [x] 核对官方 Model Zoo 输出：每尺度为 box DFL、class confidence、score sum，共 9 路。
- [x] 核对 Rockchip exporter：必须设置 `format=rknn` 才走友好输出分支。
- [x] 修复服务器 exporter 参数。
- [x] 使用兼容的 `yolov8n.pt` 执行真实友好 ONNX 导出。
- [x] 校验 9 个输出的形状；ONNX 无标签元数据，正确回退官方 COCO 标签文件。
- [x] 使用 bus 图片对比 SDK 与官方 Python 后处理结果。
- [x] 构建 RK3588 FP RKNN 和 SDK ZIP。
- [ ] 板端 RKNNLite `predict()`：当前 Toolkit wheel 缺少 `npu_transfer_proxy`，无法探测设备；转换机验收不受影响。
