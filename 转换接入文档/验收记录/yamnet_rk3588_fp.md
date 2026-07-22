# YAMNet / RK3588 / FP 验收记录

- 权威来源：TensorFlow Hub/Kaggle YAMNet SavedModel；模型结构与标签表按 RKNN Model Zoo `examples/yamnet` 固定。
- 原始输入：包含唯一 `saved_model.pb` 和 `variables/` 的 SavedModel ZIP，大小 `14,243,484` 字节，SHA-256 `53e98f0a1c1d652d2310220e17414820b1476193d66a928f67275f889ab635cd`。
- 导出环境：独立 TensorFlow 2.13.1、TensorFlow Hub 0.16.1、tf2onnx 1.16.1、ONNX 1.16.0；公开入口只接受原始 SavedModel ZIP，不接受预转换 ONNX。
- 输入契约：单声道 float32 波形 `[1,48000]`；SDK 接收 PCM WAV、波形数组或 `(波形,采样率)`，自动重采样至 16 kHz，并截断/补零为 3 秒。
- ONNX：大小 `16,114,565` 字节，SHA-256 `bd8377f23b9bcab9842eb15847833c416b8f027520a57c444e302a37b4796aad`。
- RKNN：RK3588 FP 构建成功，大小 `10,736,581` 字节，SHA-256 `06ac807c32045ce2139d190c3cf698c75474b4ba1abefd6937ad95f42bbaf83d`。
- SDK ZIP：大小 `32,692,197` 字节，SHA-256 `1bbeadef4d51d0f4b9e77e2ef3fd23685b8f8113706eefe1782d4fc0f7f1356a`；包含官方 `test.wav`、显式 `YAMNet`、521 类标签、模型配置和一键安装脚本。
- SDK 契约：返回 `AudioClassification(class_id, label, score, rank)`；从三个输出中按标签维识别 score 张量，对时间帧求平均并输出 Top-K，避免依赖运行时输出顺序。
- 官方 `test.wav` 验收：Top-1 为 `Animal`（class ID 67），平均分 `0.7112376`，与 Zoo 官方示例预期一致。
- 自动化：全量 `27 passed`；原始 SavedModel ZIP → ONNX → RKNN → SDK 包装及 ONNX Runtime 结构化调用均通过。
- 板端状态：转换机缺少可连接的 NPU 代理，RKNNLite 板端执行待补；原始模型导出、RKNN 构建和 SDK 后处理已通过。
