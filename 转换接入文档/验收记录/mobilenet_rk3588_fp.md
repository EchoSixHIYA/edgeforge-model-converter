# MobileNet V2 / RK3588 / FP 验收记录

- 验收输入：TorchVision `MobileNet_V2_Weights.IMAGENET1K_V1` 状态字典，封装为 `.pth` checkpoint；SHA-256 `1075e72a77a319b24fedf9e947294cfb91cae39a9b442fac91947f490fbaca48`。
- RKNN 官方参考：仓库提交 `bad6c7334531becaf90a561988519b7bec34d0ab`，示例 `examples/mobilenet`。
- 输入契约：OpenCV BGR、`uint8`、NHWC `[1,224,224,3]`；RKNN 配置 mean `[123.675,116.28,103.53]`、std `[58.395,57.12,57.375]`。
- 导出：专用 worker 重建 TorchVision MobileNet V2，严格加载状态字典，导出 opset 12 ONNX；公开入口不接受 ONNX 旁路。
- 真实产物：RK3588 FP RKNN `7,674,353` 字节；SDK ZIP `29,639,270` 字节。
- SDK 验证：官方 `bell.jpg` 返回 Top-5；Top-1 为类别 `494`、标签 `chime`、分数 `0.9833705`。RKNN 官方示例的预期 Top-1 同为类别 `494`（chime/bell/gong 语义）。
- 结构化结果：`Classification(class_id, label, score, rank)`；`predict(..., top_k=5)` 与独立 `preprocess → ONNX Runtime → postprocess` 调用一致。
- 板端状态：转换机缺少可连接的 NPU 代理，RKNNLite 板端执行待补；不影响原始权重导出、RKNN 构建和 SDK 后处理验收。
