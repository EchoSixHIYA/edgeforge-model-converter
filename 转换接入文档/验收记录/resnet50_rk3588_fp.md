# ResNet-50 / RK3588 / FP 验收记录

- 验收输入：TorchVision `ResNet50_Weights.IMAGENET1K_V2` 状态字典，封装为 `.pth` checkpoint；SHA-256 `c2b8387d9647efb7a5531a2f5865e7f1332dbefa93ccf54ac513dc15f570a2f9`。
- RKNN 官方参考：仓库提交 `bad6c7334531becaf90a561988519b7bec34d0ab`，示例 `examples/resnet`；该示例引用 ONNX Models 的 `resnet50-v2-7.onnx`。
- 输入契约：OpenCV BGR 读取后转换 RGB，`uint8`、NHWC `[1,224,224,3]`；RKNN 配置 mean `[123.675,116.28,103.53]`、std `[58.395,57.12,57.375]`。
- 导出：专用 worker 重建 TorchVision ResNet-50，严格加载状态字典并导出 opset 12 ONNX；公开入口不接受 ONNX 旁路。
- 真实产物：RK3588 FP RKNN `52,164,273` 字节；SDK ZIP `193,759,262` 字节。
- SDK 验证：官方 `dog_224x224.jpg` 返回 Top-5；Top-1 为类别 `155`、标签 `Shih-Tzu`、分数 `0.5807709`。RKNN 官方示例预期 Top-1 同为类别 `155`，但使用的是另一套 ONNX Models 权重，官方记录分数约 `0.794123`，因此不要求分数完全相同。
- 结构化结果：`Classification(class_id, label, score, rank)`；分类模型使用自己的 RGB 颜色契约，不复用 MobileNet 的 BGR 契约。
- 板端状态：转换机缺少可连接的 NPU 代理，RKNNLite 板端执行待补；原始权重导出、RKNN 构建和 SDK 后处理已通过。
