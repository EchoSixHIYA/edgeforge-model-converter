# PP-LiteSeg / RK3588 / FP 验收记录

- 权威仓库：PaddlePaddle/PaddleSeg `release/2.8`，验收提交 `a485168da281e942af7d1643aa726d44df8c9951`；模型为 PP-LiteSeg STDC1 Cityscapes。
- 原始权重：`model.pdparams`，大小 `32,639,554` 字节，SHA-256 `86543ad69437495f8e7688d497a52d2472ee098efa918dcfa551803570f5edf3`。
- 导出契约：公开入口只接受 `.pdparams`；导出器对 STDC1/STDC2 的全部参数名称和形状进行匹配，再使用 PaddleSeg 原始网络定义导出固定 `[1,3,512,512]` ONNX。
- 输入契约：OpenCV 图片直接缩放至 `512×512`、BGR 转 RGB，mean `[123.675,116.28,103.53]`、std `[58.395,57.12,57.375]`。
- 输出契约：单个 float32 logits `[1,19,512,512]`，对应 Cityscapes 19 类；SDK argmax 后以最近邻恢复至原图尺寸。
- SDK：显式 `PPLiteSeg("model")`，返回 `SegmentationMask(mask, labels)`；`draw()` 使用随包颜色表生成叠加图，包内包含官方 `test.png`。
- 官方 `test.png` 验收：从原始 `.pdparams` 导出的 ONNX 与 RKNN Model Zoo 官方 ONNX 掩码逐像素一致率 `1.0`；512×512 掩码 SHA-256 为 `305a5aba205d3cf9896c34983886481193bd56be7bb329caaf05a7faa99bf176`。
- ONNX：大小 `32,211,040` 字节，SHA-256 `951481c328b03399df2fc4b3fc90da37b41cc33659ae3f3cef311cf68d29e5e8`。
- RKNN：RK3588 FP 真实构建通过，大小 `18,237,255` 字节，SHA-256 `37c50da3f5411a69f781804a44a5fc3912d4183736d9d8487e9606d6b8bdcd2a`。
- SDK ZIP：大小 `67,990,267` 字节，SHA-256 `ace51ba5c7585e8bb3854a1e76e0fa9f07f4fccc942e18e97d8e77b5f55ecc13`。
- 自动化：全量 `28 passed`；最终 SDK 在官方原始尺寸 `2048×1024` 上返回 12 个实际出现的 Cityscapes 类别及像素统计。
- 板端状态：转换机缺少可连接的 NPU 代理，RKNNLite 板端执行待补；原始权重导出、RKNN 构建和 ONNX Runtime SDK 后处理已通过。
