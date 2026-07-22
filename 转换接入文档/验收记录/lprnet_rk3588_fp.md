# LPRNet / RK3588 / FP 验收记录

- 权威训练仓库：`sirius-ai/LPRNet_Pytorch`；导出网络定义直接复用 RKNN Model Zoo `examples/LPRNet/python/export_onnx.py`。
- 原始权重：`Final_LPRNet_model.pth`，大小 `1,806,995` 字节，SHA-256 `d1ca22ca7a251a90702a166c8ad8ce7f05acb39f5ef8055139a33d462716d1ad`。
- 输入契约：OpenCV BGR、直接缩放为 `94×24`，mean `[127.5,127.5,127.5]`、std `[127.5,127.5,127.5]`。
- ONNX：输入 `[1,3,24,94]`，输出 `[1,68,18]`；大小 `1,790,634` 字节，SHA-256 `c9888f91a84d9a1566e0acad7c8102986c4737b8d7a7df8de5177983a92b0cf9`。
- RKNN：RK3588 FP 构建成功，大小 `1,879,920` 字节，SHA-256 `0baf31b9014246401c730ef782a82d4f6803e260c276c6c338e95a7c5526c6d8`。
- SDK ZIP：大小 `4,913,662` 字节，SHA-256 `32d0ec20cbd4df8658dd5ffe4671f5489bd93c723f18a974533a063701761bab`。
- SDK 契约：显式 `LPRNet("model")`；返回 `TextRecognition(text, score, token_ids)`，执行 68 字符表的 greedy CTC 解码、重复字符折叠和 blank 移除。
- 官方 `test.jpg` 验收：结构化文本精确为 `湘F6CL03`，token IDs `[18,46,37,43,51,31,34]`，平均 token 置信度 `0.999988`，与 Zoo 预期一致。
- 板端状态：转换机缺少可连接的 NPU 代理，RKNNLite 板端执行待补；原始权重导出、RKNN 构建和 ONNX Runtime SDK 后处理已通过。
