# YOLOv7 / RK3588 / FP 验收记录

- 权威导出仓库：`airockchip/yolov7`，提交 `c2d39f4db6b82800ce6c61740be5ef82854c5d3e`；基于上游 YOLOv7 提交 `44d8ab41780e24eba563b6794371f29db0902271`。
- 验收权重：WongKinYiu YOLOv7 v0.1 `yolov7-tiny.pt`，SHA-256 `0fa1423606133c06362ec716029dade1169d5631e83195b75d8efa66d2ee0b77`，大小 `12,639,769` 字节。
- 导出：官方 `export.py --rknpu`，opset 12，固定输入 `[1,3,640,640]`，输出 `[1,255,80,80]`、`[1,255,40,40]`、`[1,255,20,20]`。
- 元数据：从 checkpoint 提取类别标签和像素 anchors；本验收 anchors 从 `[12,16]` 到 `[459,401]`，不复用 YOLOv5 anchors。
- 后处理：显式 `YoloV7("model")` 使用 YOLO anchor 网格解码、objectness×class、逐类 NMS 和 letterbox 坐标还原。
- 真实产物：RK3588 FP RKNN `14,086,720` 字节；SDK ZIP `48,260,436` 字节。
- SDK 验证：官方 `bus.jpg` 返回 5 个结果并包含 `bus/person`。首个结果约 `(212.8,242.4,285.5,511.4)`、分数 `0.8920`；官方记录约 `(212,241,285,511)`、`0.886`。
- 板端状态：转换机缺少可连接的 NPU 代理，RKNNLite 板端执行待补；原始权重导出、RKNN 构建与 ONNX Runtime SDK 后处理验收已通过。
