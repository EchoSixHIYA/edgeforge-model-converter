# YOLOv6 / RK3588 / FP 验收记录

- 权威导出仓库：`airockchip/yolov6`，提交 `0e7c2d5a93f6d49ed5ab6f005ccdd9d9bbd3db9b`。
- 验收权重：Meituan YOLOv6 0.3.0 `yolov6n.pt`，SHA-256 `3e2944e1c2d6a0dc4eed72e1493e3afe1c052373588b33abfecf0bfb4212b1a4`，大小 `10,511,334` 字节。
- 导出：官方 `deploy/RKNN/export_onnx_for_rknn.py`，opset 12，固定输入 `[1,3,640,640]`。
- ONNX 输出：三个尺度 80/40/20；每尺度依次为 box `[1,4,H,W]`、class `[1,80,H,W]`、score sum `[1,1,H,W]`，共 9 路。SDK 忽略仅用于快速筛选的 score sum，4 通道 box 直接执行距离解码，不套用 DFL。
- 标签：官方验收权重通过固定 SHA-256 回退 Model Zoo COCO 标签；自训练 checkpoint 必须携带 `names`，缺失时拒绝打包，禁止静默套用 COCO。
- 真实产物：RK3588 FP RKNN `10,599,696` 字节；SDK ZIP `37,465,247` 字节。
- SDK 验证：显式 `YoloV6("model")` 在官方 `bus.jpg` 上返回 7 个结果，包含 `bus` 与 `person`。首个人框约 `(109.1,236.0,223.4,535.3)`、分数 `0.9361`；官方记录约 `(109,236,222,535)`、`0.938`。
- 板端状态：转换机缺少可连接的 NPU 代理，RKNNLite 板端执行待补；原始权重导出、RKNN 构建与 ONNX Runtime SDK 后处理验收已通过。
