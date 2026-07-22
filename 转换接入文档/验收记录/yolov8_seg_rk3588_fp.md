# YOLOv8 Seg / RK3588 / FP 验收记录

- 权威导出仓库：`airockchip/ultralytics_yolov8`，固定版本与 YOLOv8 Detect 一致。
- 验收权重：Ultralytics 官方 v8.2.0 `yolov8s-seg.pt`，SHA-256 `0bac0770b55e5eb5b76a61bc535673288dcec36c2bc0cd25ee0d584c632f3413`，大小 `23,914,764` 字节。
- 导出：`format=rknn`、opset 12、`imgsz=640`；每尺度为 DFL 框、80 类分数、辅助分数和 32 维 mask 系数，最后为 `[1,32,160,160]` prototype，共 13 路。
- SDK 契约：显式 `YoloV8Seg("model")`；返回 `InstanceMask`，`mask` 为还原到原图尺寸的 bool ndarray，`as_dict()` 返回 mask_shape/mask_area 而不展开整张图。
- 真实产物：RK3588 FP RKNN `25,644,467` 字节；SDK ZIP `87,936,219` 字节。
- SDK 验证：官方 `bus.jpg` 返回 6 个实例，包含 bus/person，每个掩码非空。框、分数和 mask shape 与同一 PT 的 Ultralytics 推理一致。
- 精度入口：FP 已验收；INT8 必须上传校准图 ZIP；板端 NPU 实机执行待补。
