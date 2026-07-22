# YOLOv5 Seg / RK3588 / FP 验收记录

- 权威导出仓库：`airockchip/yolov5`，使用 `export.py --rknpu`。
- 验收权重：Ultralytics YOLOv5 v7.0 官方 `yolov5s-seg.pt`，SHA-256 `78558867515e9d9654c52ed87c40c22d90a983bea63450c611cbd8c2b13e3aa2`，大小 `15,593,297` 字节。
- 导出：三路 `[1,255,80/40/20,80/40/20]` anchor 检测、三路 `[1,96,80/40/20,80/40/20]` mask 系数和一路 `[1,32,160,160]` prototype，共 7 路。
- 元数据：从 checkpoint 提取 COCO 标签和三组 anchors，不硬编码为其他 YOLOv5 变体的 anchors。
- SDK 契约：显式 `YoloV5Seg("model")`；返回通用 `InstanceMask`，但使用 YOLOv5 自身的 anchor/grid/objectness 和 mask 系数解码。
- 真实产物：RK3588 FP RKNN `16,960,542` 字节；SDK ZIP `58,182,966` 字节。
- SDK 验证：官方 `bus.jpg` 返回 5 个实例，包含 bus/person，掩码均为原图尺寸且非空；首项约为 `(212.17,242.49,285.96,510.72)`、`0.87057`。
- 精度入口：FP 已验收；INT8 必须上传校准图 ZIP；板端 NPU 实机执行待补。
