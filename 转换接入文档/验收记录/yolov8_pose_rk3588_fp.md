# YOLOv8 Pose / RK3588 / FP 验收记录

- 权威导出仓库：`airockchip/ultralytics_yolov8`，固定版本与 YOLOv8 Detect 一致。
- 验收权重：Ultralytics 官方 v8.2.0 `yolov8n-pose.pt`，SHA-256 `c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212`，大小 `6,832,633` 字节。
- 导出：Rockchip 修改版 Ultralytics `format=rknn`，opset 12，输出三路 `[1,65,80/40/20,80/40/20]` 检测特征与一路 `[1,17,3,8400]` 关键点。
- SDK 契约：显式 `YoloV8Pose("model")`；`predict()` 返回 `Pose`，包含 box、score、person 类别和 17 个 `[x,y,confidence]` 关键点。
- 真实产物：RK3588 FP RKNN `8,564,259` 字节；SDK ZIP `27,834,423` 字节。
- SDK 验证：官方 `bus.jpg` 检出 3 人，框约为 `(109,235,224,537)`、`(211,241,285,508)`、`(476,234,560,518)`，分数约 `0.886/0.870/0.860`；每个结果均为 17×3 关键点。
- 精度入口：FP 已验收并可直接转换；INT8 依 Rockchip 要求必须上传校准图 ZIP。
- 板端状态：转换机缺少可连接的 NPU 代理，RKNNLite 板端执行待补。
