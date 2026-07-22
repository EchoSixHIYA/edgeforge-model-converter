# YOLOv8 OBB / RK3588 / FP 验收记录

- 权威导出仓库：`airockchip/ultralytics_yolov8`，固定版本与 YOLOv8 Detect 一致。
- 验收权重：Ultralytics 官方 v8.2.0 `yolov8n-obb.pt`，SHA-256 `fa6e4cd2691f132875c143135affaa66b5d89394ebb1d07d19770a9b6382c1b8`，大小 `6,567,590` 字节。
- 导出：Rockchip 修改版 Ultralytics `format=rknn`，opset 12，服务器显式固定 `imgsz=640`；输出三路 `[1,79,80/40/20,80/40/20]` 检测特征与一路 `[1,1,8400]` 角度特征。
- SDK 契约：显式 `YoloV8Obb("model")`；`predict()` 返回 `OrientedDetection`，包含 center、size、弧度 angle、四顶点 points、score 和类别。
- 后处理：DFL 距离、`(angle_feature-0.25)×π` 角度、旋转中心解码、分类别旋转 NMS 与 letterbox 还原。
- 真实产物：RK3588 FP RKNN `7,658,336` 字节；SDK ZIP `26,680,204` 字节。
- SDK 验证：官方 `test.jpg`，confidence 0.5 返回 160 个旋转目标。首项与同一 PT 的 Ultralytics 结果一致：center `(388.97,21.16)`、size `(54.87,18.90)`、angle `0.50210 rad`、score `0.86307`、class `ship`。
- 精度入口：FP 已验收并可直接转换；INT8 必须上传校准图 ZIP。板端 RKNNLite 执行待有 NPU 设备后补测。
