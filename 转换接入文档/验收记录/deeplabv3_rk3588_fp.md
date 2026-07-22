# DeepLabV3+ / RK3588 / FP 验收记录

- 权威来源：RKNN Model Zoo `examples/deeplabv3` 提供的 DeepLabV3+ MobileNetV2 TensorFlow 冻结图；公开入口接收 `.pb`，拒绝预转换 ONNX。
- 上游源模型：`deeplab-v3-plus-mobilenet-v2.pb`，大小 `8,546,188` 字节，SHA-256 `0299ad261a2455e687b9c80acb815db1e62ccc38ec1ba5166783271bd150f316`。
- 专用导出：固定输入节点 `sub_7:0`、输出节点 `logits/semantic/BiasAdd:0`，使用 tf2onnx 1.16.1 裁图；内部 ONNX 增加 NCHW→NHWC 入口转置以符合 Toolkit2 三通道布局识别。
- 输入契约：RGB `513×513`，mean/std 均为 `[127.5,127.5,127.5]`；内部 ONNX 输入 `[1,3,513,513]`。
- 输出契约：NHWC logits `[1,65,65,21]`；SDK 先双线性放大 logits 至 `513×513`、再 argmax，并以最近邻恢复到原图尺寸。
- SDK：显式 `DeepLabV3Plus("model")`，返回 Pascal VOC 21 类 `SegmentationMask`；包内包含官方 `test_image.jpg`、颜色表和参考实现。
- 官方样例验收：类别及像素数为 background `202230`、bicycle `41318`、person `19621`；掩码 SHA-256 `90c2be3754069b9b458e3da7c34cb674c45944a83043d3c4d36353d1ce64609b`。
- 与 Zoo `python_demo_result.png` 比较：叠加图平均绝对像素误差 `0.0954114`，整像素完全一致率 `0.9973135`。
- ONNX：大小 `8,428,205` 字节，SHA-256 `3c6bb593bc965bf9e08d0f66e8d78b4acf399cae0ab45f164231a7ec0e94850e`。
- RKNN：RK3588 FP 真实构建通过，大小 `7,676,809` 字节，SHA-256 `d95c160b5853e3da3fedc0527cbd359350c5ec9866646f611fc24122454f0696`。
- SDK ZIP：大小 `19,806,988` 字节，SHA-256 `70d7be3a08b2469812907c28ff8f16bda7d6a19940e520200bafb8920aada34b`。
- 自动化：全量 `28 passed`；最终 SDK 结构化调用及官方参考图比较均通过。
- 板端状态：转换机缺少可连接的 NPU 代理，RKNNLite 板端执行待补；源模型导出、RKNN 构建和 ONNX Runtime SDK 后处理已通过。
