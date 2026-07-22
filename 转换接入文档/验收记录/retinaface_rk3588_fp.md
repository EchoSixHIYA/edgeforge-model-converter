# RetinaFace / RK3588 / FP 验收记录

- 权威训练仓库：`biubug6/Pytorch_Retinaface`，验收提交 `b984b4b775b2c4dced95c1eadd195a5c7d32a60b`。
- 验收权重：官方 MobileNet0.25 `mobilenet0.25_Final.pth`，大小 `1,789,735` 字节，SHA-256 `2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220`。
- 输入契约：OpenCV BGR、固定 `320×320`，mean `[104,117,123]`、std `[1,1,1]`。
- ONNX 契约：三路输出依次为 `loc [1,4200,4]`、`confidence [1,4200,2]`、`landmarks [1,4200,10]`；ONNX 大小 `1,729,017` 字节，SHA-256 `3c6a7f2f54f1420fa7cde794cbae0ae4ed5519772e4cd4a4d0a222f955cff2b1`。
- RKNN：RK3588 FP 构建成功，大小 `1,572,607` 字节，SHA-256 `d187635d27f7384031b08f0bda69e1172fb0b789966bf13d302a2983d9dc9e72`。
- SDK ZIP：大小 `6,952,682` 字节，SHA-256 `a92fa1f5ed6c64fdf17d4a5cfc21f5fbc9ff4a82aa901f0ba835f215b98f687a`。
- SDK 契约：显式 `RetinaFace("model")`；返回 `FaceDetection(box, score, landmarks, label, class_id)`，其中 `landmarks` 固定为 5 个原图坐标点。
- 官方 `test.jpg` 验收：返回 1 张人脸，置信度 `0.999484`，框约为 `[302.92,76.57,476.93,302.40]`，与 RKNN Model Zoo 参考框 `[302,76,476,300]` 一致。
- 板端状态：转换机缺少可连接的 NPU 代理，RKNNLite 板端执行待补；原始权重导出、RKNN 构建和 ONNX Runtime SDK 后处理已通过。
