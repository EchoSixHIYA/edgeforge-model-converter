# YOLOX / RK3588 / FP 验收记录

- 权威导出仓库：`airockchip/YOLOX`，提交 `fc69c1e2ccb30fce7e18250bddc13a68090dca09`。
- 验收权重：官方 `yolox_s.pth`，SHA-256 `f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30`，大小 `72,089,125` 字节。
- 导出：官方 `tools/export_onnx.py --rknpu --no-onnxsim`，固定输入 `[1,3,640,640]`，三路输出为 `[1,85,80,80]`、`[1,85,40,40]`、`[1,85,20,20]`。
- 预处理：RGB letterbox，按 Rockchip 官方配置使用 `mean=[0,0,0]`、`std=[1,1,1]`，不对像素执行 `/255`。
- 后处理：显式 `YoloX("model")` 使用 YOLOX 专属 grid/stride 和指数宽高解码、objectness×class、NMS 与 letterbox 坐标还原。
- 真实产物：RK3588 FP RKNN `20,076,677` 字节；SDK ZIP `70,658,509` 字节。
- SDK 验证：Rockchip 官方 `bus.jpg` 返回 5 个结果并包含 `bus/person`。首个 bus 约为 `(87.95,132.26,555.44,436.18)`，分数 `0.9394`。
- 板端状态：转换机缺少可连接的 NPU 代理，RKNNLite 板端执行待补；原始权重导出、RKNN 构建与 ONNX Runtime SDK 后处理验收已通过。
