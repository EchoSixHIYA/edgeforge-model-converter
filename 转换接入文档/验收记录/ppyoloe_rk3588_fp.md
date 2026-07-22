# PP-YOLOE / RK3588 / FP 验收记录

- 权威训练仓库：PaddleDetection `release/2.5`，固定提交 `e3f8dd16bffca04060ec1edc388c5a618e15bbf8`，应用 RKNN Model Zoo 随附 `ppyoloe_rknn_optimize.patch`。
- 导出环境：Python 3.8、PaddlePaddle 2.5.2、Paddle2ONNX 1.0.9；环境独立于主服务，系统依赖为 `libgomp1`。
- 输入契约：RGB、固定 `640×640` letterbox、mean `[0,0,0]`、std `[255,255,255]`。
- ONNX 契约：每个 20/40/80 尺度依次输出 68 通道 DFL、80 类概率和 1 通道 score-sum，共 9 路；batch 维在导出后固定为 1。
- SDK 契约：显式 `PPYoloE("model")`；按 PP-YOLOE 自身 17-bin DFL 解码，忽略仅用于加速筛选的 score-sum，执行分类别 NMS，返回通用 `Detection`。

## PP-YOLOE+ CRN-S

- 原始权重：`ppyoloe_plus_crn_s_80e_coco.pdparams`，大小 `32,466,105` 字节，SHA-256 `9ab2b1fb7ddec42a0bb6cb544b0ac627c2d0054f0655bf402cf7c8f4def3219c`。
- ONNX：`31,837,617` 字节，SHA-256 `fd6093bc95bb19116e5410d215aaaae2b57c1c430f12f1cc5290122dca3a797b`。
- RKNN：`17,913,025` 字节，SHA-256 `ac2dbc6c1e68a4788cfbe7261038d39b6a5b18b2ee26cf4d6bb3978278d35576`。
- SDK ZIP：`62,980,003` 字节，SHA-256 `1fffa2ad96ebfe4127ae92712f17f33b07601055a8eb43a097b70a2c9abb2974`。
- 官方 bus 图：返回 9 个结果；首个 person 框 `[108.62,234.18,224.98,536.55]`、分数 `0.947052`，与 Zoo 参考 `(108,234,225,536) 0.947` 一致，并检测到 bus。

## PP-YOLOE+ CRN-M

- 原始权重：`ppyoloe_plus_crn_m_80e_coco.pdparams`，大小 `96,769,306` 字节，SHA-256 `3470057adb1eeb1d01c898e3ac944c1346533fefcd617044e75ffdfeb1df8528`。
- ONNX：`93,904,657` 字节，SHA-256 `5263245a46d5dd40bee892bfacbe48be11b3874e82c823874cd27296f50ec975`。
- RKNN：`49,907,521` 字节，SHA-256 `1258f259c54c04112ecebfe49c54c3ad49504e6af2ad4ad2d616eed62400f1d1`。
- SDK ZIP：`178,362,189` 字节，SHA-256 `d8f0a20aa6577cb028298bfd55f9608c1ebf0ab7d762b434b9ea4366e1d1f16d`。
- 官方 bus 图：返回 8 个结果；首个 person 框 `[109.35,235.65,225.88,536.02]`、分数 `0.961626`，并检测到 bus。

- 板端状态：转换机缺少可连接的 NPU 代理，RKNNLite 板端执行待补；S/M 原始权重导出、RKNN 构建和 ONNX Runtime SDK 后处理均已通过。
