# YOLO11 Detect / RK3588 FP 验收记录

- 验收日期：2026-07-21
- 原始权重：Ultralytics YOLO11n `.pt`，5,613,764 bytes
- 原始权重 SHA-256：`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`
- 专用仓库：`airockchip/ultralytics_yolo11`，本地提交 `0692e9297670acf4cc6d0cec773d7a9493cb8a5f`
- 导出参数：`format=rknn`，opset 12
- 编译器：RKNN-Toolkit2 2.3.2
- 目标：RK3588，FP

## 产物

| 产物 | 大小 | SHA-256 |
|---|---:|---|
| 友好 ONNX | 10,527,859 bytes | `6f2b53b150251766b5b4254ca2caf7c66b6c7c4b94b4bd207ea3dbc9efb6565c` |
| RKNN | 7,030,857 bytes | `5fb61fcc38dbb6cc982fa50bef16594d26dd65b2a4a282b8f93b63b6ca0d7b45` |
| SDK ZIP | 23,005,963 bytes | `3eb9f9a041d05b7278ca145b910cbd3033fe4afcd8f49146d15ada6249ace7a9` |

## 验收结论

友好 ONNX 通过结构校验，包含官方约定的 9 路输出：三个尺度分别输出 64 通道 DFL box、80 类置信度和 1 通道 score sum。模型没有 `names` 元数据，打包器正确回退官方 COCO 标签文件。

生成 SDK 使用官方 `examples/yolo11/model/bus.jpg` 做 ONNX 参考推理，输出 5 个目标且同时包含 `person` 和 `bus`。首个目标为：

```text
person @ (108.81, 236.47, 223.43, 535.84) 0.8956
```

自动验收使用 `--min-results 5 --expect-label person --expect-label bus`，验证通过。RKNN 已真实构建；板端 RKNNLite 实测受当前 `npu_transfer_proxy` 缺失限制，状态与 YOLOv8 验收记录相同。
