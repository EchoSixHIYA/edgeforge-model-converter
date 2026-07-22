# YOLOv8 Detect / RK3588 FP 验收记录

- 验收日期：2026-07-21
- 原始权重：Ultralytics YOLOv8n `.pt`，6,549,796 bytes
- 原始权重 SHA-256：`f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`
- 专用仓库：`airockchip/ultralytics_yolov8`，本地提交 `4674fe6e003dfbc5f2250d3b39dd31faaf7a9877`
- 导出参数：`format=rknn`，opset 12
- 编译器：RKNN-Toolkit2 2.3.2
- 目标：RK3588，FP

## 产物

| 产物 | 大小 | SHA-256 |
|---|---:|---|
| 友好 ONNX | 12,650,245 bytes | `8fb87617be11ead000cc52ce05512eac2bff5332567b9ba270b465e6ea361ddd` |
| RKNN | 7,607,177 bytes | `685a308d974e3e0092d27ed06b0c1f9e5fb81aeee406213d951aa8ef92bebf4c` |
| SDK ZIP | 26,655,396 bytes | `e86ab87ce3ba66b7b31f677a21b043257abff9dbc6bf3a868f7f9dc3bb764f11` |

## ONNX 契约

模型通过 `onnx.checker`。输出共 9 路，每个尺度依次为 DFL box、80 类置信度和 score sum：

```text
[1,64,80,80] [1,80,80,80] [1,1,80,80]
[1,64,40,40] [1,80,40,40] [1,1,40,40]
[1,64,20,20] [1,80,20,20] [1,1,20,20]
```

该模型没有 `names` 元数据，打包器从 RKNN Model Zoo 官方 COCO 标签文件回退得到 80 类。

## SDK 结果对比

输入：`vendor/rknn_model_zoo/examples/yolov8/model/bus.jpg`。

SDK ONNX 参考推理输出 5 个目标：4 个 `person`、1 个 `bus`。首个目标为：

```text
person @ (211.01, 241.51, 283.41, 507.20) 0.8803
```

`bus` 为：

```text
bus @ (99.34, 135.87, 550.54, 456.66) 0.8470
```

类别、数量、坐标和置信度与 Model Zoo README 的预期结果一致，差异处于框架/版本允许范围。自动验收命令要求至少 5 个结果并同时包含 `person`、`bus`。

## 板端状态

RKNN 模型已真实构建。当前 x86 Toolkit wheel 缺少 `npu_transfer_proxy`，设备枚举失败，因此本次没有执行板端 RKNNLite `predict()`。有可访问板卡及完整传输工具后应补跑此项；该限制不影响已完成的 exporter、RKNN 构建和 SDK/官方后处理对比结果。
