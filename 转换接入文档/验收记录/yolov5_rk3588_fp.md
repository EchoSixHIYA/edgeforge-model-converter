# YOLOv5 Detect / RK3588 FP 验收记录

- 验收日期：2026-07-21
- 原始权重：Ultralytics YOLOv5s v7.0 `.pt`，14,808,437 bytes
- 原始权重 SHA-256：`8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee`
- 专用仓库：`airockchip/yolov5`，本地提交 `d25a07534c14f44296f9444bab2aa5c601cdaaab`
- 导出参数：`export.py --rknpu --include onnx --opset 12 --simplify`
- 编译器：RKNN-Toolkit2 2.3.2
- 目标：RK3588，FP

## 产物

| 产物 | 大小 | SHA-256 |
|---|---:|---|
| 友好 ONNX | 28,956,460 bytes | `18a966214958b1c160dfa10a9edea1cf41ff2167a0b70ff1cfb23c170f716274` |
| RKNN | 15,952,836 bytes | `35180b291474eb8f718d01a63aef7322884242e17ea0a4ffa14f8adde40d2b03` |
| SDK ZIP | 55,482,330 bytes | `5de93ae792354b36c5b58df0830837cadfb4967459743af964e150849c88e8f2` |

## ONNX 与 SDK 契约

ONNX 通过结构校验，输出为三个尺度：

```text
[1,255,80,80]
[1,255,40,40]
[1,255,20,20]
```

独立元数据 worker 从用户原始 `.pt` 提取 80 个类别名及像素空间 anchors。SDK `config.json` 本次包含：

```text
[[[10,13],[16,30],[33,23]],
 [[30,61],[62,45],[59,119]],
 [[116,90],[156,198],[373,326]]]
```

后处理按每尺度 3 个 anchor 重排输出，执行 grid/anchor 解码、objectness×类别概率过滤、分类别 NMS 和 letterbox 坐标还原。

## 结果对比

输入官方 `examples/yolov5/model/bus.jpg`，SDK ONNX 参考推理输出 5 个目标并同时包含 `person` 与 `bus`。首个结果：

```text
person @ (210.58, 241.47, 285.03, 519.71) 0.8245
```

与 Model Zoo README 的类别、目标数量和坐标基本一致。RKNN 已真实构建；板端实测仍受当前 `npu_transfer_proxy` 缺失限制。
