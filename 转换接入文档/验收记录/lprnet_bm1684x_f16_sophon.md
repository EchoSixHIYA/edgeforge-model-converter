# LPRNet / BM1684X F16 算能转换验收记录

- 任务 ID：`09a9448a3a15495db82e1bf7d52f2da1`
- 来源权重：`sirius-ai/LPRNet_Pytorch/weights/Final_LPRNet_model.pth`
- SHA256：`D1CA22CA7A251A90702A166C8AD8CE7F05ACB39F5EF8055139A33D462716D1AD`
- 目标：Sophon BM1684X
- 精度：F16
- 输入：`[1, 3, 24, 94]`

## 转换链路

```text
Final_LPRNet_model.pth
  → sophon-demo/sample/LPRNet/tools/LPRNet.py
  → model.onnx
  → TPU-MLIR model_transform.py
  → lprnet.mlir
  → model_deploy.py --quantize F16 --chip bm1684x
  → model.bmodel
```

## 结果

- ONNX 导出成功并通过结构校验。
- BModel 生成成功，文件大小约 5.9 MB。
- SDK 包生成成功：`lprnet_bm1684x_f16_sdk.zip`，约 9.6 MB。
- `scripts/smoke_bmodel.py` 包结构检查通过。
- SDK 包含 `edgeforge_sdk/`、`model/model.bmodel`、`example.py`、`SDK_README.md` 和官方 `reference/`。

## 板端状态

本次尝试连接 BM1684X 板端 `192.168.1.67` 时 SSH 超时，因此尚未把本任务标记为“板端推理验收通过”。BModel 和 SDK 已留在任务目录，板端恢复后可直接执行 SDK 包中的 `example.py` 完成最终验收。
