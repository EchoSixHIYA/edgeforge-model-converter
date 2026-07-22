# Agent instructions

在接入或修改任何模型仓库前，完整阅读 [MODEL_REPOSITORY_INTEGRATION.md](MODEL_REPOSITORY_INTEGRATION.md)。

不可绕过的要求：

1. 用户上传原始训练权重，服务调用模型专用 exporter 生成厂商友好中间模型。
2. 只有同时完成转换、结构化 SDK 后处理和真实验收的架构才能标记为可用。
3. 下载 SDK 必须包含 SDK 本体、`example.py` 和 `SDK_README.md`；示例必须调用 SDK 本体。
4. 官方仓库示例只能作为 `reference/`，不能作为 SDK 主调用方案。

