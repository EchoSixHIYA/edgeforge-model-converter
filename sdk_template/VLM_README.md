# Cosmos-Reason2-2B / Qwen3-VL SDK（实验性）

这是平台导出的可调用 SDK，不是启动交互式官方 demo。**未在本次开发环境完成真实权重编译、libsophon 原生模块编译、板端推理或数值对齐验收**；单元测试使用固定数据/假设备，只证明逻辑与打包行为。请在本机转换和目标设备上完成文末验收。

## 运行架构与边界

- 原生执行器衍生自 [Sophgo LLM-TPU Qwen3_VL](https://github.com/sophgo/LLM-TPU/tree/2866d403dae89c4164d8b96c8c615d7f67693e6d/models/Qwen3_VL/python_demo)，固定 commit `2866d403dae89c4164d8b96c8c615d7f67693e6d`。保留官方 embedding → ViT（主输出 + 三路 DeepStack）→ 28 层 prefill → lm_head → 单 token decode/KV cache 流程；**不是** CV 的首 graph `sail.Engine.process()`。
- 面向单个 BM1684X、batch=1，静态 no-history 编译配置。支持一个图像或一个视频以及纯文本；暂不支持混合多图/多视频、批处理、跨调用 KV 历史、流式回调。`Qwen3VL` 是相同受限 profile 的底层入口，不声称支持所有 Qwen3-VL 尺寸。
- `model/compilation_manifest.json` 包含源 model_config、编译版本/参数、官方 runtime revision、图名契约和资产 SHA256；`model/processor/` 原样包含 tokenizer/processor/chat/generation 配置。不得仅用六份配置文件冒充完整模型；缺 tokenizer.json 或 vocab+merges 时平台拒绝打包。
- 原生检查要求 `embedding`, `embedding_cache`, `vit`, `add`, `lm_head`, `block_0..27`, `block_cache_0..27`，ViT 四输出、patch16/temporal2/merge2、hidden2048、KV8×128、vocab151936。`--do_sample` 必须保留 raw lm_head logits，不能使用融合 greedy 的模型替代。
- 原官方 C++ 默认 greedy；本 SDK 有明确记录的本地修改：从原始 fp32/fp16/bf16 logits 做 CPU 采样，支持 temperature/top_p/top_k/repetition_penalty/seed/greedy。默认读取用户 generation_config（本模型 0.7 / 0.8 / 20，EOS=[151645,151643]）。不使用 demo 的 `<|end|>` 停止条件，不声称与 HF 随机数流逐 token 一致。保留原始配置和本地修改说明，见 `runtime/qwen3_vl/NOTICE.md`。

## 安装（目标板，或接有 SOPHGO PCIe 卡的 Linux 主机）

纯 WSL/CPU 主机可以运行转换工具，但没有受支持 SOPHGO 设备/驱动不能运行此推理 SDK。

1. 安装与芯片/驱动匹配的官方 **libsophon runtime + development SDK**（头文件、libbmrt、libbmlib），默认路径 `/opt/sophon/libsophon-current`。`sophon.sail` 单独存在并不满足本 SDK；也不要把其他架构的 `.so` 拷过来。上游未给出可验证的最低 libsophon 版本，必须记录自己实际使用版本并进行板端验收。
2. 准备 C++17 编译器、**CMake >=3.19**、Python **3.10+** 与对应 Python 开发头文件。使用独立环境，避免污染平台 CV/转换器环境：

   ```bash
   python3.10 -m venv .venv-vlm
   source .venv-vlm/bin/activate
   # 使用自定义 SDK 目录时：export SOPHON_SDK_ROOT=/path/to/libsophon
   bash install.sh --vlm
   ```

   该脚本安装 CPU `torch==2.9.0 / torchvision==0.24.0`、`transformers==4.57.1`、`av==16.0.1`、`pybind11==2.13.6` 及 SDK，然后在当前 Python 环境编译 `edgeforge_qwen3_vl`。目标系统若没有匹配的官方 PyTorch CPU wheel，需要先按官方平台说明准备这两个匹配版本，不能把安装失败当成推理通过。基础 SDK 允许 tokenizers 0.20–0.22；VLM 的 transformers 依赖会选相容的 0.22 系列。

3. 如 Python 依赖已安装，可单独重建 native 模块：

   ```bash
   PYTHON_BIN="$(command -v python)" bash runtime/qwen3_vl/build.sh
   ```

   此操作会在当前虚拟环境安装 `.so`；不下载权重，不安装系统驱动。构建成功不等于设备推理通过。

## Python API

```python
from edgeforge_sdk import CosmosReason

with CosmosReason("model", device_id=0) as model:
    result = model.generate(
        "Describe what happens and explain why.",
        video="clip.mp4", num_frames=8,
        min_pixels=4096, max_pixels=147456,
        max_new_tokens=256, generation={"seed": 7},
    )
    print(result.text)
    print(result.as_dict())  # token_ids, finish_reason, prompt_tokens, media_metadata
```

- `CosmosReason.load(path)` 和 `predict(...)` 同样可用。`CosmosReason()` 默认读取当前目录的 `model/`；BModel 在 ZIP 中仅保留这一份，不复制进安装 wheel。后续可以只换独立 `model/` 目录。
- 图像：`model.generate(prompt, image="image.jpg", max_pixels=147456)`；纯文本：`model.generate("Hello")`。
- `generation={"do_sample": False}` 显式 greedy；否则默认尊重原 generation_config。支持覆盖 temperature/top_p/top_k/repetition_penalty/eos_token_id/seed；未实现的生成选项明确报错，不会假装生效。
- `max_new_tokens` 是上限；到达 EOS 或编译序列容量即停止，`finish_reason` 为 `eos` 或 `length`。不截断输入，不缓存上一轮内容；若需对话历史，请显式加入 prompt。
- 实例串行化 `preprocess/generate/close`，独占可变 KV 缓存。`preprocess` 返回 PreparedVLMInput，便于检查 patch/position，不能将其送到普通 CV `infer_raw`。

## 视频、时间与预算（不会悄悄改成 demo 的 1 FPS）

- 空间 resize、均值/标准差、插值、patch 排列继续由原 HF processor 配置处理；不会强制 `do_resize=False`。**省略预算时保留源配置**，若结果超过编译上限则明确报错，需要你降低预算或重新编译。
- 可传 `fps`（采样速率）或 `num_frames`，二者互斥；`max_frames` 是显式采样数量上限。都省略时使用该版本官方视频 processor 的默认/配置采样规则，不猜测 1 FPS，也不把“请求 FPS”当成视频真实 FPS。
- PyAV 两遍顺序解码：先读取真实帧数、source FPS 和 presentation timestamps，然后只保留采样帧，避免保存整个视频的全分辨率帧。时间元数据不可用则报错，不回退到 24 FPS。
- `media_metadata` 返回真实 `fps`、原始 `frames_indices`、逐帧 `timestamps`、temporal2 组时间。VFR 使用解码 PTS 计算组时间，而非伪造 indices/FPS。奇数帧末帧重复，与官方 temporal2 行为一致；单帧视频显式重复，记录在 metadata。
- 编译 `max_pixels/256` 是**每次 ViT（一个图像/一个 temporal2 视频组）**的最大 patch 数；合并后的视觉 token 数是 `max_pixels/1024`。视频各 temporal 组分别执行 ViT，其总视觉 token、时间文本、用户文本和生成 token 共同受 `sequence_length` 限制。不存在编译器固定帧数参数。
- `max_pixels` 是每图/每组空间分辨率预算，`min_pixels` 是对应下限；`video_total_pixels` 可额外限制 HF 视频 T×H×W 总预算（含 temporal padding）。不允许下限高于上限。长视频一般需要显式帧数和分辨率预算，不能同时保留源模型的超大默认预算又指望小 BModel 静默容纳。

## 本机验收

```bash
python example.py --model model --prompt 'Hello' --max-new-tokens 8 --greedy
python example.py --model model --prompt 'Describe this image' --image image.jpg --max-pixels 147456
python example.py --model model --prompt 'Describe this video' --video clip.mp4 --num-frames 8 --min-pixels 4096 --max-pixels 147456
```

分别检查：实际芯片/driver/libsophon 版本，图名/shape 契约，文字/图像/含奇数帧视频，source FPS/indices/时间戳，EOS两种停止，输入/输出容量边界、重复调用 KV 重置、与原 HF 模型的 greedy 输出对齐。将真实结果与版本记录下来；没有这些证据，清单中的 hardware_inference_verified 必须继续是 false。
