# Cosmos Reason 2 2B → BM1684X（实验性本地链路）

此入口接入 **NVIDIA Cosmos Reason 2 2B 的 Qwen3-VL 结构**，不是 Qwen-Image 文生图模型，也不是把一个 `[1,3,H,W]` ONNX 交给 CV 后端。

**交付边界：代码集成及无硬件自动测试，不代表真实权重已成功编译、量化质量合格或板端推理已验收。** 目录仍标记 `available=false / verified=false / experimental=true`；用户可显式确认后从独立实验入口运行。原有 YOLO、OCR、CLIP、RKNN 与 CV BModel 路线不变。

## 1. 固定官方方案与适配范围

- 编译器：[`sophgo/tpu-mlir` 1.28.1](https://github.com/sophgo/tpu-mlir/tree/43676b3fcf07f44158e63baaefe6c0e809d16890)，实际调用包内 `python/tools/llm_convert.py`，由官方分发到 `Qwen3VLConverter`。
- 推理参考：[`sophgo/LLM-TPU/models/Qwen3_VL`](https://github.com/sophgo/LLM-TPU/tree/2866d403dae89c4164d8b96c8c615d7f67693e6d/models/Qwen3_VL)，SDK 与 native runtime 版本见生成包 `VLM_README.md` 和编译清单。
- 接入原始 BF16 safetensors：28 层、hidden=2048、16 Q / 8 KV、head=128、ViT 24 层、DeepStack `[5,11,17]`、interleaved MRoPE `[24,20,20]`。不自动接收同名不同结构、预量化、任意 remote code 或 pickle 权重。
- 目前只允许 BM1684X、单设备、batch=1、静态图、无跨调用 KV 历史、无 CPU embedding 外置。其他芯片、dynamic、embedding_disk、LoRA 等没有在此入口开放。
- 显式 `-q w4bf16 / w8bf16 / bf16`；**不使用 auto**，原始配置没有 `quantization_config` 时官方 auto 会报错。
- 始终加 `--do_sample`，保留 raw `lm_head` logits、greedy/sample head。SDK 依据真实 `generation_config.json` 与调用参数执行生成；不能把默认 greedy 冒充 temperature/top-p/top-k。
- `--only_mlir` 在该版本仅导出第 0 语言层用于分析，不是全模型导出；没有 `--dry_run`。本平台的 `--preflight-only` 是自己的文件/环境检查，绝不伪装成官方转换成功。

## 2. 环境隔离

在 Linux x86_64 / WSL2 的**本机**运行。编译不要求 GPU，但内存和临时磁盘需求较大。启动服务的基础环境沿用项目依赖；不要把大模型依赖升级进原 RKNN/Paddle/TF/CV 环境。

官方 1.28.1 PyPI wheel 使用 Ubuntu 22.04 / glibc >= 2.35 与 Python 3.10，并固定 Transformers 5.3.0。Ubuntu-20.04 的 glibc 2.31 不适用于本链路：wheel 注入的 glibc 库会与旧系统加载器混用，使 GNU parallel、model_tool 等进程崩溃。安装器和 worker 会在导入工具包之前明确拒绝此环境。本仓库提供显式运行的独立安装器（不自动执行、不下载模型、不装系统包）：

```bash
# 先由系统管理员安装 Python 3.10、venv、GNU parallel 等。
# 例：Ubuntu 对应发行版的软件源中安装 python3.10-venv 与 parallel。
COSMOS_PYTHON=python3.10 bash scripts/setup_cosmos_compiler.sh
export EDGEFORGE_LLM_PYTHON="$PWD/.venv-cosmos-compiler/bin/python"
```

安装器选择 CPU Torch 2.4.0 / torchvision 0.19.0、`tpu_mlir==1.28.1`、`transformers==5.3.0`、`setuptools==75.8.0`。不使用 `tpu_mlir[torch]`（旧 Torch extra 与 Transformers 5.3 的要求冲突），固定 setuptools 保留官方初始化所需的 `pkg_resources`。安装后 `pip check` 与 worker probe 必须通过。**这组环境在本次云执行中未完整安装或实跑模型，真实兼容性仍须本机验证。**

2026-10-23 本机补充：Ubuntu-22.04 的独立编译环境已完成上述固定版本安装，`pip check` 和真实 worker `--probe` 通过；原始 Cosmos 权重目录通过预检。此记录不等同于完整模型编译或板端推理成功。

worker 先调用官方包初始化工具路径，再启动新的 Python 进程让 `LD_LIBRARY_PATH` 生效，然后导入真实编译器模块。Web/CLI 只启动隔离解释器，不导入这些依赖到 Flask 内。

板端 SDK 使用自己的隔离推理依赖及匹配的 libsophon C++ runtime/dev 与驱动；**不是仅 `pip install sophon.sail` 就能运行**。请以生成 SDK 的 `VLM_README.md` 为准，不与编译 venv 混装。

## 3. 准备完整本地 HF 目录

用户自行获取合法授权权重；平台不会下载、替换成另一个 Qwen 模型或伪造权重。使用完整物化的 local-dir，而不是指向 HF cache 外部文件的 symlink 快照。

```text
/mnt/models/Cosmos-Reason2-2B/
  config.json
  model.safetensors                 # 或完整分片 + model.safetensors.index.json
  tokenizer_config.json
  tokenizer.json                    # 或 vocab.json + merges.txt
  preprocessor_config.json
  video_preprocessor_config.json
  generation_config.json
  chat_template.json                # 或 chat_template.jinja
```

仅六个配置 JSON 不够；缺词表、merge 表或权重时预检会明确拒绝。禁止目录内符号链接、`.py`、`.bin`、`.pt`、`.pth`、pickle 等。单个 tokenizer/processor 配套文件最多 128 MiB。分片索引不能包含路径穿越。Safetensors 预检只读有界头部，检查数据偏移、重复 tensor、关键权重 shape、28 语言层和视觉/DeepStack 映射；这不是所有权重数值/精度的验证。

**从预检到转换结束请勿修改模型源目录。** 平台记录文件 inode/size/mtime，并在 staging 前后和编译结束核对。任务目录保存冻结配置副本，权重只建立任务私有只读用途链接，避免额外复制数 GB 权重；SDK 只打包冻结配置与编译模型，不打包原始权重。

## 4. 命令行（无需上传数 GB 权重）

```bash
export EDGEFORGE_LLM_PYTHON="$PWD/.venv-cosmos-compiler/bin/python"
export EDGEFORGE_MODEL_ROOTS="/mnt/models"

# 此命令不会导出或编译；失败返回非零退出码。
.venv/bin/python scripts/convert_cosmos.py \
  --model-dir /mnt/models/Cosmos-Reason2-2B --preflight-only

# 真正调用官方完整导出→量化编译→combined BModel→平台 SDK 打包。
.venv/bin/python scripts/convert_cosmos.py \
  --model-dir /mnt/models/Cosmos-Reason2-2B \
  --precision w4bf16 --seq-length 2048 --max-pixels 147456 --jobs 1 \
  --acknowledge-experimental
```

也可用 `--allow-root /mnt/models` 显式授权 CLI 读取根目录。多根环境变量用 Linux `:` 分隔。CLI 默认输出 `data/cosmos-cli/<task-id>/`，与 Web 的队列目录分开；不要同时用多个 Web/CLI 实例对同一任务目录操作，也不要同时启动大模型编译争用内存。CLI 同目录锁可防止两次 CLI 同时运行。Ctrl-C 请求取消并等待编译进程组终止。

官方编译器内部有 shell 命令；**任务输出工作路径须只含 ASCII 字母、数字、`_-. /` 中的非空格字符**。仓库路径含空格/中文时，请设置 `--output-dir /tmp/edgeforge-cosmos` 或 Web 的 `EDGEFORGE_DATA_DIR=/tmp/edgeforge-tasks`。原模型源目录可位于其他允许的目录，staging 路径会标准化。

## 5. 网页与安全范围

```bash
export EDGEFORGE_MODEL_ROOTS="/mnt/models"
export EDGEFORGE_LLM_PYTHON="$PWD/.venv-cosmos-compiler/bin/python"
export EDGEFORGE_DATA_DIR=/tmp/edgeforge-tasks
.venv/bin/gunicorn --workers 1 --threads 8 --bind 127.0.0.1:5000 app:app
```

打开 `http://localhost:5000`，选择 Sophon / Cosmos Reason 2 2B 实验入口，填写**服务所在机器**的绝对模型目录，先预检，再勾选实验风险确认并开始。浏览器文件选择器不是服务器目录授权，不能填另一台电脑的路径。

- 未设置 `EDGEFORGE_MODEL_ROOTS` 时导入禁用；不要设为 `/`。
- 新 VLM API 只接受 loopback 客户端、localhost/127.0.0.1 Host、同源 Origin；POST 必须带 `X-EdgeForge-Local-Import: 1`。不信任伪造 `X-Forwarded-For`。
- VLM 的列表、状态、SSE、日志和产物下载对非本地请求也会拒绝/隐藏；原 CV API 保持原行为。反向代理不属于当前安全契约，不要把此服务暴露公网。
- 输入是目录，不走现有 4 GiB HTTP 文件上传限制。VLM 不支持普通 ONNX、单独 safetensors 或 ZIP 上传。

### JSON API

```bash
curl -sS http://localhost:5000/api/vlm/preflight \
  -H 'Content-Type: application/json' -H 'X-EdgeForge-Local-Import: 1' \
  --data '{"model_dir":"/mnt/models/Cosmos-Reason2-2B","platform":"bm1684x","precision":"w4bf16","seq_length":2048,"max_pixels":147456,"jobs":1}'
```

创建任务使用相同请求的 `/api/vlm/tasks`，加 `"acknowledge_experimental":true`。成功只表示入队（202），不是转换成功。

- `GET /api/tasks/<id>` / `events`：状态、阶段消息及有界 `log_tail`。
- `POST /api/tasks/<id>/cancel`：同样需本地自定义请求头；仅支持新 VLM，旧 CV 不会假装可安全取消。
- `GET /api/tasks/<id>/download/log`：日志存在后可读取，包括运行/失败/取消。
- 完成后 `/download/model`、`/download/package`、`/download/manifest`。
- 状态：queued → running → completed/failed；取消 running → canceling → canceled。重启中断任务标为 failed，不能自动当成成功或默默续编。
- **强制杀死服务（SIGKILL）/系统崩溃不等于显式取消：** 编译子进程组可能继续运行，重启目前只标记任务 failed，不会自动杀死可能已复用的 PID。重新提交前须检查并停止上一编译任务，避免重复占用 RAM/磁盘；`compiler_process.json` 记录 PID/PGID 供核对，不能未经身份检查直接照抄 kill。需要停止时优先使用网页取消或 CLI Ctrl-C，再关闭服务。

## 6. 输入与资源约束

| 参数 | 默认 / 范围 |
|---|---|
| `seq_length` | 2048；256–8192 的 128 倍数 |
| `max_pixels` | 147456（384×384）；65536–1048576 的 1024 倍数 |
| `jobs` | 只允许 1 |
| `precision` | w4bf16 / w8bf16 / bf16 |

`max_pixels` 是每次视觉塔调用的像素预算，不是边长，也不是固定最大视频帧数。patch=16、spatial merge=2，所以默认最多 576 patches / 144 merged visual tokens。视频 temporal patch=2，需要结合真实采样帧数、时间戳和帧分辨率按组处理；全部视觉 token + 文本 + 生成 token 仍不能超过序列上限。不同帧数不能共用一个凭空指定的“最大帧数”。SDK 必须按编译清单校验预算，超限不能静默裁剪、伪造 fps 或忽略输入。

官方 Qwen3-VL converter 使用 scalar `max_pixels`；本入口不传无效的 max_shape/frames 参数。静态 ViT 的 5 个输入 shape 与 4 路输出、所需 graph 名称记录在 manifest；完成编译会检查官方 `model_tool --info` 的 graph 名称，实际 tensor shape/板端运行由 SDK 再验证，不能把预期 shape 当实测结果。

**jobs=1 真正控制两个层面：** wrapper 暂时启用官方 `gen_all_mlir()` 的串行分支并恢复原 debug 设置；同时覆盖 `converter.max_workers=1`，使官方 `GNU parallel -j 1` 生效。不仅仅设置 OMP，也不修改已安装官方源码。仍需保留足够单图编译内存。

**编译主机资源**预检显示：运行 EdgeForge 的电脑当前可用 RAM（含 cgroup 限额）、工作盘余量、权重大小，以及 `max(20 GiB, 权重大小×6)` 的保守工作盘估算。低于该估算或主机可用 RAM <6 GiB 时拒绝启动；这是**编译电脑**的最低拦截阈值，不是板端容量要求，也**不是足够编译的保证或实测峰值**。大序列/视觉塔可能显著增加峰值，需要预留更多资源。日志、临时 MLIR/NPZ、子图 BModel 会留在任务目录用于排障，用户确认后自行清理。

### 6.1 板端内存：计算项不等于实际峰值

预检和生成 SDK 的 `compilation_manifest.json` 增加 `device_memory_estimate`。它始终记录 `fit_assessed=false`；未测量的峰值、TPU heap 和板端 Linux total/available 为 `null`，**不会因为主机预检通过就判断板端可运行**。

必须分开三种口径：

1. **编译主机 RAM**：在转换电脑上供 exporter/compiler 使用。
2. **板端 Linux `free` total/available**：当前内核可见的系统内存，不必等于整板物理 DDR，也不等于 TPU 可用容量。资料中的物理容量与 `free` 读数不同不能直接选一个当 TPU 限额。
3. **实际 TPU heap**：受 BSP、TPU/VPU/VPP/固件保留分区与当前占用影响，须用匹配 BSP 的官方工具/分区配置核对。若不同产品资料也相互冲突，应先确认实物版本和分区，不能写死成某个 6/8/12/16 GB 型号。

**本次设备的一次匿名采样样例（不能作为所有板卡配置）：** Linux `free` total=5,820,072 KiB（约5.55 GiB）、available=4,821,884 KiB（约4.60 GiB）；另行实测 NPU 池 size=7,984,906,240 bytes（约7.436 GiB）、used=132,902,912 bytes，二者差额=7,852,003,328 bytes（约7.312 GiB）。这说明 Linux 与 NPU 池确实是不同口径，不能把前一个读数叫作“全板只有6GB”。物理 DDR 规格资料仍有冲突，不据此断言是12GB或16GB。

该差额只是采样时 `size-used`，不是长期保留量、任意大小连续分配保证或未来推理可用量；当时工具显示的 peak=132,902,912 bytes **不是 Cosmos 加载/推理峰值**。此记录支持继续做 W4 实机试验，但不保证模型可运行。代码不会把这些样例数字写入任意目标设备的估算：API/manifest 仍明确未自动探测容量，运行前应重新采样实际目标板。

以下按本入口固定的 2B 配置及官方实现推导，不读取设备、不代表实测：

| 已列项 | 计算 |
|---|---|
| Decoder 可量化矩阵 | 1,409,286,144 个参数，W4 为 704,643,072 bytes |
| W4/W8 group=64 元数据 | 每组 BF16 scale 2 bytes + UINT8 zero point 1 byte，共 66,060,288 bytes |
| Embedding | 311,164,928 个参数 × 2 bytes |
| ViT + 主 merger + 3 个 DeepStack merger | 406,957,056 个参数 × 2 bytes，**不按 W4** |
| Decoder/最终 norms | 123,904 个参数 × 2 bytes |
| KV cache | `2 × 28层 × 8 KV头 × 128维 × 2 bytes × seq_length` |
| 官方显式额外 device buffers | `(1 + 3个DeepStack) × seq_length × 2048 × 2 bytes` |

W4BF16 在 **seq=1024** 时，KV=112 MiB、上述显式额外 buffers=16 MiB：

- 假设 combined BModel 的 embedding / tied LM head 共享一份权重：权重约 **2.056 GiB**，已列项合计约 **2.181 GiB**。
- 假设 LM head 另存一份：权重约 **2.635 GiB**，已列项合计约 **2.760 GiB**。

这是两种**存储假设**，不是实测上下界；不能把公开总参数数直接除以 2 当 W4 峰值。量化元数据来自固定版本 [A16MatMul 实现](https://github.com/sophgo/tpu-mlir/blob/43676b3fcf07f44158e63baaefe6c0e809d16890/lib/Dialect/Tpu/Canonicalize/A16MatMul.cpp#L332-L410)，图精度与共享意图见 [官方 LlmConverter](https://github.com/sophgo/tpu-mlir/blob/43676b3fcf07f44158e63baaefe6c0e809d16890/python/llm/LlmConverter.py)。

**没有算进合计：**编译器/运行时 workspace、ViT 激活与注意力临时张量、图输入输出、对齐、可能的重复权重/KV分配，以及 Linux 应用、CPU tokenizer、图片/视频解码和 processor RAM。因此这些数字不能直接与 `free.available` 比较后宣称能跑。减少 `max_pixels` 会影响尚未估算的视觉激活/工作区；上表权重与 KV 公式不随像素预算改变，并不表示视觉分辨率没有内存影响。

### 6.2 低内存试验预设（不改变默认、不保证适配）

网页“低内存试验预设”只填写本次 `w4bf16 / seq_length=1024 / max_pixels=65536 / jobs=1` 并使旧预检失效；全局默认仍是 2048 / 147456。它不是“6 GB 可运行”认证。

```bash
.venv/bin/python scripts/convert_cosmos.py \
  --model-dir /mnt/models/Cosmos-Reason2-2B \
  --precision w4bf16 --seq-length 1024 --max-pixels 65536 --jobs 1 \
  --acknowledge-experimental
```

实际板端应先确认 TPU heap，再按小步验证并记录真实系统/TPU 峰值；任何 OOM、图 shape 不符或 token 超限都停止本轮，不增加负载：

```bash
# 在按 VLM_README.md 安装好的生成 SDK 中，先做一张图片。
python example.py --model model --prompt 'Describe the image briefly.' \
  --image image.jpg --max-pixels 65536 --max-new-tokens 64

# 只有单图通过后，才试短视频的4个实际采样帧；不是保证4帧能跑。
python example.py --model model --prompt 'Describe the short clip briefly.' \
  --video short.mp4 --num-frames 4 --max-frames 4 \
  --min-pixels 4096 --max-pixels 65536 --video-total-pixels 262144 \
  --max-new-tokens 64
```

视频仍需实际时间戳、temporal2 分组和整个 prompt 的序列预算检查；帧数少、分辨率低也不能代替真实 heap/峰值核验。通过后再逐项增加输出长度或媒体预算，不同时扩大多个参数。

## 7. 产物与验收

每个成功任务包含实际 `model.bmodel`、平台生成 SDK ZIP、`compilation_manifest.json`、`model_info.txt`、`conversion.log`、`preflight.json`、`command.json`。SDK 的完整词表、processor、chat 模板、generation 配置随包保存；清单记录版本、芯片、精度、图契约、输入预算和实验状态。

成功必须满足：官方完整命令 exit=0、唯一非空 combined `.bmodel`、官方 model info 包含必需 graphs、SDK 打包成功。任何失败都不能发布“成功”产物。取消会给整个子进程组 SIGTERM，宽限后 SIGKILL（即使主进程先退出），部分文件不开放为成功下载。

本仓库自动化测试覆盖 schema、目录安全、缺权重/编译器失败、任务路由、取消语义、worker 串行控制、manifest 和 SDK 假 runtime 行为。**mock 只证明代码分支；不证明导出、编译、内存峰值、质量或硬件推理。**

```bash
# 建议另建轻量测试 venv；不需要安装 Torch、TPU-MLIR、RKNN 或板卡驱动。
python3 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-test.txt
.venv-test/bin/python -m pytest -q -ra
```

每次报告保留 passed / skipped / failed 和 `-ra` 原因；不能把 `145 passed, 1 skipped` 写成 146/146 全通过。CI 范围、各 skip 条件和 Windows 已有 checkout 的处理见 [测试与验收说明](testing.md)。

本机下一步验收顺序：完整权重预检 → 实际导出/编译 → 校对 combined graphs → 生成包安装 → BM1684X 图片/视频生成 → 与原模型相同采样/停止参数对比。只有这些实测通过后，才能另行提高目录的 verified/available 状态。
