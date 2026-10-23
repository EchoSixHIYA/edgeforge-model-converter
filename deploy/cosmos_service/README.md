# Cosmos Reason2 板端推理服务

部署目标：BM1684X `192.168.1.117`，项目目录 `/data/cosmos-reason2-service`。独立 Python 3.11 虚拟环境 `.venv`，SDK 解压在 `sdk/`，不替换系统 libsophon/驱动，不注册开机自启。

网页：`http://192.168.1.117:8080/`。支持纯文本、一张图片、一个不超过 30 秒的短视频。图片为 JPEG/PNG/WebP/BMP，视频为 MP4/MOV/MKV/WebM/AVI。上传总大小 ≤32 MiB。单次生成 1–256 token，默认 64；视频采样 2–8 帧，默认 4。上下文 2048、视觉预算 147456；输入超预算时由 SDK 明确拒绝，不静默裁剪。

## 手动启动 / 停止

```bash
cd /data/cosmos-reason2-service
bash start.sh
curl http://127.0.0.1:8080/health
bash stop.sh
```

启动后在后台加载模型，`/health` 中的 `status` 从 `loading` 变为 `ready` 后可推理。状态 `error` 表示加载失败，不代表服务可推理。日志：`logs/server.log`、`logs/access.log`；依赖安装日志和版本也保存在 `logs/`。

只运行一个 Gunicorn worker、四个 HTTP 线程；模型常驻，推理并发 1。模型忙时返回 409，不无限排队。不要增加 worker 数，否则会重复加载模型并占用 TPU 内存。服务无登录鉴权，仅供可信局域网；不映射到公网。上传媒体推理后清理，最近 20 条结果仅在内存中保留，重启即丢失。

## HTTP 接口

- `GET /health`：加载状态、busy、模型参数。HTTP 200 不代表模型 ready，须检查 JSON 的 `status`。
- `GET /v1/models`：模型信息。
- `POST /v1/infer`：同步推理，返回完整 JSON。仅文字可用 JSON；有媒体使用 multipart/form-data。
- `POST /api/jobs`：同样的输入，HTTP 202 返回任务 `id`。
- `GET /api/jobs/<id>`：查询异步任务；`queued/running/completed/failed`。

字段：`model`（默认 `cosmos-reason2-2b`）、必填 `prompt`、可选 `max_new_tokens`（默认 64）、`greedy`（默认 true）；文件字段 `image` 或 `video`，两者互斥。`num_frames` 仅视频适用。不支持跨调用隐式对话历史。

Windows 一行测试（当前目录须有 test.jpg）：

```powershell
curl.exe -sS http://192.168.1.117:8080/v1/infer -F "model=cosmos-reason2-2b" -F "prompt=Describe this image briefly." -F "image=@test.jpg" -F "max_new_tokens=64" -F "greedy=true"
```

纯文本（Linux）：

```bash
curl -sS http://192.168.1.117:8080/v1/infer -H 'Content-Type: application/json' -d '{"prompt":"Say hello briefly.","max_new_tokens":16}'
```

成功返回包含 `ok/model/id/status/elapsed_ms/result`；`result` 内为 `text/token_ids/finish_reason/prompt_tokens/media_metadata`。错误返回 `ok=false/error`，参数错误 400、未知模型 404、模型忙 409、上传超限 413、未就绪 503。网页通过异步接口轮询，可展开完整 JSON，复制或下载结果。

## 初次部署

```bash
python3.11 -m venv .venv
bash prepare_env.sh
# 将网站导出的 sdk.zip 解压到 sdk/ 后：
bash install_sdk.sh
bash start.sh
```

当前板端已准备并验证 CPU torch 2.9.0 / torchvision 0.24.0。`prepare_env.sh` 保留这两个已安装版本，使用清华 PyPI 源安装其余 processor 依赖和虚拟环境内 CMake；新板需先准备匹配的 CPU torch wheel。`install_sdk.sh` 同样使用清华源安装 SDK extras，并调用 SDK 自带构建脚本在板端编译 native 执行器。不安装或升级系统驱动。

清华源地址为 `https://pypi.tuna.tsinghua.edu.cn/simple`，同时写入本项目 `.venv/pip.conf`，不修改用户全局配置。参见[清华 PyPI 镜像说明](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)。

## 2026-10-23 实机记录

板端 Python 3.11.9，libsophon/driver 0.5.1；SDK native 模块在板端编译安装，未替换系统运行库。实际 BModel SHA256：

```text
9b703ca59e7a5b600ae8d54691ff5439826a3702da6345db9a6114ff345d0ef7
```

部署中修正两处 SDK 适配：HF 4.57.1 fast processor 使用 `return_tensors="pt"`，在 native 边界转 NumPy；异步 `bmrt_launch_tensor_ex` 后显式等待完成再读取结果。修正前文字出现重复 token，修正后相同 prompt 返回正常问候。对应修复已同步转换服务器 SDK 模板。

已完成以下实际测试（单次暖机结果，不作为性能基准）：

| 输入 | 实测结果 | 耗时 |
| --- | --- | --- |
| 文字 | `Hello! How can I help you today?`，EOS 正常结束 | 1.58 秒 |
| 图片 | 官方 Model Zoo `bus.jpg`，描述行人与蓝色公交车，40 个输出 token，EOS 正常结束 | 3.23 秒 |
| 视频通路 | 由上述照片编码的四帧静态回放，返回 `Men are standing near a bus.`，帧时间戳和两组视觉输入正常 | 1.52 秒 |

手机/桌面 Chromium 实际提交图片、完整 JSON 展示、页面无横向溢出检查通过；HTTP 层 14 项单元测试通过。这里未做原始 HF 模型逐 token 数值对齐，也不把静态回放当作真实运动理解验收。模型包内的实验性标记仍保留。

`tests/browser_smoke.cjs` 是实际板端浏览器测试，不模拟推理结果。使用已安装的 Playwright，设置 `COSMOS_TEST_IMAGE` 为真实照片路径；必要时通过 `PLAYWRIGHT_MODULE` 指定 Playwright 模块路径。输出截图和接口返回到本地 `evidence/`。
