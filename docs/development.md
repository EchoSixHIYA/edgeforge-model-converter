# dev 分支本机测试

`dev` 基于 PR #1 的 `1982b89` 提交建立，用于本机测试与修改 Cosmos Reason2 / Qwen3-VL 转换流程。`main` 保留原版本。切换分支前请先保存工作区改动；本地 `dev` 不跟踪 PR 源分支，避免修改时误推送到该分支。

## 启动

在 Windows PowerShell 中执行：

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "cd '/mnt/e/边缘部署/模型转换服务器' && bash scripts/start_dev_wsl.sh"
```

宿主机打开 `http://localhost:5000`。进程前台运行，按 Ctrl+C 停止；本项目没有新增开机自启设置。

Cosmos 编译请使用 Ubuntu-22.04（glibc >= 2.35）。Ubuntu-20.04 可以运行原 Web/RKNN 流程，但其系统加载器与固定 TPU-MLIR 1.28.1 wheel 的 glibc 不兼容，可能使 GNU parallel、model_tool 等进程崩溃。仅升级 Python 或补装 parallel 不能解决这个系统兼容问题。

启动脚本沿用已有 Web 虚拟环境，默认任务目录为 `/home/当前用户/.edgeforge-model-converter-dev/tasks`。该目录与原 `data/tasks` 隔离，并使用纯英文路径，满足官方 VLM 编译器内部 shell 对工作路径的要求。可用 `EDGEFORGE_DEV_STATE` 改变开发数据根目录，或用 `EDGEFORGE_DATA_DIR` 明确指定任务目录。两者均应使用服务可写的 ASCII 路径。

服务启动后不会自动安装编译器、下载模型或创建转换任务。编辑 Python 后手动重启服务即可生效；请在当前转换结束或取消完成后重启。

## 启用本地 HF 目录测试

只有管理员明确授权的模型根目录才可导入。配置示例：

```bash
cd /mnt/e/边缘部署/模型转换服务器
export EDGEFORGE_MODEL_ROOTS=/mnt/d/models
export EDGEFORGE_LLM_PYTHON=/path/to/cosmos-compiler-venv/bin/python
bash scripts/start_dev_wsl.sh
```

将示例路径替换为实际模型根目录和独立编译环境。模型目录应包含完整原始 safetensors 权重与 tokenizer/processor。浏览器输入该根目录下服务可见的模型绝对路径，点击预检。

本机长期使用的环境变量可保存为项目内 `data/dev.env`，每行使用 `export 变量名=值` 的 Bash 语法。开发启动脚本会读取该文件；`data/` 已被 Git 忽略，因此本机路径不会进入提交。也可通过 `EDGEFORGE_DEV_CONFIG` 指定其他可信的本地配置文件。

`EDGEFORGE_LLM_PYTHON` 指向专门的 Python 3.10 环境，需通过 `server/vlm_worker.py --probe`。编译环境要求见 [Cosmos 转换指南](cosmos_reason2.md)；`scripts/setup_cosmos_compiler.sh` 是手动安装入口。单纯启动网站或通过轻量测试不表示已满足真实编译条件。

## 轻量测试

WSL 中运行：

```bash
cd /mnt/e/边缘部署/模型转换服务器
/home/root/.edgeforge-model-converter-venv-py310/bin/python -m pytest -q -ra
```

上面是当前本机已有的虚拟环境路径。其他电脑应先按 [测试指南](testing.md) 建立独立环境。测试命令会报告实际 skipped 原因。

Windows 项目根目录中可单独运行 JavaScript 和浏览器回归：

```powershell
npm ci --prefix tests/ui
node tests/ui/vlm_dom_unit.cjs
node tests/ui/node_modules/playwright/cli.js install chromium
$env:EDGEFORGE_UI_URL = 'http://localhost:5000'
node tests/ui/vlm_workflow.cjs
```

浏览器回归会模拟任务接口，不会执行模型编译。截图和 trace 默认输出到系统临时目录的 `edgeforge-vlm-ui` 下；可用 `EDGEFORGE_UI_OUTPUT` 指定输出目录。真实转换、SDK 原生模块构建与板端推理仍需分别测试。

GitHub CI 已包含 `dev` 的 push 触发；本地提交后推送 `dev` 才会触发远程检查。
