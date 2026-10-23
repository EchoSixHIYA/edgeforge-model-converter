# 轻量测试与真实验收边界

## GitHub Actions 检查范围

[Lightweight CI](../.github/workflows/lightweight-ci.yml) 对目标为 `main` 的
`pull_request`，以及 `main` / `dev` / `feat/cosmos-reason2-sophon-20261022` 的 `push`
运行。保留 feature 分支 push 作为首次接入的验证入口，因此同一 PR 提交可能同时
产生 push 和 pull_request 两个有超时限制的 run；二者不是两份不同的验收证据。
只有 `contents: read`，checkout 不保留凭据，无 secrets、无
`pull_request_target`，不访问模型目录、真实权重、TPU 编译器或板卡。

| 检查 | 实际覆盖 | 不代表 |
|---|---|---|
| Linux hardware-free tests | Python 3.10 全量 pytest、g++ 真实编译并执行 CPU sampling 辅助测试、Node DOM 单元、Python/JS/bash 语法 | 完整 libsophon 原生模块构建、真实权重导出/编译 |
| Windows LF checkout and shell regression | `core.autocrlf=true` checkout、全部 tracked `.sh` 原始字节及 Git Bash `bash -n`、Node DOM 单元 | Windows 原生运行 TPU-MLIR；也不是 Windows 全量 pytest |
| Chromium UI regression (mocked conversion APIs) | 固定 Playwright + 官方安装的 Chromium，真实 Flask 页面和浏览器事件、截图/trace；转换任务 API 被 mock | 模型编译/推理端到端测试，模拟“完成”不是 BModel 产物 |

Linux/浏览器任务上限各 15 分钟，Windows 上限 10 分钟。Flask readiness 上限
30 秒，浏览器命令上限 180 秒，页面操作另有超时；退出时清理本次 Flask 服务。
浏览器只绑定 `127.0.0.1`，不授权任何本地模型根目录。

pytest 使用 `-q -ra`，保留所有非通过项的原因。日志/JUnit 和浏览器截图/trace
在 Actions artifacts 保留 14 天；查看**对应 commit 的实际 job 结果**，不能因
workflow 文件存在、某个 job 通过或本地检查通过就声称 CI 全绿。Linux CI 在提供
全部先决工具后要求 JUnit 的 skipped=0；Windows 仅对所选 checkout 回归要求
skipped=0，并非声称全量跨平台测试无 skip。本地/非 Linux 的合理 skip 仍按原样报告。浏览器依赖安装
按照 [Playwright 官方 CI 文档](https://playwright.dev/docs/ci) 执行。

提供 `workflow_dispatch` 供将来手动触发，但 [GitHub 要求 workflow 已存在于默认分支](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)。
首次尚未合并的 Draft PR 依赖 `pull_request` / `push` 自动触发或重跑已有 run，
不能承诺此时已有 “Run workflow” 按钮；不应为启用按钮而提前合并。

## 本地复现与 skip 诊断

```bash
python3 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-test.txt
.venv-test/bin/python -m pytest -q -ra --junitxml=/tmp/edgeforge-pytest.xml
node tests/ui/vlm_dom_unit.cjs
node --check static/app.js
# 只解析脚本；绝不执行这些安装器或板端探测脚本。
while IFS= read -r -d '' script; do bash -n "$script"; done < <(git ls-files -z '*.sh')
```

pytest 不是硬件验收；即使全部通过也只表示这些测试通过。`-ra` 包含 skip 原因；
只想查看 skip 可用 `python -m pytest -q -rs`。报告应同时记录 commit、操作系统、
Python/Node/g++ 版本、命令、passed/skipped/failed 数量和 skip 节点/理由。

当前显式 skip 条件：

- `tests/test_vlm_pipeline.py::test_cancel_kills_sigterm_ignoring_descendant_even_if_leader_exits`：
  要用 Linux `/proc` 观察后代终止并使用 POSIX 进程组，因此非 Linux 平台跳过。
  Windows 上这是平台约束，不应改为“通过”。旧版本只检查 `os.name != "posix"`；
  本次收紧为 Linux，避免其他 POSIX 系统没有 `/proc` 却误把未观察到进程当成成功。
- `tests/test_vlm_sdk.py::test_native_sampling_cpp`：没有 `g++` 时跳过；不是完整
  `qwen3_vl_native` 模块构建测试。Linux CI 显式安装并打印 g++ 版本。
- `tests/test_vlm_ui.py::test_vlm_javascript_logic_with_dom_stubs`：没有 Node.js 时跳过。
  Linux CI 显式安装 Node；DOM 单元不是 Playwright。
- `tests/test_shell_portability.py`：缺 Git 时 checkout 测试不可运行；缺 Bash 时只
  跳过 bash 语法测试，LF 字节检查仍运行。Linux/Windows CI 均提供这些工具。

### 如何理解 146 passed 与 145 passed, 1 skipped

`a10c1f1` 原有 146 个测试。2026-10-22 的 Linux 测试记录为 146 passed；
2026-10-23 在独立 Linux 环境（Python 3.12.14、Node 24.19.0、g++ 14.2.0）
重跑同一基线，结果仍为 **146 passed，0 skipped**。这是指定环境的有效记录，
不是所有平台永远 146/146，也不是尚未运行的 GitHub Actions 结果。

另一次用户环境的 **145 passed, 1 skipped** 必须原样保留。只有总数不能唯一定位
是哪一个 skip；原基线有以上三种条件（进程组、g++、Node），需该次运行的
`pytest -rs` 节点和理由才能归因。不要把缺编译器、Windows 或缺 Node 之一猜成
已证实原因。新增 checkout 回归后测试总数会变化，更不能继续硬编码“146/146”。

## Windows / WSL 已有工作副本的换行

仓库 `.gitattributes` 强制 tracked `.sh` 为 LF。Git 中已有的 9 个 shell blob
本来就是 LF；风险来自 Windows checkout 时的 `core.autocrlf` 转换，而不是需要
重写所有脚本逻辑。回归会在临时仓库构造 `core.autocrlf=true` checkout 并检查字节。

**新增属性或运行 `git add --renormalize .` 不保证旧工作副本的 CRLF 字节立即重写。**
先保存工作、查看 `git status`；不要为修换行执行会覆盖未提交修改的 `git reset --hard` / `git restore`。在 WSL/Git Bash 的仓库根目录可一次性只替换 tracked
`.sh` 内的 CRLF，保留其余内容、文件模式和用户修改，不修改 Git index：

```bash
python3 - <<'PY'
from pathlib import Path
import os
import subprocess
for raw in subprocess.check_output(['git', 'ls-files', '-z', '--', '*.sh']).split(b'\0'):
    if not raw:
        continue
    path = Path(os.fsdecode(raw))
    if path.is_symlink():
        raise SystemExit(f'Refusing to rewrite symlink: {path}')
    original = path.read_bytes()
    normalized = original.replace(b'\r\n', b'\n')
    if normalized != original:
        path.write_bytes(normalized)
        print(f'Normalized: {path}')
PY
git diff --check
while IFS= read -r -d '' script; do bash -n "$script"; done < <(git ls-files -z '*.sh')
```

不要因此执行 `setup_cosmos_compiler.sh` 或任何板端脚本；`bash -n` 仅语法检查。

## 本次修复交付与后续硬件验收

本次修复范围是 LF checkout、CI、无权重流程、SDK mock 打包与 CPU 单元测试；
**按用户要求，本次不需要跑完整模型转换，也不以此阻止交付本次修复。**
浏览器/CI 结果仍应按实测报告，缺失项不得冒充通过。

PR 保持实验性和 Draft，不自动转 Ready。以下真实硬件验收留给用户本机后续开展；
它们不是本次云端修复必须执行的步骤：

1. 完整合法 Cosmos 权重 + 独立官方编译环境的真实导出、量化编译、combined BModel
   与必需 graphs 核对，不是 `--preflight-only` 或 mock。
2. 在匹配的 libsophon runtime/dev、驱动、CMake/C++17 环境构建完整原生 SDK 模块。
3. BM1684X 文本和单图冒烟，记录实际输出、停止/采样行为及系统/TPU 内存峰值；
   单图通过后才逐步做视频和精度对比。

本 CI 不自动安装 TPU 工具、不下载模型权重，不更改原有模型目录的 available /
verified 状态，也不授权合并。
