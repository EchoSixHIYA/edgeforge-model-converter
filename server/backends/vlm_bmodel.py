"""Official multi-graph VLM compilation; deliberately separate from CV ONNX."""
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from ..vlm import ASSET_NAMES, ConversionCanceled


def log_tail(path, limit=8192):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


class VlmBModelBackend:
    def __init__(self, root):
        self.root = Path(root)

    def convert(self, source, work, params, toolchain, manifest, update, source_identities=None):
        work = Path(work).resolve()
        cancel = work / "cancel.requested"
        if cancel.exists():
            raise ConversionCanceled("转换已取消")
        source = Path(source).resolve(strict=True)
        if source_identities is None:
            raise ValueError("Missing source identity snapshot; rerun preflight")
        def check_source():
            for name, expected in source_identities.items():
                path = source / name
                if path.is_symlink() or not path.is_file() or path.resolve().parent != source:
                    raise ValueError(f"Model source changed after preflight: {name}")
                actual = [int(getattr(path.stat(), attr)) for attr in ("st_dev", "st_ino", "st_size", "st_mtime_ns")]
                if actual != expected:
                    raise ValueError(f"Model source changed after preflight: {name}")
        check_source()
        stage = work / "hf_source"
        stage.mkdir()
        # Stage ONLY the exact preflight snapshot. Newly appeared optional
        # processor assets, shard indexes and weights were not validated and
        # must never silently enter the compiler/SDK input directory.
        for name in sorted(source_identities):
            if name.endswith(".safetensors"):
                (stage / name).symlink_to((source / name).resolve())
            elif name in ASSET_NAMES or name == "model.safetensors.index.json":
                shutil.copy2(source / name, stage / name)
            else:
                raise ValueError(f"Unexpected file in preflight snapshot: {name}")
        check_source()
        output_dir = work / "compiled"
        output_dir.mkdir()
        command = [toolchain["python"], str(self.root / "server" / "vlm_worker.py"),
                   "--model-path", str(stage), "--out-dir", str(output_dir),
                   "--seq-length", str(params["seq_length"]), "--max-pixels", str(params["max_pixels"]),
                   "--precision", params["precision"], "--jobs", str(params["jobs"])]
        (work / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        environment = dict(os.environ)
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        environment["PATH"] = str(Path(toolchain["python"]).parent) + os.pathsep + environment.get("PATH", "")
        update(20, "调用官方 TPU-MLIR 1.28.1 Qwen3-VL 导出/编译（jobs=1）")
        process = None
        with (work / "conversion.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.Popen(command, cwd=work, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                (work / "compiler_process.json").write_text(json.dumps({
                    "pid": process.pid, "process_group": process.pid,
                    "note": "diagnostics only; restart does not automatically kill a possibly reused PID"}), encoding="utf-8")
                last = None
                while process.poll() is None:
                    if cancel.exists():
                        self._terminate(process)
                        raise ConversionCanceled("转换已取消；已终止编译进程组，保留日志，不发布部分产物")
                    tail = log_tail(work / "conversion.log", 2000)
                    stage_name = None
                    try:
                        stage_name = json.loads((work / "compiler_stage.json").read_text())["stage"]
                    except (OSError, ValueError, KeyError):
                        pass
                    if stage_name == "compile" or "EDGEFORGE_STAGE:compile" in tail:
                        message = (60, "官方多图编译中（GNU parallel -j 1）；详见日志")
                    elif stage_name == "export" or "EDGEFORGE_STAGE:export" in tail:
                        message = (35, "官方逐层 MLIR/视觉塔导出中（串行）；详见日志")
                    else:
                        message = None
                    if message and message != last:
                        update(*message)
                        last = message
                    time.sleep(0.5)
                if process.returncode:
                    raise RuntimeError(f"官方 VLM 导出/编译失败（exit={process.returncode}）：\n" + log_tail(work / "conversion.log", 4000))
            finally:
                if process is not None and process.poll() is None:
                    self._terminate(process)
        if cancel.exists():
            raise ConversionCanceled("转换已取消")
        check_source()
        candidates = list(output_dir.glob("*.bmodel"))
        if len(candidates) != 1 or not candidates[0].is_file() or candidates[0].stat().st_size < 64:
            raise RuntimeError("官方转换器未生成唯一且非空的最终 combined .bmodel；任务失败")
        info = output_dir / "model.log"
        if not info.is_file():
            raise RuntimeError("缺少官方 model_tool --info 验证输出 model.log")
        content = info.read_text(encoding="utf-8", errors="replace")
        missing = [name for name in manifest["graph_contract"]["required_graphs"]
                   if not re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", content)]
        if missing:
            raise RuntimeError("最终 BModel 缺少必需 graph（model_tool --info）：" + ", ".join(missing))
        output = work / "model.bmodel"
        shutil.move(str(candidates[0]), output)
        shutil.copy2(info, work / "model_info.txt")
        manifest["graph_contract"]["model_tool_names_checked"] = True
        manifest["graph_contract"]["actual_shapes_checked"] = False
        manifest["artifact_bytes"] = output.stat().st_size
        update(88, "已生成 combined BModel 并核验 graph 名称；尚未上板验证")
        return output

    @staticmethod
    def _terminate(process):
        # The leader may exit before children that ignore SIGTERM. Always send
        # SIGKILL to the entire group after the grace period, even after wait().
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            raise RuntimeError("无法确认编译进程组已终止；请检查 conversion.log 和系统进程")
