import json
import ast
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .backends.base import BackendRegistry
from .backends.rknn import RknnBackend
from .catalog import BY_ID
from .exporters import ArchitectureExporters


SDK_CLASSES = {
    "yolov5": "YoloV5", "yolov5_seg": "YoloV5Seg", "yolov6": "YoloV6", "yolov7": "YoloV7",
    "yolov8": "YoloV8", "yolov8_pose": "YoloV8Pose", "yolov8_obb": "YoloV8Obb",
    "yolov8_seg": "YoloV8Seg",
    "yolov10": "YoloV10",
    "yolox": "YoloX",
    "yolo11": "Yolo11", "mobilenet": "MobileNetV2", "resnet": "ResNet50",
    "retinaface": "RetinaFace",
    "ppyoloe": "PPYoloE",
    "lprnet": "LPRNet",
    "yamnet": "YAMNet",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Pipeline:
    def __init__(self, root):
        self.root = Path(root)
        self.model_zoo = Path(os.environ.get("RKNN_MODEL_ZOO", self.root / "vendor" / "rknn_model_zoo"))
        self.yolov10_repo = Path(os.environ.get("RKNN_YOLOV10", self.root / "vendor" / "yolov10"))
        self.exporters = ArchitectureExporters(self.root)
        self.backends = BackendRegistry()
        self.backends.register(RknnBackend())

    def run(self, task, update):
        work = Path(task["work_dir"])
        cfg = BY_ID[task["architecture"]]
        if not cfg["available"] and not task.get("acceptance_run"):
            raise ValueError(f"{cfg['name']} 尚未达到可交付标准：{cfg['blocker']}")
        source = Path(task["source_path"])
        update(8, "检查上传文件")
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError("上传的模型文件为空或不存在")
        if task["platform"] not in cfg["platforms"]:
            raise ValueError(f"{cfg['name']} 不支持目标平台 {task['platform']}")

        dataset = self._prepare_calibration(task.get("calibration_path"), work, update)
        onnx_path = self._to_onnx(source, work, task, update)
        self._check_onnx(onnx_path, update)

        backend = self.backends.get(task["backend"])
        output = work / f"model{backend.output_suffix}"
        backend.convert(onnx_path, output, {
            "platform": task["platform"], "precision": task["precision"],
            "dataset": str(dataset) if dataset else None, "catalog": cfg,
        }, update)
        update(91, "组装对应架构的部署 SDK")
        package = self._package_sdk(task, cfg, onnx_path, output, work)
        update(100, "转换完成，SDK 已打包")
        return {"model": str(output), "package": str(package)}

    def _prepare_calibration(self, archive, work, update):
        if not archive:
            return None
        update(15, "解压并检查量化校准集")
        dest = work / "calibration"
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                target = (dest / info.filename).resolve()
                if dest.resolve() not in target.parents and target != dest.resolve():
                    raise ValueError("校准 ZIP 包含不安全路径")
                if not info.is_dir():
                    zf.extract(info, dest)
        images = sorted(p for p in dest.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if not images:
            raise ValueError("校准 ZIP 中没有可用图片")
        dataset = work / "dataset.txt"
        dataset.write_text("\n".join(str(p.resolve()) for p in images), encoding="utf-8")
        return dataset

    def _to_onnx(self, source, work, task, update):
        if source.suffix.lower() == ".onnx" and task.get("internal_test_onnx"):
            output = work / "model.onnx"
            shutil.copy2(source, output)
            update(28, "使用内部测试 ONNX")
            return output
        if self.exporters.supports(task["architecture"], source.suffix, task["vendor"]):
            return self.exporters.export(source, work, task["architecture"], task["vendor"], update)
        raise ValueError(f"{task['architecture']} 未注册原始权重 {source.suffix} 的专用导出器")

    def _check_onnx(self, path, update):
        update(38, "校验 ONNX 模型结构")
        try:
            import onnx
            model = onnx.load(str(path))
            onnx.checker.check_model(model)
        except ImportError:
            return
        except Exception as exc:
            raise ValueError(f"ONNX 结构校验失败: {exc}") from exc

    def _package_sdk(self, task, cfg, onnx_path, output, work):
        stage = work / "sdk"
        stage.mkdir(exist_ok=True)
        shutil.copytree(self.root / "sdk_template", stage, dirs_exist_ok=True)
        for cache_dir in stage.rglob("__pycache__"):
            shutil.rmtree(cache_dir)
        for bytecode in stage.rglob("*.pyc"):
            bytecode.unlink()
        example = self.model_zoo / "examples" / cfg["example"]
        reference = stage / "reference"
        if example.is_dir():
            for name in ("python", "cpp", "model", "README.md"):
                src = example / name
                if src.is_dir():
                    shutil.copytree(src, reference / name, dirs_exist_ok=True)
                elif src.is_file():
                    reference.mkdir(exist_ok=True)
                    shutil.copy2(src, reference / name)
            py_utils = self.model_zoo / "py_utils"
            if py_utils.is_dir():
                shutil.copytree(py_utils, reference / "py_utils", dirs_exist_ok=True)
        model_dir = stage / "model"
        model_dir.mkdir(exist_ok=True)
        shutil.copy2(output, model_dir / output.name)
        shutil.copy2(onnx_path, model_dir / onnx_path.name)
        shutil.copy2(output, stage / "edgeforge_sdk" / "model.rknn")
        export_metadata = self._extract_export_metadata(work)
        labels = export_metadata.get("labels") or self._extract_labels(onnx_path, example)
        runtime_config = {
            "architecture": cfg["id"], "architecture_name": cfg["name"],
            "modality": cfg["modality"], "task_type": cfg["task_type"],
            "result_type": cfg["result_type"],
            "platform": task["platform"], "precision": task["precision"],
            "input_size": cfg["input_size"] or [640, 640], "labels": labels,
            "mean": cfg.get("mean", [0, 0, 0]), "std": cfg.get("std", [255, 255, 255]),
            "color_order": cfg.get("color_order", "rgb"),
        }
        if export_metadata.get("anchors"):
            runtime_config["anchors"] = export_metadata["anchors"]
        for key in ("sample_rate", "sample_count", "chunk_seconds"):
            if key in export_metadata:
                runtime_config[key] = export_metadata[key]
        config_text = json.dumps(runtime_config, ensure_ascii=False, indent=2)
        (stage / "edgeforge_sdk" / "config.json").write_text(config_text, encoding="utf-8")
        # This directory is the portable model bundle consumed by the one-time
        # installed universal SDK.  It may be copied independently of the ZIP.
        (model_dir / "config.json").write_text(config_text, encoding="utf-8")
        sdk_class = SDK_CLASSES.get(cfg["id"])
        if not sdk_class:
            raise ValueError(f"{cfg['name']} 尚未登记显式 SDK 类")
        demo_name = "test.wav" if cfg["modality"] == "audio" else "test.jpg"
        if cfg["modality"] == "audio":
            official_demo = example / "model" / demo_name
            if official_demo.is_file():
                shutil.copy2(official_demo, stage / demo_name)
        example_text = (
            f"from edgeforge_sdk import {sdk_class}\n\n"
            f"with {sdk_class}(\"model\") as model:\n"
            f"    results = model.predict(\"{demo_name}\")\n"
            "    for item in results:\n"
            "        print(item.as_dict())\n")
        if cfg["modality"] == "image":
            example_text += f"    model.draw(\"{demo_name}\", results, \"result.jpg\")\n"
        (stage / "example.py").write_text(example_text, encoding="utf-8")
        manifest = {
            "created_at": utc_now(), "architecture": cfg["id"], "architecture_name": cfg["name"],
            "vendor": task["vendor"], "backend": task["backend"], "platform": task["platform"], "precision": task["precision"],
            "modality": cfg["modality"], "task_type": cfg["task_type"], "result_type": cfg["result_type"],
            "input_size": cfg["input_size"], "official_example": cfg["example"],
            "model": f"model/{output.name}", "intermediate_model": f"model/{onnx_path.name}",
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (stage / "使用说明.txt").write_text(
            f"{cfg['name']} 部署 SDK\n目标: {task['platform']} / {task['precision']}\n\n"
            "edgeforge_sdk/：统一可调用 Python SDK（load/predict/infer_raw/draw）\n"
            f"显式入口：from edgeforge_sdk import {sdk_class}\n"
            f"example.py：最小调用示例（默认读取随包附带的 {demo_name}）\n"
            "install.sh：板端 RKNN-Toolkit-Lite2 与 SDK 一键安装脚本\n"
            f"{demo_name}：可直接运行示例的演示输入\nSDK_README.md：安装与接口文档\n"
            "reference/：用于核对的官方 Python/C++ 示例，不是主调用入口\n"
            "model/：RKNN 与内部 ONNX 产物\nmanifest.json：机器可读构建信息\n",
            encoding="utf-8")
        archive = work / f"{cfg['id']}_{task['platform']}_{task['precision']}_sdk.zip"
        shutil.make_archive(str(archive.with_suffix("")), "zip", stage)
        return archive

    @staticmethod
    def _extract_export_metadata(work):
        path = Path(work) / "export_metadata.json"
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _extract_labels(onnx_path, example):
        try:
            import onnx
            metadata = {item.key: item.value for item in onnx.load(str(onnx_path)).metadata_props}
            value = metadata.get("names")
            if value:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, dict):
                    return [str(parsed[key]) for key in sorted(parsed, key=lambda x: int(x))]
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed]
        except Exception:
            pass
        if example.is_dir():
            candidates = (list((example / "model").glob("*labels*.txt")) +
                          list((example / "model").glob("*label*.txt")) +
                          list((example / "model").glob("synset*.txt")))
            if candidates:
                return [line.strip() for line in candidates[0].read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
        return []
