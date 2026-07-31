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
from .backends.bmodel import BModelBackend
from .backends.rknn import RknnBackend
from .catalog import BY_ID
from .exporters import ArchitectureExporters
from .sophon_models import sophon_model


SDK_CLASSES = {
    "yolov5": "YoloV5", "yolov5_seg": "YoloV5Seg", "yolov6": "YoloV6", "yolov7": "YoloV7",
    "yolov8": "YoloV8", "yolov8_cls": "YoloV8Cls", "yolov8_pose": "YoloV8Pose",
    "yolov8_obb": "YoloV8Obb", "yolov8_seg": "YoloV8Seg",
    "yolov9": "YoloV9", "yolov9_seg": "YoloV9Seg",
    "yolov10": "YoloV10",
    "yolox": "YoloX",
    "yolo11": "Yolo11", "yolo11_obb": "Yolo11Obb", "yolov12": "YoloV12",
    "mobilenet": "MobileNetV2", "resnet": "ResNet50",
    "retinaface": "RetinaFace",
    "ppyoloe": "PPYoloE",
    "lprnet": "LPRNet",
    "yamnet": "YAMNet",
    "ppseg": "PPLiteSeg",
    "deeplabv3": "DeepLabV3Plus",
    "mobilesam": "MobileSAM",
    "ppocr": "PPOCR",
    "clip": "CLIP",
    "yolo_world": "YoloWorld",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Pipeline:
    def __init__(self, root):
        self.root = Path(root)
        self.model_zoo = Path(os.environ.get("RKNN_MODEL_ZOO", self.root / "vendor" / "rknn_model_zoo"))
        self.sophon_demo = Path(os.environ.get("SOPHON_DEMO", self.root / "vendor" / "sophon-demo"))
        self.yolov10_repo = Path(os.environ.get("RKNN_YOLOV10", self.root / "vendor" / "yolov10"))
        self.exporters = ArchitectureExporters(self.root)
        self.backends = BackendRegistry()
        self.backends.register(RknnBackend())
        self.backends.register(BModelBackend())

    def run(self, task, update):
        work = Path(task["work_dir"])
        cfg = BY_ID[task["architecture"]]
        if not cfg["available"] and not task.get("acceptance_run"):
            raise ValueError(f"{cfg['name']} 尚未达到可交付标准：{cfg['blocker']}")
        source = Path(task["source_path"])
        update(8, "检查上传文件")
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError("上传的模型文件为空或不存在")
        support = cfg.get("vendor_support", {}).get(task["vendor"])
        if not support or not support.get("available"):
            raise ValueError(f"{cfg['name']} does not support vendor {task['vendor']}")
        if task["platform"] not in support["platforms"]:
            raise ValueError(f"{cfg['name']} 不支持目标平台 {task['platform']}")
        effective_cfg = dict(cfg)
        if support.get("input_size"):
            effective_cfg["input_size"] = support["input_size"]
        sophon_cfg = sophon_model(cfg["id"]) if task["vendor"] == "sophon" else None
        if sophon_cfg and sophon_cfg.get("sequence_length"):
            effective_cfg["sequence_length"] = sophon_cfg["sequence_length"]

        dataset = self._prepare_calibration(task.get("calibration_path"), work, update)
        onnx_artifacts = self._to_onnx(source, work, task, update)
        onnx_models = onnx_artifacts if isinstance(onnx_artifacts, dict) else {"model": onnx_artifacts}
        for path in onnx_models.values():
            self._check_onnx(path, update)

        backend = self.backends.get(task["backend"])
        outputs = {}
        for component, onnx_path in onnx_models.items():
            output = work / f"{component}{backend.output_suffix}"
            options = {
                "platform": task["platform"], "precision": task["precision"],
                "dataset": str(dataset) if dataset else None, "catalog": effective_cfg,
                "component": component,
            }
            if dataset:
                options["calibration_dir"] = str(work / "calibration")
            if cfg["id"] == "mobilesam" and component == "decoder":
                options["fuse_normalization"] = False
            if cfg["id"] == "ppocr" and component in {"cls", "rec"}:
                options["fuse_normalization"] = False
            if cfg["id"] == "clip" and component == "text":
                options["fuse_normalization"] = False
            if cfg["id"] == "yolo_world":
                options["fuse_normalization"] = False
                if (task["vendor"] == "sophon" and component == "detector" and
                        str(task["precision"]).lower() in {"i8", "int8"}):
                    options["quantize_table"] = str(self._yolo_world_quantize_table())
            backend.convert(onnx_path, output, options, update)
            outputs[component] = output
        update(91, "组装对应架构的部署 SDK")
        package = self._package_sdk(task, effective_cfg, onnx_models, outputs, work)
        if len(outputs) == 1:
            model_artifact = next(iter(outputs.values()))
        else:
            model_artifact = work / f"{cfg['id']}_models.zip"
            with zipfile.ZipFile(model_artifact, "w", zipfile.ZIP_DEFLATED) as archive:
                for component, path in outputs.items():
                    archive.write(path, f"{component}{backend.output_suffix}")
        update(100, "转换完成，SDK 已打包")
        return {"model": str(model_artifact), "package": str(package)}

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

    def _yolo_world_quantize_table(self):
        candidates = [
            self.sophon_demo / "sample" / "YOLO_world_v2" / "scripts" / "yoloworld_v2_qtable",
            self.root / "vendor" / "sophon-demo" / "sample" / "YOLO_world_v2" / "scripts" / "yoloworld_v2_qtable",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise ValueError(
            "YOLO-World V2 INT8 requires the official sophon-demo "
            "sample/YOLO_world_v2/scripts/yoloworld_v2_qtable. "
            "Run scripts/setup_sophon.sh or set SOPHON_DEMO to a sophon-demo checkout.")

    def _package_sdk(self, task, cfg, onnx_models, outputs, work):
        stage = work / "sdk"
        if stage.is_dir():
            shutil.rmtree(stage)
        stage.mkdir(exist_ok=True)
        shutil.copytree(self.root / "sdk_template", stage, dirs_exist_ok=True)
        for cache_dir in stage.rglob("__pycache__"):
            shutil.rmtree(cache_dir)
        for bytecode in stage.rglob("*.pyc"):
            bytecode.unlink()
        example = self._reference_example(task, cfg)
        reference = stage / "reference"
        if cfg["id"] == "ppocr" and example.is_dir():
            shutil.copytree(example, reference, dirs_exist_ok=True, ignore_dangling_symlinks=True)
        elif example.is_dir():
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
        for component, output in outputs.items():
            shutil.copy2(output, model_dir / output.name)
            embedded_name = f"model{output.suffix}" if len(outputs) == 1 else output.name
            shutil.copy2(output, stage / "edgeforge_sdk" / embedded_name)
        for onnx_path in onnx_models.values():
            shutil.copy2(onnx_path, model_dir / onnx_path.name)
        tokenizer_dir = work / "tokenizer"
        if tokenizer_dir.is_dir():
            shutil.copytree(tokenizer_dir, model_dir / "tokenizer", dirs_exist_ok=True)
            shutil.copytree(tokenizer_dir, stage / "edgeforge_sdk" / "tokenizer", dirs_exist_ok=True)
        text_projection = work / "text_projection_512_512.npy"
        if text_projection.is_file():
            shutil.copy2(text_projection, model_dir / text_projection.name)
            shutil.copy2(text_projection, stage / "edgeforge_sdk" / text_projection.name)
        export_metadata = self._extract_export_metadata(work)
        primary_onnx = next(iter(onnx_models.values()))
        labels = ([] if cfg["id"] == "mobilesam" else
                  export_metadata.get("labels") or self._extract_labels(primary_onnx, example))
        runtime_config = {
            "architecture": cfg["id"], "architecture_name": cfg["name"],
            "modality": cfg["modality"], "task_type": cfg["task_type"],
            "result_type": cfg["result_type"],
            "platform": task["platform"], "precision": task["precision"],
            "vendor": task["vendor"], "backend": task["backend"],
            "input_size": cfg["input_size"] or [640, 640], "labels": labels,
            "mean": cfg.get("mean", [0, 0, 0]), "std": cfg.get("std", [255, 255, 255]),
            "color_order": cfg.get("color_order", "rgb"),
        }
        if export_metadata.get("anchors"):
            runtime_config["anchors"] = export_metadata["anchors"]
        if export_metadata.get("palette"):
            runtime_config["palette"] = export_metadata["palette"]
        for key in (
            "sample_rate", "sample_count", "chunk_seconds", "logit_resize",
            "input_layout", "prompt_count", "cls_input_size", "rec_input_size",
            "characters", "sequence_length", "pad_token_id", "logit_scale",
            "token_pad_id", "text_projection", "tokenizer_directory", "num_classes", "source_archives",
            "model_variant",
        ):
            if key in export_metadata:
                runtime_config[key] = export_metadata[key]
        if cfg.get("sequence_length"):
            runtime_config["sequence_length"] = cfg["sequence_length"]
        if len(outputs) > 1:
            runtime_config["model_files"] = {name: path.name for name, path in outputs.items()}
            runtime_config["intermediate_files"] = {name: path.name for name, path in onnx_models.items()}
        config_text = json.dumps(runtime_config, ensure_ascii=False, indent=2)
        (stage / "edgeforge_sdk" / "config.json").write_text(config_text, encoding="utf-8")
        # This directory is the portable model bundle consumed by the one-time
        # installed universal SDK.  It may be copied independently of the ZIP.
        (model_dir / "config.json").write_text(config_text, encoding="utf-8")
        sdk_class = SDK_CLASSES.get(cfg["id"])
        if not sdk_class:
            raise ValueError(f"{cfg['name']} 尚未登记显式 SDK 类")
        if cfg["id"] == "clip":
            demo_name = "dog_224x224.jpg"
        elif cfg["id"] == "yolo_world":
            demo_name = "bus.jpg"
        elif cfg["id"] == "mobilesam":
            demo_name = "picture.jpg"
        elif cfg["id"] == "ppseg":
            demo_name = "test.png"
        elif cfg["id"] == "deeplabv3":
            demo_name = "test_image.jpg"
        else:
            demo_name = "test.wav" if cfg["modality"] == "audio" else "test.jpg"
        if cfg["modality"] == "audio" or cfg["id"] in {"ppseg", "deeplabv3", "mobilesam", "ppocr", "clip", "yolo_world"}:
            official_demo = example / "model" / demo_name
            if official_demo.is_file():
                shutil.copy2(official_demo, stage / demo_name)
            elif cfg["id"] == "yolo_world":
                fallback_demo = self.model_zoo / "examples" / "yolo_world" / "model" / demo_name
                if fallback_demo.is_file():
                    shutil.copy2(fallback_demo, stage / demo_name)
        if cfg["id"] == "clip":
            example_text = (
                f"from edgeforge_sdk import {sdk_class}\n\n"
                f"with {sdk_class}(\"model\") as model:\n"
                f"    results = model.predict(\"{demo_name}\", texts=[\"a photo of a dog\", \"a photo of a cat\"])\n"
                "    print([item.as_dict() for item in results])\n"
                f"    model.draw(\"{demo_name}\", results, \"result.jpg\")\n")
        elif cfg["id"] == "yolo_world":
            example_text = (
                f"from edgeforge_sdk import {sdk_class}\n\n"
                f"with {sdk_class}(\"model\") as model:\n"
                f"    results = model.predict(\"{demo_name}\", texts=[\"person\", \"bus\"])\n"
                "    print([item.as_dict() for item in results])\n"
                f"    model.draw(\"{demo_name}\", results, \"result.jpg\")\n")
        elif cfg["id"] == "mobilesam":
            example_text = (
                f"from edgeforge_sdk import {sdk_class}\n\n"
                f"with {sdk_class}(\"model\") as model:\n"
                f"    result = model.predict(\"{demo_name}\", point_coords=[(190, 70), (460, 280)], point_labels=[2, 3])\n"
                "    print(result.as_dict())\n"
                f"    model.draw(\"{demo_name}\", result, \"result.jpg\")\n")
        else:
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
            "model": ({name: f"model/{path.name}" for name, path in outputs.items()}
                      if len(outputs) > 1 else f"model/{next(iter(outputs.values())).name}"),
            "intermediate_model": ({name: f"model/{path.name}" for name, path in onnx_models.items()}
                                   if len(onnx_models) > 1 else f"model/{primary_onnx.name}"),
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        backend_format = "BModel" if task["backend"] == "bmodel" else "RKNN"
        runtime_note = (
            "install.sh：安装 EdgeForge SDK；BModel 推理还需要板端已安装官方 SOPHON-SAIL\n"
            if task["backend"] == "bmodel" else
            "install.sh：板端 RKNN-Toolkit-Lite2 与 SDK 一键安装脚本\n"
        )
        (stage / "使用说明.txt").write_text(
            f"{cfg['name']} 部署 SDK\n目标: {task['platform']} / {task['precision']}\n\n"
            "edgeforge_sdk/：统一可调用 Python SDK（load/predict/infer_raw/draw）\n"
            f"显式入口：from edgeforge_sdk import {sdk_class}\n"
            f"example.py：最小调用示例（默认读取随包附带的 {demo_name}）\n"
            f"{runtime_note}"
            f"{demo_name}：可直接运行示例的演示输入\nSDK_README.md：安装与接口文档\n"
            "reference/：用于核对的官方 Python/C++ 示例，不是主调用入口\n"
            f"model/：{backend_format} 与内部 ONNX 产物\nmanifest.json：机器可读构建信息\n",
            encoding="utf-8")
        archive = work / f"{cfg['id']}_{task['platform']}_{task['precision']}_sdk.zip"
        shutil.make_archive(str(archive.with_suffix("")), "zip", stage)
        return archive

    def _reference_example(self, task, cfg):
        if task["vendor"] == "sophon":
            model = sophon_model(cfg["id"]) or {}
            return self.sophon_demo / "sample" / model.get("sample", cfg["example"])
        return self.model_zoo / "examples" / cfg["example"]

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
