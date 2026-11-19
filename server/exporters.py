"""Architecture-aware exporters from training weights to vendor-friendly ONNX."""
import os
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .catalog import BY_ID
from .input_sizes import resolve_input_size
from .sophon_models import sophon_model


class ArchitectureExporters:
    ROCKCHIP_REPOS = {
        "yolov5": "yolov5", "yolov5_seg": "yolov5",
        "yolov6": "yolov6",
        "yolov7": "yolov7",
        "yolox": "yolox",
        "ppyoloe": "ppyoloe",
        "ppseg": "ppseg",
        "lprnet": "lprnet",
        "deeplabv3": "deeplabv3",
        "yamnet": "yamnet",
        "mobilesam": "mobilesam",
        "ppocr": "ppocr",
        "clip": "clip",
        "yolo_world": "yolo_world",
        "retinaface": "retinaface",
        "yolov8": "yolov8", "yolov8_seg": "yolov8", "yolov8_pose": "yolov8", "yolov8_obb": "yolov8",
        "yolov10": "yolov10", "yolo11": "yolo11",
    }

    def __init__(self, root):
        self.root = Path(root)
        self.repos = self.root / "vendor" / "exporters"
        self.sophon_demo = Path(os.environ.get("SOPHON_DEMO", self.root / "vendor" / "sophon-demo"))

    def supports(self, architecture, suffix, vendor):
        if vendor == "sophon":
            cfg = sophon_model(architecture)
            return bool(cfg and suffix.lower() in cfg["weight_suffixes"])
        if vendor != "rockchip":
            return False
        if architecture in {"mobilenet", "resnet", "yolox", "retinaface", "lprnet"}:
            return suffix.lower() == ".pth"
        if architecture == "ppyoloe":
            return suffix.lower() == ".pdparams"
        if architecture == "ppseg":
            return suffix.lower() == ".pdparams"
        if architecture == "yamnet":
            return suffix.lower() == ".zip"
        if architecture == "deeplabv3":
            return suffix.lower() == ".pb"
        if architecture == "mobilesam":
            return suffix.lower() == ".pt"
        if architecture == "ppocr":
            return suffix.lower() == ".zip"
        if architecture == "clip":
            return suffix.lower() == ".bin"
        if architecture == "yolo_world":
            return suffix.lower() == ".pth"
        return suffix.lower() == ".pt" and architecture in self.ROCKCHIP_REPOS

    def export(self, source, work, architecture, vendor, progress, *, input_size=None):
        if not self.supports(architecture, source.suffix, vendor):
            raise ValueError(f"{architecture} 尚未注册 {vendor} 的 {source.suffix} 导出器；请上传模型目录声明的原始权重格式")
        input_size = resolve_input_size(BY_ID[architecture], vendor, input_size)
        if vendor == "sophon":
            if architecture == "ppocr":
                return self._export_ppocr(source, work, progress, det_input_size=640)
            if architecture == "lprnet":
                return self._export_sophon_lprnet(source, work, progress)
            return self._export_sophon_yolo(source, work, architecture, progress, input_size)
        if architecture in {"mobilenet", "resnet"}:
            output = work / "model.onnx"
            model_name = "mobilenet_v2" if architecture == "mobilenet" else "resnet50"
            progress(24, f"调用 {model_name} 专用状态字典导出器")
            self._run([
                sys.executable, str(self.root / "server" / "torchvision_export_worker.py"),
                "--architecture", model_name, "--model", str(source.resolve()),
                "--output", str(output), "--metadata", str(work / "export_metadata.json"),
            ], self.root)
            return output
        if architecture == "ppseg":
            repo = self.repos / "ppseg"
            output = work / "model.onnx"
            paddle_python = Path(os.environ.get(
                "EDGEFORGE_PPSEG_PYTHON", os.environ.get(
                    "EDGEFORGE_PPYOLOE_PYTHON",
                    str(Path(sys.prefix).parent / ".edgeforge-ppyoloe-export-venv" /
                        "bin" / "python"))))
            if not paddle_python.is_file():
                raise RuntimeError(
                    f"缺少 PaddleSeg 独立导出环境: {paddle_python}；请重新运行 scripts/setup_wsl.sh")
            self._run([
                str(paddle_python), str(self.root / "server" / "ppseg_export_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()),
                "--output", str(output), "--metadata", str(work / "export_metadata.json"),
            ], repo)
            return output
        if architecture == "retinaface":
            repo = self.repos / "retinaface"
            output = work / "model.onnx"
            self._run([
                sys.executable, str(self.root / "server" / "retinaface_export_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()),
                "--output", str(output), "--metadata", str(work / "export_metadata.json"),
                "--imgsz", "320",
            ], repo)
            return output
        if architecture == "lprnet":
            output = work / "model.onnx"
            self._run([
                sys.executable, str(self.root / "server" / "lprnet_export_worker.py"),
                "--zoo-script", str(self.root / "vendor" / "rknn_model_zoo" /
                                     "examples" / "LPRNet" / "python" / "export_onnx.py"),
                "--model", str(source.resolve()), "--output", str(output),
                "--metadata", str(work / "export_metadata.json"),
            ], self.root)
            return output
        if architecture == "yamnet":
            output = work / "model.onnx"
            tensorflow_python = Path(os.environ.get(
                "EDGEFORGE_YAMNET_PYTHON",
                str(Path(sys.prefix).parent / ".edgeforge-yamnet-export-venv" /
                    "bin" / "python")))
            if not tensorflow_python.is_file():
                raise RuntimeError(
                    f"缺少 YAMNet 独立导出环境: {tensorflow_python}；请重新运行 scripts/setup_wsl.sh")
            self._run([
                str(tensorflow_python), str(self.root / "server" / "yamnet_export_worker.py"),
                "--model", str(source.resolve()), "--output", str(output),
                "--metadata", str(work / "export_metadata.json"),
                "--labels", str(self.root / "vendor" / "rknn_model_zoo" / "examples" /
                                "yamnet" / "model" / "yamnet_class_map.txt"),
            ], self.root)
            return output
        if architecture == "deeplabv3":
            output = work / "model.onnx"
            tensorflow_python = Path(os.environ.get(
                "EDGEFORGE_DEEPLAB_PYTHON", os.environ.get(
                    "EDGEFORGE_YAMNET_PYTHON",
                    str(Path(sys.prefix).parent / ".edgeforge-yamnet-export-venv" /
                        "bin" / "python"))))
            if not tensorflow_python.is_file():
                raise RuntimeError(
                    f"缺少 DeepLab TensorFlow 导出环境: {tensorflow_python}；请重新运行 scripts/setup_wsl.sh")
            self._run([
                str(tensorflow_python), str(self.root / "server" / "deeplab_export_worker.py"),
                "--model", str(source.resolve()), "--output", str(output),
                "--metadata", str(work / "export_metadata.json"),
            ], self.root)
            return output
        if architecture == "ppyoloe":
            repo = self.repos / "ppyoloe"
            output = work / "model.onnx"
            paddle_python = Path(os.environ.get(
                "EDGEFORGE_PPYOLOE_PYTHON",
                str(Path(sys.prefix).parent / ".edgeforge-ppyoloe-export-venv" /
                    "bin" / "python")))
            if not paddle_python.is_file():
                raise RuntimeError(
                    f"缺少 PP-YOLOE 独立导出环境: {paddle_python}；请重新运行 scripts/setup_wsl.sh")
            self._run([
                str(paddle_python), str(self.root / "server" / "ppyoloe_export_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()),
                "--output", str(output), "--metadata", str(work / "export_metadata.json"),
                "--labels", str(self.root / "vendor" / "rknn_model_zoo" / "examples" /
                                "ppyoloe" / "model" / "coco_80_labels_list.txt"),
            ], repo)
            self._fix_static_batch(output)
            return output
        if architecture == "mobilesam":
            repo = self.repos / "mobilesam"
            encoder = work / "encoder.onnx"
            decoder = work / "decoder.onnx"
            progress(24, "调用 MobileSAM 官方模型定义导出 448px 编码器和双提示解码器")
            self._run([
                sys.executable, str(self.root / "server" / "mobilesam_export_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()),
                "--encoder", str(encoder), "--decoder", str(decoder),
                "--metadata", str(work / "export_metadata.json"),
            ], repo)
            return {"encoder": encoder, "decoder": decoder}
        if architecture == "ppocr":
            det = work / "det.onnx"
            cls = work / "cls.onnx"
            rec = work / "rec.onnx"
            paddle_python = Path(os.environ.get(
                "EDGEFORGE_PPOCR_PYTHON", os.environ.get(
                    "EDGEFORGE_PPYOLOE_PYTHON",
                    str(Path(sys.prefix).parent / ".edgeforge-ppyoloe-export-venv" /
                        "bin" / "python"))))
            if not paddle_python.is_file():
                raise RuntimeError(
                    f"缺少 PPOCR Paddle 导出环境: {paddle_python}；请重新运行 scripts/setup_wsl.sh")
            progress(24, "从 PP-OCRv4 官方 Paddle 推理包导出固定尺寸检测与识别 ONNX")
            self._run([
                str(paddle_python), str(self.root / "server" / "ppocr_export_worker.py"),
                "--model", str(source.resolve()), "--det", str(det), "--cls", str(cls), "--rec", str(rec),
                "--metadata", str(work / "export_metadata.json"),
                "--dictionary", str(self.root / "vendor" / "rknn_model_zoo" / "examples" /
                                    "PPOCR" / "PPOCR-Rec" / "model" / "ppocr_keys_v1.txt"),
            ], self.root)
            outputs = {"det": det, "rec": rec}
            if cls.is_file():
                outputs = {"det": det, "cls": cls, "rec": rec}
            return outputs
        if architecture == "clip":
            image_model = work / "image.onnx"
            text_model = work / "text.onnx"
            assets = self.repos / "clip"
            progress(24, "从 CLIP ViT-B/32 官方 PyTorch 权重导出图像与文本编码器")
            self._run([
                sys.executable, str(self.root / "server" / "clip_export_worker.py"),
                "--model", str(source.resolve()), "--assets", str(assets),
                "--image", str(image_model), "--text", str(text_model),
                "--tokenizer-output", str(work / "tokenizer"),
                "--metadata", str(work / "export_metadata.json"),
            ], self.root)
            return {"image": image_model, "text": text_model}
        if architecture == "yolo_world":
            detector = work / "detector.onnx"
            text_model = work / "text.onnx"
            repo = self.repos / "yolo_world"
            assets = self.repos / "clip"
            export_python = Path(os.environ.get(
                "EDGEFORGE_YOLO_WORLD_PYTHON",
                str(Path(sys.prefix).parent / ".edgeforge-yolo-world-export-venv" /
                    "bin" / "python")))
            if not export_python.is_file():
                raise RuntimeError(
                    f"缺少 YOLO-World 独立导出环境: {export_python}；请重新运行 scripts/setup_wsl.sh")
            progress(24, "从 YOLO-World V2-S 官方 checkpoint 导出检测器与 CLIP 文本编码器")
            self._run([
                str(export_python), str(self.root / "server" / "yolo_world_export_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()),
                "--assets", str(assets),
                "--labels", str(self.root / "vendor" / "rknn_model_zoo" / "examples" /
                                "yolo_world" / "model" / "detect_classes.txt"),
                "--detector", str(detector), "--text", str(text_model),
                "--tokenizer-output", str(work / "tokenizer"),
                "--metadata", str(work / "export_metadata.json"),
            ], repo)
            return {"detector": detector, "text": text_model}
        family = self.ROCKCHIP_REPOS[architecture]
        height_width = [str(value) for value in reversed(input_size)]
        metadata = work / "export_metadata.json"
        progress(24, f"调用 {family.upper()} 的 Rockchip 官方导出器")
        if family == "yolox":
            repo = self.repos / "yolox"
            self._run([
                sys.executable, str(self.root / "server" / "yolox_metadata_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()),
                "--output", str(metadata),
                "--official-labels", str(self.root / "vendor" / "rknn_model_zoo" /
                                          "examples" / "yolox" / "model" /
                                          "coco_80_labels_list.txt"),
                "--official-s-sha256", "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30",
            ], repo)
            variant = __import__("json").loads(metadata.read_text(encoding="utf-8"))["model_variant"]
            output = work / "model.onnx"
            self._run([
                sys.executable, "tools/export_onnx.py", "--output-name", str(output),
                "-n", variant, "-c", str(source.resolve()), "--rknpu", "--no-onnxsim",
            ], repo)
            return output
        if family == "yolov5":
            repo = self.repos / "yolov5"
            cmd = [sys.executable, "export.py", "--rknpu", "--weights", str(source.resolve()),
                   "--include", "onnx", "--opset", "12", "--simplify", "--imgsz", *height_width]
            self._run(cmd, repo)
            self._run([
                sys.executable, str(self.root / "server" / "yolov5_metadata_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()), "--output", str(metadata),
            ], repo)
        elif family == "yolov6":
            repo = self.repos / "yolov6"
            self._run([
                sys.executable, "deploy/RKNN/export_onnx_for_rknn.py",
                "--weights", str(source.resolve()), "--device", "cpu", "--img-size", *height_width,
            ], repo)
            self._run([
                sys.executable, str(self.root / "server" / "yolov6_metadata_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()),
                "--output", str(metadata),
                "--official-labels", str(self.root / "vendor" / "rknn_model_zoo" /
                                          "examples" / "yolov6" / "model" /
                                          "coco_80_labels_list.txt"),
                "--official-sha256", "3e2944e1c2d6a0dc4eed72e1493e3afe1c052373588b33abfecf0bfb4212b1a4",
            ], repo)
        elif family == "yolov7":
            repo = self.repos / "yolov7"
            self._run([
                sys.executable, "export.py", "--rknpu", "--weights",
                str(source.resolve()), "--device", "cpu", "--img-size", *height_width,
            ], repo)
            self._run([
                sys.executable, str(self.root / "server" / "yolov7_metadata_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()),
                "--output", str(metadata),
            ], repo)
        elif family in {"yolov8", "yolo11"}:
            repo = self.repos / family
            cmd = [sys.executable, str(self.root / "server" / "export_worker.py"),
                   "--repo", str(repo), "--model", str(source.resolve()),
                   "--format", "rknn", "--opset", "12", "--imgsz", *height_width,
                   "--metadata", str(metadata)]
            self._run(cmd, repo)
        else:
            repo = self.root / "vendor" / "yolov10"
            cmd = [sys.executable, str(self.root / "server" / "export_worker.py"),
                   "--repo", str(repo), "--model", str(source.resolve()), "--format", "rknn", "--opset", "13",
                   "--imgsz", *height_width]
            cmd.extend(["--metadata", str(metadata)])
            self._run(cmd, repo)
        candidates = sorted(source.parent.glob(f"{source.stem}*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            candidates = sorted(repo.rglob(f"{source.stem}*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError(f"{family} 导出命令结束，但未找到 ONNX 产物")
        output = work / "model.onnx"
        shutil.copy2(candidates[0], output)
        return output

    def _export_ppocr(self, source, work, progress, det_input_size=480):
        det = work / "det.onnx"
        cls = work / "cls.onnx"
        rec = work / "rec.onnx"
        paddle_python = Path(os.environ.get(
            "EDGEFORGE_PPOCR_PYTHON", os.environ.get(
                "EDGEFORGE_PPYOLOE_PYTHON",
                str(Path(sys.prefix).parent / ".edgeforge-ppyoloe-export-venv" /
                    "bin" / "python"))))
        if not paddle_python.is_file():
            raise RuntimeError(
                f"缺少 PPOCR Paddle 导出环境: {paddle_python}；请重新运行 scripts/setup_wsl.sh")
        progress(24, "从 PP-OCRv4 官方 Paddle 推理包导出固定尺寸检测与识别 ONNX")
        self._run([
            str(paddle_python), str(self.root / "server" / "ppocr_export_worker.py"),
            "--model", str(source.resolve()), "--det", str(det), "--cls", str(cls), "--rec", str(rec),
            "--metadata", str(work / "export_metadata.json"),
            "--dictionary", str(self.root / "vendor" / "rknn_model_zoo" / "examples" /
                                "PPOCR" / "PPOCR-Rec" / "model" / "ppocr_keys_v1.txt"),
            "--det-input-size", str(int(det_input_size)),
        ], self.root)
        outputs = {"det": det, "rec": rec}
        if cls.is_file():
            outputs = {"det": det, "cls": cls, "rec": rec}
        return outputs

    def _export_sophon_yolo(self, source, work, architecture, progress, input_size):
        metadata = work / "export_metadata.json"
        output = work / "model.onnx"
        sophon_cfg = sophon_model(architecture) or {}
        height_width = [str(value) for value in reversed(input_size)]
        progress(24, f"Exporting {architecture.upper()} to SOPHGO-friendly ONNX")
        if architecture == "yolo_world":
            detector = work / "detector.onnx"
            text_model = work / "text.onnx"
            repo = self.repos / "yolo_world"
            assets = self.repos / "clip"
            export_python = Path(os.environ.get(
                "EDGEFORGE_YOLO_WORLD_PYTHON",
                str(Path(sys.prefix).parent / ".edgeforge-yolo-world-export-venv" /
                    "bin" / "python")))
            if not export_python.is_file():
                raise RuntimeError(
                    f"缂哄皯 YOLO-World 鐙珛瀵煎嚭鐜: {export_python}锛涜閲嶆柊杩愯 scripts/setup_wsl.sh")
            self._run([
                str(export_python), str(self.root / "server" / "yolo_world_export_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()),
                "--assets", str(assets),
                "--labels", str(self.root / "vendor" / "rknn_model_zoo" / "examples" /
                                "yolo_world" / "model" / "detect_classes.txt"),
                "--detector", str(detector), "--text", str(text_model),
                "--tokenizer-output", str(work / "tokenizer"),
                "--metadata", str(metadata),
                "--sequence-length", str(int(sophon_cfg.get("sequence_length", 77))),
            ], repo)
            clip_python = Path(os.environ.get(
                "EDGEFORGE_YOLO_WORLD_CLIP_PYTHON",
                str(Path(sys.prefix).parent / ".edgeforge-yolo-world-clip-export-venv" /
                    "bin" / "python")))
            if not clip_python.is_file():
                raise RuntimeError(
                    f"missing YOLO-World Torch 1.13 CLIP export environment: {clip_python}; "
                    "run scripts/setup_wsl.sh")
            self._run([
                str(clip_python), str(self.root / "server" / "yolo_world_clip_text_export_worker.py"),
                "--output", str(text_model),
                "--projection", str(work / "text_projection_512_512.npy"),
                "--bpe-output", str(work / "bpe_simple_vocab_16e6.txt.gz"),
            ], self.root)
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            meta.update({
                "sequence_length": int(sophon_cfg.get("sequence_length", 77)),
                "token_pad_id": 0,
                "text_projection": "text_projection_512_512.npy",
            })
            metadata.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"detector": detector, "text": text_model}
        if architecture == "yolox":
            repo = self.repos / "yolox"
            self._run([
                sys.executable, str(self.root / "server" / "yolox_metadata_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()),
                "--output", str(metadata),
                "--official-labels", str(self.root / "vendor" / "rknn_model_zoo" /
                                          "examples" / "yolox" / "model" /
                                          "coco_80_labels_list.txt"),
                "--official-s-sha256", "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30",
            ], repo)
            variant = __import__("json").loads(metadata.read_text(encoding="utf-8"))["model_variant"]
            self._run([
                sys.executable, "tools/export_onnx.py", "--output-name", str(output),
                "-n", variant, "-c", str(source.resolve()), "--no-onnxsim",
            ], repo)
            return output
        if architecture in {
            "yolov8", "yolov8_seg", "yolov8_pose", "yolov8_obb", "yolov8_cls",
            "yolov9", "yolov9_seg", "yolo11", "yolo11_obb", "yolov12",
        }:
            repo = self.repos / ("yolo11" if architecture in {"yolo11", "yolo11_obb", "yolov12"} else "yolov8")
            self._run([
                sys.executable, str(self.root / "server" / "export_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()),
                "--format", "onnx", "--opset", "17", "--imgsz", *height_width,
                "--metadata", str(metadata),
            ], repo)
        elif architecture == "yolov10":
            repo = self.root / "vendor" / "yolov10"
            self._run([
                sys.executable, str(self.root / "server" / "export_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()),
                "--format", "onnx", "--opset", "17", "--imgsz", *height_width,
                "--metadata", str(metadata),
            ], repo)
        elif architecture == "yolov5":
            repo = self.repos / "yolov5"
            self._run([
                sys.executable, "export.py", "--weights", str(source.resolve()),
                "--include", "onnx", "--opset", "12", "--simplify", "--imgsz", *height_width,
            ], repo)
            self._run([
                sys.executable, str(self.root / "server" / "yolov5_metadata_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()), "--output", str(metadata),
            ], repo)
        elif architecture == "yolov7":
            repo = self.repos / "yolov7"
            self._run([
                sys.executable, "export.py", "--weights", str(source.resolve()),
                "--device", "cpu", "--img-size", *height_width,
            ], repo)
            self._run([
                sys.executable, str(self.root / "server" / "yolov7_metadata_worker.py"),
                "--repo", str(repo), "--model", str(source.resolve()),
                "--output", str(metadata),
            ], repo)
        else:
            raise ValueError(f"{architecture} is not registered for SOPHGO export")
        candidates = sorted(source.parent.glob(f"{source.stem}*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            candidates = sorted(repo.rglob(f"{source.stem}*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError(f"{architecture} export finished but no ONNX artifact was found")
        shutil.copy2(candidates[0], output)
        return output

    def _export_sophon_lprnet(self, source, work, progress):
        output = work / "model.onnx"
        metadata = work / "export_metadata.json"
        network = self.sophon_demo / "sample" / "LPRNet" / "tools" / "LPRNet.py"
        if not network.is_file():
            raise RuntimeError(f"缺少 sophon-demo LPRNet 网络定义: {network}")
        progress(24, "调用 sophon-demo LPRNet 官方 PyTorch 导出器")
        self._run([
            sys.executable, str(self.root / "server" / "sophon_lprnet_export_worker.py"),
            "--network", str(network), "--model", str(source.resolve()),
            "--output", str(output), "--metadata", str(metadata),
        ], self.root)
        return output

    @staticmethod
    def _fix_static_batch(model_path):
        import onnx

        model = onnx.load(str(model_path))
        for value in list(model.graph.input) + list(model.graph.output):
            dimensions = value.type.tensor_type.shape.dim
            if dimensions:
                dimensions[0].ClearField("dim_param")
                dimensions[0].dim_value = 1
        onnx.save(model, str(model_path))

    @staticmethod
    def _run(cmd, cwd):
        if not cwd.is_dir():
            raise RuntimeError(f"缺少架构官方导出仓库: {cwd}")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(Path(cwd).resolve()) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError("模型架构导出失败:\n" + proc.stdout[-8000:])
