import shutil
import subprocess
from pathlib import Path

from .base import ConversionBackend


class BModelBackend(ConversionBackend):
    name = "bmodel"
    output_suffix = ".bmodel"

    _QUANTIZE = {
        "fp": "F16",
        "f16": "F16",
        "fp16": "F16",
        "f32": "F32",
        "fp32": "F32",
        "bf16": "BF16",
        "i8": "INT8",
        "int8": "INT8",
    }

    def convert(self, source, output, options, progress):
        if Path(source).suffix.lower() != ".onnx":
            raise ValueError("SOPHGO BModel backend currently requires an ONNX intermediate model")
        self._require_tool("model_transform.py")
        self._require_tool("model_deploy.py")

        precision = str(options["precision"]).lower()
        quantize = self._QUANTIZE.get(precision)
        if not quantize:
            raise ValueError(f"unsupported SOPHGO precision: {options['precision']}")
        chip = str(options["platform"]).lower()
        if chip != "bm1684x":
            raise ValueError(f"SOPHGO backend currently supports BM1684X only, got: {options['platform']}")

        cfg = options["catalog"]
        component = options.get("component", "model")
        work = Path(output).parent
        model_name = self._model_name(cfg["id"], Path(source).stem, component)
        mlir = work / f"{model_name}.mlir"

        progress(48, "Initializing TPU-MLIR model transform")
        command = [
            "model_transform.py",
            "--model_name", model_name,
            "--model_def", str(Path(source).resolve()),
            "--input_shapes", self._input_shapes(cfg["id"], cfg.get("input_size", [640, 640]), component),
            "--mlir", str(mlir),
        ]
        if component != "text":
            command.extend([
                "--mean", self._mean(cfg),
                "--scale", self._scale(cfg),
                "--keep_aspect_ratio",
                "--pixel_format", cfg.get("color_order", "rgb").lower(),
            ])
        if cfg["id"] == "yolo_world" and component == "detector":
            command.extend(["--output_names", "output"])
        self._run(command, work)

        calibration_table = None
        if quantize == "INT8":
            self._require_tool("run_calibration.py")
            dataset = options.get("dataset")
            if not dataset:
                raise ValueError("SOPHGO INT8 conversion requires a calibration image ZIP")
            calibration_table = work / f"{model_name}_cali_table"
            progress(65, "Running TPU-MLIR INT8 calibration")
            self._run([
                "run_calibration.py", str(mlir),
                "--dataset", str(Path(dataset).resolve()),
                "--input_num", str(int(options.get("calibration_input_num", 128))),
                "-o", str(calibration_table),
            ], work)

        progress(78, f"Deploying BM1684X BModel ({quantize})")
        deploy = [
            "model_deploy.py",
            "--mlir", str(mlir),
            "--quantize", quantize,
            "--chip", chip,
            "--model", str(Path(output).resolve()),
        ]
        if calibration_table:
            deploy.extend(["--calibration_table", str(calibration_table)])
        self._run(deploy, work)

    @staticmethod
    def _require_tool(name):
        if not shutil.which(name):
            raise RuntimeError(
                f"missing TPU-MLIR command '{name}'. Activate the official TPU-MLIR environment "
                "before running SOPHGO conversion.")

    @staticmethod
    def _model_name(architecture, fallback, component="model"):
        if architecture == "yolo_world" and component == "text":
            return "clip_text"
        return {
            "yolov5": "yolov5",
            "yolov7": "yolov7",
            "yolov8": "yolov8",
            "yolov8_seg": "yolov8s-seg",
            "yolov8_pose": "yolov8s-pose",
            "yolov8_obb": "yolov8s-obb",
            "yolov10": "yolov10",
            "yolo11": "yolo11",
            "yolox": "yolox",
            "yolo_world": "yoloworld_v2",
        }.get(architecture, fallback)

    @staticmethod
    def _input_shapes(architecture, input_size, component="model"):
        if architecture == "yolo_world" and component == "text":
            return "[[1,20]]"
        if architecture == "yolo_world":
            return "[[1,3,640,640],[1,80,512]]"
        width, height = [int(value) for value in input_size]
        return f"[[1,3,{height},{width}]]"

    @staticmethod
    def _mean(cfg):
        return ",".join(str(float(value)) for value in cfg.get("mean", [0, 0, 0]))

    @staticmethod
    def _scale(cfg):
        std = cfg.get("std", [255, 255, 255])
        return ",".join(str(1.0 / float(value)) if float(value) else "1.0" for value in std)

    @staticmethod
    def _run(command, cwd):
        result = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout)[-4000:]
            raise RuntimeError(f"{command[0]} failed with code {result.returncode}:\n{tail}")
