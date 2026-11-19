import shutil
import subprocess
from pathlib import Path

from .base import ConversionBackend
from ..sophon_models import sophon_model


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
        if chip not in {"bm1684", "bm1684x"}:
            raise ValueError(
                f"SOPHGO backend currently supports BM1684/BM1684X only, got: {options['platform']}")
        if chip == "bm1684" and quantize == "F16":
            raise ValueError("BM1684 不支持 FP16；请选择 F32 或 INT8。")

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
        self._run(command, work)

        calibration_table = None
        if quantize == "INT8":
            self._require_tool("run_calibration.py")
            if options.get("calibration_list"):
                calibration_source = Path(options["calibration_list"]).resolve()
                if not calibration_source.is_file():
                    raise ValueError("SOPHGO INT8 calibration image list is missing")
                calibration_args = ["--data_list", str(calibration_source)]
            else:
                calibration_source = options.get("calibration_dir") or options.get("dataset")
                if not calibration_source or not Path(calibration_source).is_dir():
                    raise ValueError("SOPHGO INT8 calibration requires the extracted image directory")
                calibration_args = ["--dataset", str(Path(calibration_source).resolve())]
            calibration_table = work / f"{model_name}_cali_table"
            progress(65, "Running TPU-MLIR INT8 calibration")
            self._run([
                "run_calibration.py", str(mlir),
                *calibration_args,
                "--input_num", str(int(options.get("calibration_input_num", 128))),
                "-o", str(calibration_table),
            ], work)

        progress(78, f"Deploying {chip.upper()} BModel ({quantize})")
        deploy = [
            "model_deploy.py",
            "--mlir", str(mlir),
            "--quantize", quantize,
            "--chip", chip,
            "--model", str(Path(output).resolve()),
        ]
        if calibration_table:
            deploy.extend(["--calibration_table", str(calibration_table)])
        quantize_table = options.get("quantize_table")
        if quantize_table:
            quantize_table = Path(quantize_table).resolve()
            if not quantize_table.is_file():
                raise ValueError(f"SOPHGO quantize_table not found: {quantize_table}")
            deploy.extend(["--quantize_table", str(quantize_table)])
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
            return "clip_text_vitb32"
        cfg = sophon_model(architecture)
        component_names = cfg.get("component_names", {}) if cfg else {}
        if component in component_names:
            return component_names[component]
        return cfg.get("bmodel_name", fallback) if cfg else fallback

    @staticmethod
    def _input_shapes(architecture, input_size, component="model"):
        if architecture == "yolo_world" and component == "text":
            cfg = sophon_model(architecture) or {}
            return f"[[1,{int(cfg.get('sequence_length', 77))}]]"
        if architecture == "yolo_world":
            return "[[1,3,640,640],[1,80,512]]"
        if architecture == "ppocr" and component == "rec":
            return "[[1,3,48,320]]"
        if architecture == "ppocr" and component == "cls":
            return "[[1,3,48,192]]"
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
