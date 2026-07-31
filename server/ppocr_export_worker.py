"""Export the two official PP-OCRv4 Paddle inference archives to static ONNX."""
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import onnx

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--det", required=True)
parser.add_argument("--rec", required=True)
parser.add_argument("--metadata", required=True)
parser.add_argument("--dictionary", required=True)
parser.add_argument("--det-input-size", type=int, default=480)
args = parser.parse_args()


def safe_extract_tar(archive, destination):
    destination = destination.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"unsafe path in Paddle archive: {member.name}")
        handle.extractall(destination)


workspace = Path(tempfile.mkdtemp(prefix="edgeforge-ppocr-export-"))
try:
    with zipfile.ZipFile(args.model) as package:
        tar_members = [item for item in package.infolist() if not item.is_dir() and item.filename.endswith(".tar")]
        if len(tar_members) != 2:
            raise ValueError("PPOCR source ZIP must contain det and rec Paddle inference .tar archives")
        archives = {}
        for member in tar_members:
            name = Path(member.filename).name
            kind = "det" if "det" in name.lower() else "rec" if "rec" in name.lower() else None
            if not kind or kind in archives:
                raise ValueError(f"cannot identify PPOCR component archive: {name}")
            output = workspace / name
            with package.open(member) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
            archives[kind] = output
    if set(archives) != {"det", "rec"}:
        raise ValueError("PPOCR source ZIP must contain one det archive and one rec archive")
    det_size = int(args.det_input_size)

    paddle2onnx = Path(sys.executable).with_name("paddle2onnx")
    if not paddle2onnx.is_file():
        raise RuntimeError("paddle2onnx executable is not available in the PPOCR exporter environment")
    for kind, shape, destination in (
        ("det", [1, 3, det_size, det_size], Path(args.det).resolve()),
        ("rec", [1, 3, 48, 320], Path(args.rec).resolve()),
    ):
        extracted = workspace / kind
        extracted.mkdir()
        safe_extract_tar(archives[kind], extracted)
        models = list(extracted.rglob("inference.pdmodel"))
        if len(models) != 1 or not models[0].with_name("inference.pdiparams").is_file():
            raise ValueError(f"{kind} archive does not contain one complete Paddle inference model")
        model_dir = models[0].parent
        dynamic = workspace / f"{kind}.dynamic.onnx"
        subprocess.run([
            str(paddle2onnx), "--model_dir", str(model_dir),
            "--model_filename", "inference.pdmodel", "--params_filename", "inference.pdiparams",
            "--save_file", str(dynamic), "--opset_version", "12", "--enable_dev_version", "True",
        ], check=True)
        subprocess.run([
            sys.executable, "-m", "paddle2onnx.optimize",
            "--input_model", str(dynamic), "--output_model", str(destination),
            "--input_shape_dict", str({"x": shape}),
        ], check=True)
    characters = Path(args.dictionary).read_text(encoding="utf-8").splitlines()
    rec_model = onnx.load(str(Path(args.rec).resolve()))
    rec_shape = rec_model.graph.output[0].type.tensor_type.shape
    rec_shape.ClearField("dim")
    for value in (1, 40, len(characters) + 2):
        rec_shape.dim.add().dim_value = value
    onnx.checker.check_model(rec_model)
    onnx.save(rec_model, str(Path(args.rec).resolve()))
    Path(args.metadata).write_text(json.dumps({
        "input_layout": "nhwc", "input_size": [det_size, det_size], "rec_input_size": [320, 48],
        "characters": characters + [" "],
        "model_files": {"det": "det.rknn", "rec": "rec.rknn"},
        "source_archives": {"det": "ch_PP-OCRv4_det_infer.tar", "rec": "ch_PP-OCRv4_rec_infer.tar"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
finally:
    shutil.rmtree(workspace, ignore_errors=True)
