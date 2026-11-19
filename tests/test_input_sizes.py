import io
import json
import zipfile

import numpy as np
import pytest

from sdk_template.edgeforge_sdk.model import YoloV8, _nchw, _postprocess_yolov8_obb
from server.catalog import BY_ID
from server.exporters import ArchitectureExporters
from server.pipeline import Pipeline
from server.web import create_app


@pytest.fixture
def cv_api(tmp_path, monkeypatch):
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path), "SKIP_SOPHON_ENV_CHECK": True})
    manager = app.extensions["task_manager"]
    monkeypatch.setattr(manager.executor, "submit", lambda *args: None)

    def submit(architecture="yolov8", vendor="rockchip", **size_fields):
        platform, precision = ("bm1684x", "f16") if vendor == "sophon" else ("rk3588", "fp")
        return app.test_client().post("/api/tasks", data={
            "vendor": vendor, "architecture": architecture, "platform": platform,
            "precision": precision, "model": (io.BytesIO(b"weights"), "model.pt"), **size_fields,
        })

    return submit, tmp_path


@pytest.mark.parametrize("vendor", ["rockchip", "sophon"])
@pytest.mark.parametrize("size", [[640, 384], [1920, 1088], [2080, 96]])
def test_api_accepts_aligned_sizes_without_extra_bounds(cv_api, vendor, size):
    submit, path = cv_api
    response = submit(vendor=vendor, input_width=str(size[0]), input_height=str(size[1]))
    assert response.status_code == 202
    task = response.get_json()
    assert task["input_size"] == size
    data = json.loads((path / task["id"] / "task.json").read_text(encoding="utf-8"))
    assert data["input_size"] == size
    assert BY_ID["yolov8"]["input_size"] == [640, 640]


@pytest.mark.parametrize("vendor,expected", [("rockchip", [640, 640]), ("sophon", [1024, 1024])])
def test_api_preserves_vendor_default_without_custom_fields(cv_api, vendor, expected):
    submit, _ = cv_api
    response = submit(architecture="yolov8_obb", vendor=vendor)
    assert response.status_code == 202
    assert response.get_json()["input_size"] == expected


@pytest.mark.parametrize("size_fields", [
    {"input_width": "640"}, {"input_height": "384"},
    {"input_width": "640.0", "input_height": "384"},
    {"input_width": "-1", "input_height": "384"},
    {"input_width": "640", "input_height": "0"},
    {"input_width": "", "input_height": "384"},
    {"input_width": ["640", "320"], "input_height": "384"},
])
def test_api_rejects_bad_sizes_before_saving_upload(cv_api, size_fields):
    submit, path = cv_api
    response = submit(**size_fields)
    assert response.status_code == 400
    assert "输入" in response.get_json()["error"]
    assert not list(path.iterdir())


@pytest.mark.parametrize("vendor", ["rockchip", "sophon"])
def test_api_rejects_unaligned_size_before_saving_upload(cv_api, vendor):
    submit, path = cv_api
    response = submit(vendor=vendor, input_width="1920", input_height="1080")
    assert response.status_code == 400
    assert "32 的倍数" in response.get_json()["error"]
    assert not list(path.iterdir())


def test_api_rejects_dimensions_for_fixed_architecture(cv_api):
    submit, path = cv_api
    response = submit(architecture="mobilesam", input_width="640", input_height="384")
    assert response.status_code == 400
    assert "固定输入尺寸" in response.get_json()["error"]
    assert not list(path.iterdir())


@pytest.mark.parametrize("vendor,architecture,flag", [
    (vendor, arch, "--img-size" if arch in {"yolov6", "yolov7"} else "--imgsz")
    for arch, cfg in BY_ID.items() for vendor, support in cfg["vendor_support"].items()
    if support.get("custom_input_size")
])
def test_exporters_pass_height_then_width(tmp_path, vendor, architecture, flag):
    exporter = ArchitectureExporters(tmp_path)
    source = tmp_path / "upload.pt"
    calls = []

    def run(command, cwd):
        calls.append(command)
        source.with_suffix(".onnx").write_bytes(b"mock ONNX")

    exporter._run = run
    exporter.export(source, tmp_path, architecture, vendor, lambda *_: None, input_size=[640, 384])
    assert calls[0][calls[0].index(flag) + 1:][:2] == ["384", "640"]


def make_onnx(path, shape):
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph([helper.make_node("Identity", ["image"], ["output"])], "identity",
                              [helper.make_tensor_value_info("image", TensorProto.FLOAT, shape)],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)])
    onnx.save(helper.make_model(graph), str(path))


@pytest.mark.parametrize("shape", [[1, 3, 640, 384], [1, 3, 640, 640], [1, 3, "h", "w"]])
def test_rejects_rounded_transposed_or_dynamic_onnx_shape(tmp_path, shape):
    path = tmp_path / "model.onnx"
    make_onnx(path, shape)
    with pytest.raises(ValueError, match="ONNX 输入尺寸.*不一致"):
        Pipeline(tmp_path)._check_onnx(path, lambda *_: None, input_size=[640, 384])


def test_onnx_size_check_uses_graph_instead_of_a_global_stride(tmp_path):
    path = tmp_path / "model.onnx"
    pipeline = Pipeline(tmp_path)
    make_onnx(path, [1, 3, 1080, 1920])
    pipeline._check_onnx(path, lambda *_: None, input_size=[1920, 1080])
    make_onnx(path, [1, 3, 1088, 1920])
    with pytest.raises(ValueError, match="1920 × 1088.*1920 × 1080"):
        pipeline._check_onnx(path, lambda *_: None, input_size=[1920, 1080])


@pytest.mark.parametrize("vendor,backend,platform,precision", [
    ("rockchip", "rknn", "rk3588", "fp"), ("sophon", "bmodel", "bm1684x", "f16"),
])
def test_pipeline_uses_verified_size_for_compiler_and_sdk(tmp_path, monkeypatch, vendor, backend, platform, precision):
    from pathlib import Path

    pipeline = Pipeline(Path(__file__).resolve().parents[1])
    # Use a tiny SDK template so packaging regressions don't copy board runtime wheels.
    pipeline.root = tmp_path
    pipeline.model_zoo = tmp_path / "zoo"
    pipeline.sophon_demo = tmp_path / "sophon"
    (tmp_path / "sdk_template" / "edgeforge_sdk").mkdir(parents=True)
    (tmp_path / "sdk_template" / "SDK_README.md").touch()
    source = tmp_path / "upload.pt"
    source.write_bytes(b"weights")
    work = tmp_path / "work"
    work.mkdir()

    def export(source, work, architecture, vendor, update, *, input_size):
        assert input_size == [640, 384]
        path = work / "model.onnx"
        make_onnx(path, [1, 3, 384, 640])
        return path

    def convert(source, output, options, progress):
        assert options["catalog"]["input_size"] == [640, 384]
        output.write_bytes(b"compiled")

    monkeypatch.setattr(pipeline.exporters, "export", export)
    monkeypatch.setattr(pipeline.backends.get(backend), "convert", convert)
    artifacts = pipeline.run({
        "architecture": "yolov8", "vendor": vendor, "backend": backend, "platform": platform,
        "precision": precision, "source_path": str(source), "work_dir": str(work), "input_size": [640, 384],
    }, lambda *_: None)
    with zipfile.ZipFile(artifacts["package"]) as sdk:
        for name in ["edgeforge_sdk/config.json", "model/config.json", "manifest.json"]:
            assert json.loads(sdk.read(name))["input_size"] == [640, 384]


def test_sophon_sdk_predict_uses_rectangular_float_nchw_and_dense_output():
    model = object.__new__(YoloV8)
    model.config = {"architecture": "yolov8", "task_type": "object_detection", "backend": "bmodel",
                    "input_size": [640, 384], "labels": ["person", "bus"]}
    image = np.full((192, 320, 3), [10, 20, 30], dtype=np.uint8)

    def infer_raw(tensor):
        assert tensor.shape == (1, 3, 384, 640)
        assert tensor.dtype == np.float32
        assert tensor.flags.c_contiguous
        np.testing.assert_allclose(tensor[0, :, 0, 0], np.array([30, 20, 10]) / 255)
        rows = np.array([[320, 192, 128, 64, .1, .9],
                         [320, 192, 128, 64, .1, .8],
                         [320, 192, 128, 64, .85, .1]], dtype=np.float32)
        return [rows.T[None]]

    model.infer_raw = infer_raw
    results = model.predict(image)
    assert [item.label for item in results] == ["bus", "person"]
    assert results[0].box == pytest.approx((128, 80, 192, 112))


def test_rockchip_rectangular_output_layout_and_obb_angle_count():
    feature = np.zeros((1, 48, 80, 64), dtype=np.float32)
    assert _nchw(feature, (640, 384), {64}).shape == (1, 64, 48, 80)
    assert _nchw(feature.transpose(0, 3, 1, 2), (640, 384), {64}).shape == (1, 64, 48, 80)
    outputs = [np.zeros((1, 65, 384 // stride, 640 // stride), dtype=np.float32)
               for stride in (8, 16, 32)]
    outputs[0][0, 64, 0, 0] = 8
    outputs.append(np.zeros((1, 1, sum(out.shape[2] * out.shape[3] for out in outputs)), dtype=np.float32))
    rects, classes, scores = _postprocess_yolov8_obb(outputs, (640, 384), 1, .6, .45)
    assert len(rects) == 1
    assert classes.tolist() == [0]
    assert scores[0] > .99
