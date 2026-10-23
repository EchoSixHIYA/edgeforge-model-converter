import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from server.web import create_app
from server.catalog import BY_ID
from server.pipeline import Pipeline
from server.sophon_models import SOPHON_MODELS
from sdk_template.edgeforge_sdk.model import (
    Classification, MobileNetV2, YoloV10, _ModelBase,
    _decode, _nchw, _postprocess_classification, _postprocess_yolov5, _postprocess_yolox,
)


def test_health_and_catalog(tmp_path):
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path)})
    client = app.test_client()
    health_response = client.get("/health")
    assert health_response.status_code == 200
    health = health_response.get_json()
    assert {"ready", "rknn_toolkit2", "model_zoo"}.issubset(health["rockchip"])
    assert {"ready", "tpu_mlir", "sophon_demo", "sophon_sail"}.issubset(health["sophon"])
    assert "sophon_demo" in health
    data = client.get("/api/catalog").get_json()
    assert len(data["models"]) >= 33
    assert {"rk3588", "rk3566"}.issubset(data["platforms"])
    assert {"bm1684", "bm1684x"}.issubset(data["platforms"])
    assert {vendor["id"] for vendor in data["vendors"]} >= {"rockchip", "sophon"}
    assert {backend["id"] for backend in data["backends"]} >= {"rknn", "bmodel"}
    assert len(data["models"]) == 34
    assert all("status" in model and "blocker" in model for model in data["models"])
    assert data["models"][2]["vendor_support"]["sophon"]["platforms"] == ["bm1684", "bm1684x"]
    assert BY_ID["yolov8_obb"]["vendor_support"]["sophon"]["input_size"] == [1024, 1024]
    assert BY_ID["yolox"]["vendor_support"]["sophon"]["weight_formats"] == [".pth"]
    assert BY_ID["yolov9"]["vendor_support"]["sophon"]["blocker"] == ""
    assert BY_ID["yolov9"]["vendor_support"]["rockchip"]["available"] is False
    assert {model["id"] for model in data["models"]} == {
        "mobilenet", "resnet", "yolov5", "yolov6", "yolov7", "yolov8", "yolov8_obb",
        "yolov8_cls", "yolov9", "yolov9_seg", "yolov10", "yolo11", "yolo11_obb",
        "yolov12", "yolox", "ppyoloe", "yolo_world", "yolov8_pose", "deeplabv3",
        "yolov5_seg", "yolov8_seg", "ppseg", "mobilesam", "retinaface", "lprnet", "ppocr",
        "lite_transformer", "clip", "wav2vec2", "whisper", "zipformer", "yamnet", "mms_tts",
        "cosmos_reason2_2b",
    }


def test_sdk_docs_and_example_image_are_shipped(tmp_path):
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path)})
    response = app.test_client().get("/docs/sdk-api")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    for method in ("`load()`", "`predict()`", "`infer_raw()`", "`postprocess()`"):
        assert method in body

    image = Path(app.root_path).parent / "sdk_template" / "test.jpg"
    assert image.is_file()
    assert image.read_bytes()[:2] == b"\xff\xd8"
    installer = Path(app.root_path).parent / "sdk_template" / "install.sh"
    sdk_readme = (Path(app.root_path).parent / "sdk_template" / "SDK_README.md").read_text(encoding="utf-8")
    wheels = list((Path(app.root_path).parent / "sdk_template" / "runtime" /
                   "rknn-toolkit-lite2" / "packages").glob("*.whl"))
    assert installer.is_file()
    assert len(wheels) == 6
    assert "bash install.sh" in sdk_readme
    assert "RKNN-Toolkit-Lite2 2.3.2" in sdk_readme
    assert "不提供自动识别入口" in sdk_readme


def test_frontend_script_has_valid_sophon_health_copy():
    script = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text(encoding="utf-8")
    assert "health.sophon?.ready" in script
    assert "support.weight_formats" in script
    assert "待板端验收" not in script
    assert "querySelectorAll('optgroup')" in script
    assert "SOPHGO 环境就绪" in script
    assert "转换环境组件待安装" in script
    assert "鐜" not in script


def test_architecture_selector_uses_vendor_neutral_groups(tmp_path):
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path)})
    body = app.test_client().get("/").get_data(as_text=True)
    assert "可提交转换" in body
    assert "暂未开放提交" in body
    assert "RK 官方目录（待完成闭环）" not in body
    assert "data-label=" in body


def test_verified_mobilenet_checkpoint_is_accepted(tmp_path):
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path)})
    response = app.test_client().post("/api/tasks", data={
        "vendor": "rockchip", "architecture": "mobilenet", "platform": "rk3588", "precision": "fp",
        "model": (io.BytesIO(b"weights"), "model.pth"),
    })
    assert response.status_code == 202
    assert response.get_json()["architecture"] == "mobilenet"


def test_sophon_yolo_task_is_accepted_with_bm1684x_precision(tmp_path):
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path), "SKIP_SOPHON_ENV_CHECK": True})
    response = app.test_client().post("/api/tasks", data={
        "vendor": "sophon", "architecture": "yolov8", "platform": "bm1684x", "precision": "f16",
        "model": (io.BytesIO(b"weights"), "model.pt"),
    })
    assert response.status_code == 202
    task = response.get_json()
    assert task["vendor"] == "sophon"
    assert task["backend"] == "bmodel"


@pytest.mark.parametrize("architecture,filename", [
    ("yolov8", "model.pt"),
    ("yolo_world", "model.pth"),
    ("ppocr", "model.zip"),
])
def test_sophon_priority_models_accept_bm1684_target(tmp_path, architecture, filename):
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path), "SKIP_SOPHON_ENV_CHECK": True})
    response = app.test_client().post("/api/tasks", data={
        "vendor": "sophon", "architecture": architecture, "platform": "bm1684", "precision": "f16",
        "model": (io.BytesIO(b"weights"), filename),
    })
    assert response.status_code == 202
    task = response.get_json()
    assert task["vendor"] == "sophon"
    assert task["platform"] == "bm1684"
    assert task["backend"] == "bmodel"


def test_sophon_rejects_non_bm_platform(tmp_path):
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path), "SKIP_SOPHON_ENV_CHECK": True})
    response = app.test_client().post("/api/tasks", data={
        "vendor": "sophon", "architecture": "yolov8", "platform": "rk3588", "precision": "f16",
        "model": (io.BytesIO(b"weights"), "model.pt"),
    })
    assert response.status_code == 400


def test_sophon_yolov8_obb_accepts_vendor_specific_size(tmp_path):
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path), "SKIP_SOPHON_ENV_CHECK": True})
    response = app.test_client().post("/api/tasks", data={
        "vendor": "sophon", "architecture": "yolov8_obb", "platform": "bm1684x", "precision": "f16",
        "model": (io.BytesIO(b"weights"), "model.pt"),
    })
    assert response.status_code == 202
    assert response.get_json()["backend"] == "bmodel"


def test_sophon_only_yolo_model_does_not_open_rockchip_entry(tmp_path):
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path), "SKIP_SOPHON_ENV_CHECK": True})
    client = app.test_client()
    sophon = client.post("/api/tasks", data={
        "vendor": "sophon", "architecture": "yolov9", "platform": "bm1684x", "precision": "f16",
        "model": (io.BytesIO(b"weights"), "model.pt"),
    })
    rockchip = client.post("/api/tasks", data={
        "vendor": "rockchip", "architecture": "yolov9", "platform": "rk3588", "precision": "fp",
        "model": (io.BytesIO(b"weights"), "model.pt"),
    })
    assert sophon.status_code == 202
    assert rockchip.status_code == 400


def test_sophon_uses_vendor_specific_upload_formats(tmp_path):
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path), "SKIP_SOPHON_ENV_CHECK": True})
    response = app.test_client().post("/api/tasks", data={
        "vendor": "sophon", "architecture": "yolox", "platform": "bm1684x", "precision": "f16",
        "model": (io.BytesIO(b"weights"), "model.pt"),
    })
    assert response.status_code == 400
    assert ".pth" in response.get_json()["error"]


def test_sophon_task_requires_ready_compiler_environment(monkeypatch, tmp_path):
    monkeypatch.setattr("server.web.shutil.which", lambda _: None)
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path)})

    response = app.test_client().post("/api/tasks", data={
        "vendor": "sophon", "architecture": "ppocr", "platform": "bm1684x", "precision": "f16",
        "model": (io.BytesIO(b"weights"), "model.zip"),
    })

    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "SOPHGO BModel 转换环境未就绪" in error
    assert "model_transform.py" in error
    assert "model_deploy.py" in error


def test_generated_sdk_contains_runnable_example_assets(tmp_path):
    root = Path(__file__).resolve().parents[1]
    pipeline = Pipeline(root)
    pipeline.model_zoo = tmp_path / "missing-model-zoo"
    onnx_path = tmp_path / "model.onnx"
    rknn_path = tmp_path / "model.rknn"
    onnx_path.write_bytes(b"test onnx placeholder")
    rknn_path.write_bytes(b"test rknn placeholder")
    (tmp_path / "export_metadata.json").write_text(json.dumps({
        "labels": ["custom-class"],
        "anchors": [[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]],
                    [[116, 90], [156, 198], [373, 326]]],
    }), encoding="utf-8")
    task = {"platform": "rk3588", "precision": "fp", "vendor": "rockchip", "backend": "rknn"}

    archive = pipeline._package_sdk(
        task, BY_ID["yolov5"], {"model": onnx_path}, {"model": rknn_path}, tmp_path)
    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
        assert {"test.jpg", "example.py", "install.sh", "SDK_README.md",
                "edgeforge_sdk/config.json", "model/config.json"}.issubset(names)
        assert any(name.startswith("runtime/rknn-toolkit-lite2/packages/") and
                   name.endswith("cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl")
                   for name in names)
        config = json.loads(package.read("edgeforge_sdk/config.json"))
        example = package.read("example.py").decode("utf-8")
        assert "from edgeforge_sdk import YoloV5" in example
        assert 'with YoloV5("model") as model:' in example
        assert "EdgeModel" not in example
        assert config["modality"] == "image"
        assert config["task_type"] == "object_detection"
        assert config["result_type"] == "Detection"
        assert config["labels"] == ["custom-class"]
        assert config["anchors"][0][0] == [10, 13]
        assert json.loads(package.read("model/config.json")) == config


def test_sophon_package_uses_sophon_demo_reference(tmp_path):
    root = Path(__file__).resolve().parents[1]
    pipeline = Pipeline(root)
    pipeline.sophon_demo = tmp_path / "sophon-demo"
    sample = pipeline.sophon_demo / "sample" / SOPHON_MODELS["yolov8_obb"]["sample"]
    (sample / "python").mkdir(parents=True)
    (sample / "python" / "README.md").write_text("sophon obb reference", encoding="utf-8")
    onnx_path = tmp_path / "model.onnx"
    bmodel_path = tmp_path / "model.bmodel"
    onnx_path.write_bytes(b"onnx")
    bmodel_path.write_bytes(b"bmodel")
    task = {"platform": "bm1684x", "precision": "f16", "vendor": "sophon", "backend": "bmodel"}

    archive = pipeline._package_sdk(
        task, BY_ID["yolov8_obb"], {"model": onnx_path}, {"model": bmodel_path}, tmp_path)

    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        assert "reference/python/README.md" in names
        assert "edgeforge_sdk/model.bmodel" in names
        readme = package.read("SDK_README.md").decode("utf-8")
        usage = package.read("使用说明.txt").decode("utf-8")
        installer = package.read("install.sh").decode("utf-8")
        assert ".rknn` / `.bmodel`" in readme
        assert "SOPHON-SAIL" in readme
        assert "BModel" in usage
        assert "SOPHON-SAIL" in usage
        assert "BMODELS=" in installer
        assert "RKNN_MODELS=" in installer
        assert "sophon.sail" in installer
        config = json.loads(package.read("edgeforge_sdk/config.json"))
        assert config["vendor"] == "sophon"
        assert config["backend"] == "bmodel"


def test_rejects_unknown_architecture(tmp_path):
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path)})
    response = app.test_client().post("/api/tasks", data={
        "vendor": "rockchip", "architecture": "unknown", "platform": "rk3588", "precision": "fp",
        "model": (io.BytesIO(b"x"), "model.onnx"),
    })
    assert response.status_code == 400


def test_requires_supported_upload_type(tmp_path):
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path)})
    response = app.test_client().post("/api/tasks", data={
        "vendor": "rockchip", "architecture": "yolov10", "platform": "rk3588", "precision": "fp",
        "model": (io.BytesIO(b"x"), "model.exe"),
    })
    assert response.status_code == 400


def test_rejects_preconverted_onnx(tmp_path):
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path)})
    response = app.test_client().post("/api/tasks", data={
        "vendor": "rockchip", "architecture": "yolov10", "platform": "rk3588", "precision": "fp",
        "model": (io.BytesIO(b"onnx"), "model.onnx"),
    })
    assert response.status_code == 400
    assert "原始权重" in response.get_json()["error"]


def test_sdk_output_layout_detection():
    assert _nchw(np.zeros((1, 80, 20, 20))).shape == (1, 80, 20, 20)
    assert _nchw(np.zeros((1, 20, 20, 80))).shape == (1, 80, 20, 20)


def test_yolov6_four_channel_distance_decode_does_not_apply_dfl():
    distances = np.ones((1, 4, 2, 2), dtype=np.float32)
    boxes = _decode(distances, (640, 640))
    assert boxes.shape == (4, 4)
    assert boxes[0].tolist() == pytest.approx([-160.0, -160.0, 480.0, 480.0])


def test_yolox_exp_grid_decode_returns_structured_candidate():
    output = np.zeros((1, 7, 1, 1), dtype=np.float32)
    output[0, 2:4, 0, 0] = 0.0
    output[0, 4, 0, 0] = 0.9
    output[0, 5, 0, 0] = 0.8
    boxes, classes, scores = _postprocess_yolox(
        [output], (640, 640), 2, confidence=0.25, iou=0.45)
    assert boxes.shape == (1, 4)
    assert boxes[0].tolist() == pytest.approx([-320.0, -320.0, 320.0, 320.0])
    assert classes.tolist() == [0]
    assert scores[0] == pytest.approx(0.72)


def test_yolov5_anchor_postprocess_returns_structured_candidate():
    nc = 2
    anchors = [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]],
    ]
    outputs = [np.zeros((1, 3 * (nc + 5), size, size), dtype=np.float32)
               for size in (80, 40, 20)]
    first = outputs[0].reshape(1, 3, nc + 5, 80, 80)
    first[0, 0, 0:4, 0, 0] = 0.5
    first[0, 0, 4, 0, 0] = 0.9
    first[0, 0, 5, 0, 0] = 0.8

    boxes, classes, scores = _postprocess_yolov5(
        outputs, (640, 640), nc, anchors, confidence=0.25, iou=0.45)

    assert boxes.shape == (1, 4)
    assert classes.tolist() == [0]
    assert scores[0] == pytest.approx(0.72)


def test_sdk_preprocess_adds_rknn_batch_dimension():
    model = object.__new__(_ModelBase)
    model.config = {"modality": "image", "input_size": [640, 640]}
    tensor, context = model.preprocess(np.zeros((360, 480, 3), dtype=np.uint8))
    assert tensor.shape == (1, 640, 640, 3)
    assert tensor.dtype == np.uint8
    assert context["original_shape"] == [360, 480]


def test_classification_preprocess_and_top_k_results():
    model = object.__new__(_ModelBase)
    model.config = {"modality": "image", "task_type": "classification",
                    "input_size": [224, 224], "color_order": "bgr"}
    source = np.zeros((360, 480, 3), dtype=np.uint8)
    source[..., 0] = 17
    tensor, context = model.preprocess(source)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor[0, 0, 0].tolist() == [17, 0, 0]
    assert context == {"original_shape": [360, 480]}

    results = _postprocess_classification(
        [np.array([[0.0, 1.0, 3.0]], dtype=np.float32)], ["a", "b", "c"], 2)
    assert all(isinstance(item, Classification) for item in results)
    assert [item.label for item in results] == ["c", "b"]
    assert [item.rank for item in results] == [1, 2]
    assert sum(item.score for item in _postprocess_classification(
        [np.array([[0.0, 1.0, 3.0]], dtype=np.float32)], [], 3)) == pytest.approx(1.0)


def test_sdk_infer_raw_expands_hwc_and_rejects_missing_outputs():
    class FakeEngine:
        def __init__(self):
            self.inputs = None

        def inference(self, *, inputs):
            self.inputs = inputs
            return None

    model = object.__new__(_ModelBase)
    model.engine = FakeEngine()
    model.load = lambda: model
    with pytest.raises(RuntimeError, match="returned no outputs"):
        model.infer_raw(np.zeros((640, 640, 3), dtype=np.uint8))
    assert model.engine.inputs[0].shape == (1, 640, 640, 3)


def test_explicit_sdk_class_loads_external_model_bundle(tmp_path):
    bundle = tmp_path / "mobilenet_bundle"
    bundle.mkdir()
    (bundle / "custom.rknn").write_bytes(b"rknn")
    (bundle / "config.json").write_text(json.dumps({
        "architecture": "mobilenet", "modality": "image",
        "task_type": "classification", "input_size": [224, 224],
    }), encoding="utf-8")
    model = MobileNetV2(bundle)
    assert model.model_path == bundle / "custom.rknn"
    assert model.config["architecture"] == "mobilenet"

    with pytest.raises(ValueError, match="config not found"):
        MobileNetV2(bundle / "custom.rknn", config_path=bundle / "missing.json")

    with pytest.raises(ValueError, match="requires architecture=yolov10"):
        YoloV10(bundle)


def test_explicit_sdk_class_loads_external_bmodel_bundle(tmp_path):
    bundle = tmp_path / "yolov8_bmodel_bundle"
    bundle.mkdir()
    (bundle / "custom.bmodel").write_bytes(b"bmodel")
    (bundle / "config.json").write_text(json.dumps({
        "architecture": "yolov8", "vendor": "sophon", "backend": "bmodel",
        "modality": "image", "task_type": "object_detection", "input_size": [640, 640],
    }), encoding="utf-8")
    from sdk_template.edgeforge_sdk.model import YoloV8

    model = YoloV8(bundle)
    assert model.model_path == bundle / "custom.bmodel"
    assert model.config["backend"] == "bmodel"


def test_automatic_edge_model_entry_is_not_public():
    import sdk_template.edgeforge_sdk as sdk

    assert not hasattr(sdk, "EdgeModel")
    assert sdk.YoloV9.architecture == "yolov9"
    assert sdk.YoloV9Seg.architecture == "yolov9_seg"
    assert sdk.Yolo11Obb.architecture == "yolo11_obb"
    assert sdk.YoloV12.architecture == "yolov12"
