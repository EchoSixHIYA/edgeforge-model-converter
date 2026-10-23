"""Rendered UI contract tests; no compiler, browser renderer or board is exercised."""
from html.parser import HTMLParser
from pathlib import Path

import pytest

from server.web import create_app

ROOT = Path(__file__).resolve().parents[1]


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = {}
        self.all_ids = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.all_ids.append(attrs["id"])
            self.ids[attrs["id"]] = (tag, attrs)


@pytest.fixture
def rendered(tmp_path):
    client = create_app({"TESTING": True, "DATA_DIR": str(tmp_path)}).test_client()
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    elements = Elements()
    elements.feed(body)
    return body, elements


def test_vlm_ui_is_separate_local_directory_workflow(rendered):
    body, elements = rendered
    assert len(elements.all_ids) == len(set(elements.all_ids))
    assert elements.ids["convertForm"][1]["enctype"] == "multipart/form-data"
    assert elements.ids["modelFile"][1]["type"] == "file"
    assert "hidden" in elements.ids["vlmForm"][1]["class"]
    assert elements.ids["vlmModelDir"][1]["name"] == "model_dir"
    assert "type" not in elements.ids["vlmModelDir"][1]  # Plain path, not a directory upload.
    assert "EDGEFORGE_MODEL_ROOTS" in body
    assert "服务内部可见的绝对路径" in body
    assert "vlmAcknowledge" not in elements.ids
    for phrase in ("实验性", "尚未板端验收", "不能保证", "不保证", "免责"):
        assert phrase not in body
    assert "disabled" in elements.ids["vlmSubmit"][1]
    assert "max_frames" not in body
    assert "每个视频时序组（2 帧）" in body
    assert "上下文包含提示词、视觉 token 与生成内容" in body


@pytest.mark.parametrize("element_id,minimum,maximum,step", [
    ("vlmSeqLength", "256", "8192", "128"),
    ("vlmMaxPixels", "65536", "1048576", "1024"),
    ("vlmJobs", "1", "1", "1"),
])
def test_vlm_budget_bounds_match_backend(rendered, element_id, minimum, maximum, step):
    _, elements = rendered
    attributes = elements.ids[element_id][1]
    assert (attributes["min"], attributes["max"], attributes["step"]) == (minimum, maximum, step)
    if element_id == "vlmJobs":
        assert "readonly" in attributes


def test_vlm_ui_wire_protocol_and_recovery_guards():
    script = (ROOT / "static" / "app.js").read_text()
    for value in ("/api/vlm/preflight", "/api/vlm/tasks", "X-EdgeForge-Local-Import",
                  "acknowledge_experimental", "seq_length:", "max_pixels:", "jobs:",
                  "vlmCheckedFingerprint", "AbortController", "version !== vlmCheckVersion",
                  "log_tail", "/download/log", "/download/package", "/cancel", "canceled"):
        assert value in script
    assert "max_frames:" not in script
    assert "new FormData(form)" in script  # Keep the pre-existing CV upload protocol.
    assert "待板端验收" not in script
    assert "不是编译成功、模型精度或板端验收证明" not in script
    assert "acknowledge_experimental: true" in script


def test_vlm_javascript_logic_with_dom_stubs():
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is optional; install it to run JS logic regression")
    result = subprocess.run([node, str(ROOT / "tests" / "ui" / "vlm_dom_unit.cjs")],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout


def test_low_memory_preset_is_opt_in_and_preserves_global_defaults(rendered):
    body, elements = rendered
    assert elements.ids["vlmLowMemoryPreset"][1]["type"] == "button"
    assert "低内存预设" in body
    assert elements.ids["vlmSeqLength"][1]["value"] == "2048"
    assert elements.ids["vlmMaxPixels"][1]["value"] == "147456"
    assert "6GB" not in body  # No device capacity is inferred from Linux free output.
    script = (ROOT / "static" / "app.js").read_text()
    assert "applyVlmLowMemoryPreset" in script
    assert "device_memory_estimate" in script
    assert "预计内存" in script
    assert "转换电脑资源" in script
