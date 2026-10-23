import importlib.util
import io
import threading
import time
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("cosmos_service", Path(__file__).parents[1] / "app.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeModel:
    def generate(self, prompt, **options):
        if prompt == "bad":
            raise ValueError("input exceeds capacity")
        time.sleep(0.05)
        class Result:
            def as_dict(self):
                return {"text": "测试回复", "token_ids": [1, 2], "finish_reason": "eos",
                        "prompt_tokens": 5, "media_metadata": {}}
        return Result()


@pytest.fixture
def app(tmp_path):
    instance = module.create_app(FakeModel, preload=False, temp_root=tmp_path)
    instance.extensions["inference"].load()
    return instance


def test_text_returns_complete_json_and_cleans_temp(app, tmp_path):
    response = app.test_client().post("/v1/infer", json={"prompt": "Hello"})
    assert response.status_code == 200
    result = response.get_json()
    assert result["ok"] and result["status"] == "completed"
    assert result["result"]["text"] == "测试回复"
    assert result["result"]["token_ids"] == [1, 2]
    assert not list(tmp_path.iterdir())


def test_async_job_and_single_inflight(app):
    event = threading.Event()
    class Blocking(FakeModel):
        def generate(self, *args, **kwargs):
            event.wait(2)
            return super().generate(*args, **kwargs)
    service = app.extensions["inference"]
    service.model = Blocking()
    client = app.test_client()
    first = client.post("/api/jobs", data={"prompt": "Hello"})
    assert first.status_code == 202
    assert client.post("/api/jobs", data={"prompt": "Hello"}).status_code == 409
    event.set()
    service.jobs[first.json["id"]]["done"].wait(3)
    assert client.get("/api/jobs/" + first.json["id"]).json["status"] == "completed"


@pytest.mark.parametrize("payload", [{}, {"prompt": "Hi", "max_new_tokens": 257},
    {"prompt": "Hi", "max_new_tokens": True}, {"prompt": "Hi", "model": "other"},
    {"prompt": "Hi", "num_frames": 4}, {"prompt": "Hi", "path": "/etc/passwd"},
    {"prompt": "Hi", "greedy": "maybe"}])
def test_bad_input(app, payload):
    assert app.test_client().post("/v1/infer", json=payload).status_code in {400, 404}


def test_sdk_validation_error_is_400(app):
    response = app.test_client().post("/v1/infer", json={"prompt": "bad"})
    assert response.status_code == 400
    assert response.json["error"] == "input exceeds capacity"
    assert "http_status" not in response.json


def test_unready_cross_origin_and_host(app):
    client = app.test_client()
    assert client.get("/", headers={"Host": "evil.test"}).status_code == 403
    assert client.post("/v1/infer", json={"prompt": "Hi"}, headers={"Origin": "https://evil.test"}).status_code == 403
    app.extensions["inference"].state = "loading"
    assert client.post("/v1/infer", json={"prompt": "Hi"}).status_code == 503


def test_corrupt_or_multiple_media(app, tmp_path):
    client = app.test_client()
    assert client.post("/api/jobs", data={"prompt": "Hi", "image": (io.BytesIO(b"bad"), "test.jpg")}).status_code == 400
    assert client.post("/api/jobs", data={"prompt": "Hi", "image": (io.BytesIO(b"a"), "a.jpg"),
        "video": (io.BytesIO(b"a"), "b.mp4")}).status_code == 400
    assert not list(tmp_path.iterdir())
    assert app.extensions["inference"].slot.acquire(blocking=False)
    app.extensions["inference"].slot.release()


def test_image_with_safe_local_filename(app):
    from PIL import Image
    data = io.BytesIO()
    Image.new("RGB", (10, 10)).save(data, format="PNG")
    data.seek(0)
    response = app.test_client().post("/v1/infer", data={"prompt": "Describe", "image": (data, "../../a.png")})
    assert response.status_code == 200


def test_home_and_size_limit(app):
    assert app.test_client().get("/").status_code == 200
    app.config["MAX_CONTENT_LENGTH"] = 10
    assert app.test_client().post("/v1/infer", data=b"x" * 11, content_type="application/json").status_code == 413
