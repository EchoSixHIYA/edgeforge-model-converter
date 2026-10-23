"""Single-board, single-model LAN service for the exported Cosmos SDK."""
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

ROOT = Path(__file__).resolve().parent
MODEL_ID = "cosmos-reason2-2b"
LOG = logging.getLogger(__name__)


class ServiceError(Exception):
    def __init__(self, message, status=400):
        self.status = status
        super().__init__(message)


class InferenceService:
    def __init__(self, model_factory, temp_root):
        self.model_factory = model_factory
        self.temp_root = Path(temp_root)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.state = "loading"
        self.error = None
        self.lock = threading.Lock()
        self.slot = threading.BoundedSemaphore(1)
        self.jobs = {}

    def load(self):
        try:
            model = self.model_factory()
            with self.lock:
                self.model, self.state = model, "ready"
        except Exception as exc:
            LOG.exception("Model initialization failed")
            with self.lock:
                self.state, self.error = "error", str(exc)

    def status(self):
        with self.lock:
            return {"status": self.state, "model": MODEL_ID, "error": self.error,
                    "busy": any(j["status"] in {"queued", "running"} for j in self.jobs.values()),
                    "platform": "bm1684x", "precision": "w4bf16",
                    "sequence_length": 2048, "max_pixels": 147456,
                    "max_new_tokens": 256, "concurrency": 1}

    def get_job(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise ServiceError("任务不存在或已过期", 404)
            return {key: value for key, value in job.items() if key != "done"}

    def submit(self, fields, files):
        if self.status()["status"] != "ready":
            raise ServiceError("模型尚未就绪，请查看 /health 和服务日志", 503)
        if not self.slot.acquire(blocking=False):
            raise ServiceError("模型正在推理，请等待当前任务完成后重试", 409)
        temporary = None
        try:
            prompt, options, media_type = parse_options(fields, files)
            temporary = tempfile.TemporaryDirectory(prefix="request-", dir=self.temp_root)
            if media_type:
                upload = files[media_type]
                suffix = Path(upload.filename).suffix.lower()
                allowed = {"image": {".jpg", ".jpeg", ".png", ".webp", ".bmp"},
                           "video": {".mp4", ".mov", ".mkv", ".webm", ".avi"}}
                if suffix not in allowed[media_type]:
                    raise ServiceError("不支持的媒体格式")
                media_path = Path(temporary.name) / ("input" + suffix)
                upload.save(media_path)
                if not media_path.stat().st_size:
                    raise ServiceError("上传文件为空")
                if media_type == "image":
                    validate_image(media_path)
                else:
                    validate_video(media_path)
                options[media_type] = str(media_path)
            job_id = uuid.uuid4().hex
            done = threading.Event()
            with self.lock:
                # Retain at most 20 result records; never store uploaded files.
                while len(self.jobs) >= 20:
                    del self.jobs[next(iter(self.jobs))]
                self.jobs[job_id] = {"id": job_id, "model": MODEL_ID, "status": "queued", "done": done}
            threading.Thread(target=self._run, args=(job_id, prompt, options, temporary), daemon=True).start()
            return job_id, done
        except Exception:
            if temporary is not None:
                temporary.cleanup()
            self.slot.release()
            raise

    def _run(self, job_id, prompt, options, temporary):
        started = time.monotonic()
        with self.lock:
            self.jobs[job_id]["status"] = "running"
        try:
            result = self.model.generate(prompt, **options)
            output = {"status": "completed", "ok": True, "result": result.as_dict()}
        except (ValueError, ServiceError) as exc:
            output = {"status": "failed", "ok": False, "error": str(exc), "http_status": 400}
        except Exception as exc:
            LOG.exception("Inference failed: %s", job_id)
            output = {"status": "failed", "ok": False, "error": str(exc), "http_status": 500}
        finally:
            try:
                temporary.cleanup()
            except OSError:
                LOG.exception("Could not clean request media")
        with self.lock:
            self.jobs[job_id].update(output, elapsed_ms=round((time.monotonic() - started) * 1000, 2))
            self.jobs[job_id]["done"].set()
        self.slot.release()


def integer(fields, name, default, minimum, maximum):
    value = fields.get(name, default)
    try:
        if isinstance(value, bool) or str(int(value)) != str(value):
            raise ValueError()
        value = int(value)
        if not minimum <= value <= maximum:
            raise ValueError()
    except (TypeError, ValueError):
        raise ServiceError(f"{name} 必须是 {minimum}–{maximum} 之间的整数")
    return value


def parse_options(fields, files):
    allowed = {"model", "prompt", "max_new_tokens", "greedy", "num_frames"}
    if set(fields) - allowed or set(files) - {"image", "video"}:
        raise ServiceError("请求包含未知字段")
    if fields.get("model", MODEL_ID) != MODEL_ID:
        raise ServiceError("未知模型；当前可选模型为 " + MODEL_ID, 404)
    prompt = fields.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
        raise ServiceError("prompt 不能为空，且不能超过 8000 个字符")
    active = [name for name in ("image", "video") if name in files and files[name].filename]
    if len(active) > 1:
        raise ServiceError("image 和 video 不能同时上传")
    greedy = fields.get("greedy", True)
    if greedy in (True, "true", "1", 1):
        greedy = True
    elif greedy in (False, "false", "0", 0):
        greedy = False
    else:
        raise ServiceError("greedy 必须是 true 或 false")
    options = {"max_new_tokens": integer(fields, "max_new_tokens", 64, 1, 256),
               "generation": {"do_sample": not greedy}, "min_pixels": 4096, "max_pixels": 147456}
    media_type = active[0] if active else None
    if media_type == "video":
        options["num_frames"] = integer(fields, "num_frames", 4, 2, 8)
    elif "num_frames" in fields:
        raise ServiceError("num_frames 仅适用于视频")
    return prompt.strip(), options, media_type


def validate_image(path):
    from PIL import Image
    try:
        with Image.open(path) as image:
            if image.width * image.height > 20_000_000:
                raise ServiceError("图片不得超过 2000 万像素")
            image.verify()
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError("图片无法解码，请使用有效的 JPEG/PNG/WebP/BMP") from exc


def validate_video(path):
    import av
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ServiceError("文件中没有视频流")
            if container.duration is None or container.duration / av.time_base > 30:
                raise ServiceError("请上传时长不超过 30 秒、时间信息完整的视频")
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError("视频无法解析") from exc


def create_app(model_factory=None, *, preload=True, temp_root=None):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=32 * 1024**2, MAX_FORM_MEMORY_SIZE=128 * 1024)
    app.json.ensure_ascii = False
    if model_factory is None:
        def model_factory():
            from edgeforge_sdk import CosmosReason
            return CosmosReason(os.environ.get("COSMOS_MODEL_DIR", str(ROOT / "sdk" / "model")), device_id=0)
    service = InferenceService(model_factory, temp_root or ROOT / "tmp")
    app.extensions["inference"] = service

    @app.before_request
    def protect_origin():
        allowed_hosts = os.environ.get("COSMOS_ALLOWED_HOSTS", "192.168.1.117,127.0.0.1,localhost").split(",")
        if urlsplit("//" + request.host).hostname not in allowed_hosts:
            raise ServiceError("不接受此 Host", 403)
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and (urlsplit(origin).netloc != request.host or urlsplit(origin).scheme not in {"http", "https"}):
                raise ServiceError("不接受跨站请求", 403)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(service.status())

    @app.get("/v1/models")
    def models():
        return jsonify({"models": [service.status()]})

    def submit():
        fields = request.get_json() if request.is_json else request.form
        if not isinstance(fields, dict) and not hasattr(fields, "getlist"):
            raise ServiceError("请求必须是 JSON 对象或 multipart/form-data")
        for mapping in (fields, request.files):
            if hasattr(mapping, "getlist") and any(len(mapping.getlist(k)) != 1 for k in mapping):
                raise ServiceError("不允许重复字段")
        return service.submit(fields, request.files)

    @app.post("/api/jobs")
    def async_generate():
        job_id, _ = submit()
        return jsonify(service.get_job(job_id)), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        return jsonify(service.get_job(job_id))

    @app.post("/v1/infer")
    def infer():
        job_id, done = submit()
        if not done.wait(1700):
            return jsonify({"ok": False, "id": job_id, "error": "等待超时；可通过 /api/jobs/<id> 查询"}), 504
        result = service.get_job(job_id)
        status = result.pop("http_status", 200)
        return jsonify(result), status

    @app.errorhandler(ServiceError)
    def service_error(exc):
        return jsonify({"ok": False, "error": str(exc)}), exc.status

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify({"ok": False, "error": "上传内容不得超过 32 MiB" if exc.code == 413 else exc.description}), exc.code

    @app.errorhandler(Exception)
    def unexpected_error(exc):
        LOG.exception("HTTP handler failed")
        return jsonify({"ok": False, "error": "服务内部错误，请查看日志"}), 500

    if preload:
        threading.Thread(target=service.load, daemon=True).start()
    return app
