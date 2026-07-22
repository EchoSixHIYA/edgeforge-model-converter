import json
import os
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from .catalog import BY_ID, CATALOG, PLATFORMS, VENDORS, VENDOR_BY_ID
from .pipeline import Pipeline
from .tasks import TaskManager


def create_app(test_config=None):
    root = Path(__file__).resolve().parents[1]
    app = Flask(__name__, template_folder=str(root / "templates"), static_folder=str(root / "static"))
    app.config.update(MAX_CONTENT_LENGTH=4 * 1024 * 1024 * 1024, DATA_DIR=str(root / "data" / "tasks"))
    if test_config:
        app.config.update(test_config)
    pipeline = Pipeline(root)
    manager = TaskManager(app.config["DATA_DIR"], pipeline)
    app.extensions["task_manager"] = manager

    @app.get("/")
    def index():
        return render_template("index.html", catalog=CATALOG, platforms=PLATFORMS, vendors=VENDORS)

    @app.get("/health")
    def health():
        toolkit = False
        try:
            import rknn.api  # noqa: F401
            toolkit = True
        except ImportError:
            pass
        return jsonify(status="ok", rknn_toolkit2=toolkit,
                       model_zoo=pipeline.model_zoo.is_dir(), yolov10_exporter=pipeline.yolov10_repo.is_dir())

    @app.get("/api/catalog")
    def catalog():
        return jsonify(models=CATALOG, vendors=VENDORS, backends=pipeline.backends.describe(), platforms=PLATFORMS)

    @app.get("/docs/sdk-api")
    def sdk_api_docs():
        return send_file(root / "转换接入文档" / "SDK_API.md", mimetype="text/markdown; charset=utf-8")

    @app.get("/api/tasks")
    def tasks():
        return jsonify(manager.list())

    @app.post("/api/tasks")
    def create_task():
        required = ("vendor", "architecture", "platform", "precision")
        missing = [key for key in required if not request.form.get(key)]
        if missing or "model" not in request.files:
            return jsonify(error="缺少字段: " + ", ".join(missing or ["model"])), 400
        fields = {key: request.form[key].lower() for key in required}
        vendor = VENDOR_BY_ID.get(fields["vendor"])
        if not vendor or not vendor["available"]:
            return jsonify(error="该芯片厂家后端尚未启用"), 400
        fields["backend"] = vendor["backend"]
        if fields["architecture"] not in BY_ID:
            return jsonify(error="未知模型架构"), 400
        model_cfg = BY_ID[fields["architecture"]]
        if not model_cfg["available"]:
            return jsonify(error=f"{model_cfg['name']} 尚未达到可转换标准：{model_cfg['blocker']}"), 400
        if fields["backend"] not in {x["id"] for x in pipeline.backends.describe()}:
            return jsonify(error="未知输出后端"), 400
        if fields["platform"] not in PLATFORMS:
            return jsonify(error="未知目标芯片"), 400
        if fields["precision"] not in model_cfg["precisions"]:
            return jsonify(error=(f"{model_cfg['name']} 官方支持的精度为: "
                                  + ", ".join(x.upper() for x in model_cfg["precisions"]))), 400
        model = request.files["model"]
        suffix = Path(model.filename or "").suffix.lower()
        if suffix not in set(model_cfg["weight_formats"]):
            return jsonify(error="该架构只接受原始权重格式: " + ", ".join(model_cfg["weight_formats"])), 400
        calibration = request.files.get("calibration")
        if calibration and calibration.filename and Path(calibration.filename).suffix.lower() != ".zip":
            return jsonify(error="校准集必须是 ZIP"), 400
        try:
            return jsonify(manager.create(fields, model, calibration)), 202
        except Exception as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/api/tasks/<task_id>")
    def get_task(task_id):
        task = manager.get(task_id)
        return jsonify(task) if task else (jsonify(error="任务不存在"), 404)

    @app.get("/api/tasks/<task_id>/events")
    def task_events(task_id):
        if not manager.get(task_id):
            abort(404)
        def stream():
            last = None
            while True:
                task = manager.get(task_id)
                body = json.dumps(task, ensure_ascii=False)
                if body != last:
                    yield f"data: {body}\n\n"
                    last = body
                if task["status"] in ("completed", "failed"):
                    break
                time.sleep(0.75)
        return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/tasks/<task_id>/download/<kind>")
    def download(task_id, kind):
        task = manager.tasks.get(task_id)
        if not task or task["status"] != "completed" or kind not in {"model", "package"}:
            abort(404)
        path = Path(task["artifacts"][kind])
        if not path.is_file() or Path(task["work_dir"]).resolve() not in path.resolve().parents:
            abort(404)
        return send_file(path, as_attachment=True)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="上传文件超过 4GB 限制"), 413

    return app
