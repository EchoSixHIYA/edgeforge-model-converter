import json
import os
import shutil
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

    def sophon_environment():
        if app.config.get("SKIP_SOPHON_ENV_CHECK"):
            return {
                "ready": True, "tpu_mlir": True,
                "model_transform": True, "model_deploy": True,
                "sophon_demo": True,
            }
        model_transform = bool(shutil.which("model_transform.py"))
        model_deploy = bool(shutil.which("model_deploy.py"))
        sophon_demo = pipeline.sophon_demo.is_dir()
        return {
            "ready": bool(model_transform and model_deploy and sophon_demo),
            "tpu_mlir": bool(model_transform and model_deploy),
            "model_transform": model_transform,
            "model_deploy": model_deploy,
            "sophon_demo": sophon_demo,
        }

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
        sail = False
        try:
            import sophon.sail  # noqa: F401
            sail = True
        except ImportError:
            pass
        model_zoo = pipeline.model_zoo.is_dir()
        rockchip = {
            "ready": bool(toolkit and model_zoo),
            "rknn_toolkit2": toolkit,
            "model_zoo": model_zoo,
            "yolov10_exporter": pipeline.yolov10_repo.is_dir(),
        }
        sophon = sophon_environment()
        sophon["sophon_sail"] = sail
        return jsonify(status="ok", rockchip=rockchip, sophon=sophon,
                       rknn_toolkit2=toolkit, tpu_mlir=sophon["tpu_mlir"],
                       sophon_sail=sail, sophon_demo=sophon["sophon_demo"],
                       model_zoo=model_zoo, yolov10_exporter=rockchip["yolov10_exporter"])

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
        if fields["vendor"] == "sophon":
            env = sophon_environment()
            if not env["ready"]:
                missing = []
                if not env["model_transform"]:
                    missing.append("model_transform.py")
                if not env["model_deploy"]:
                    missing.append("model_deploy.py")
                if not env["sophon_demo"]:
                    missing.append("vendor/sophon-demo")
                return jsonify(error=(
                    "SOPHGO BModel 转换环境未就绪，暂不能创建转换任务；缺少: "
                    + ", ".join(missing)
                    + "。请先按 scripts/setup_sophon.sh / 官方 TPU-MLIR Docker 环境完成安装。"
                )), 400
        fields["backend"] = vendor["backend"]
        if fields["architecture"] not in BY_ID:
            return jsonify(error="未知模型架构"), 400
        model_cfg = BY_ID[fields["architecture"]]
        support = model_cfg.get("vendor_support", {}).get(fields["vendor"])
        if not support or not support.get("available"):
            blocker = support.get("blocker") if support else "该厂家尚未接入此模型"
            return jsonify(error=f"{model_cfg['name']} 尚未达到 {vendor['name']} 可转换标准：{blocker}"), 400
        if fields["backend"] not in {x["id"] for x in pipeline.backends.describe()}:
            return jsonify(error="未知输出后端"), 400
        if fields["platform"] not in PLATFORMS:
            return jsonify(error="未知目标芯片"), 400
        if fields["platform"] not in support["platforms"]:
            return jsonify(error=f"{model_cfg['name']} 在 {vendor['name']} 上支持的平台为: "
                           + ", ".join(x.upper() for x in support["platforms"])), 400
        if fields["precision"] not in support["precisions"]:
            return jsonify(error=(f"{model_cfg['name']} 官方支持的精度为: "
                                  + ", ".join(x.upper() for x in support["precisions"]))), 400
        model = request.files["model"]
        suffix = Path(model.filename or "").suffix.lower()
        weight_formats = support.get("weight_formats", model_cfg["weight_formats"])
        if suffix not in set(weight_formats):
            return jsonify(error="unsupported source weight format / 原始权重: " + ", ".join(weight_formats)), 400
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
