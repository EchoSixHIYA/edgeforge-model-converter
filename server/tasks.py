import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .pipeline import utc_now


class TaskManager:
    def __init__(self, data_dir, pipeline):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = pipeline
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="conversion")
        self.lock = threading.RLock()
        self.tasks = {}
        self._load()

    def _load(self):
        for path in self.data_dir.glob("*/task.json"):
            try:
                task = json.loads(path.read_text(encoding="utf-8"))
                if task["status"] in ("queued", "running", "canceling"):
                    task.update(status="failed", message="服务重启，任务已中断", finished_at=utc_now())
                    self._save(task)
                self.tasks[task["id"]] = task
            except Exception:
                continue

    def create(self, fields, model_file, calibration_file=None):
        task_id = uuid.uuid4().hex
        work = self.data_dir / task_id
        work.mkdir(parents=True)
        source = work / ("upload" + Path(model_file.filename).suffix.lower())
        model_file.save(source)
        calibration = None
        if calibration_file and calibration_file.filename:
            calibration = work / "calibration.zip"
            calibration_file.save(calibration)
        task = {
            "id": task_id, "status": "queued", "progress": 0, "message": "等待转换",
            "created_at": utc_now(), "started_at": None, "finished_at": None,
            "vendor": fields["vendor"], "architecture": fields["architecture"], "backend": fields["backend"],
            "platform": fields["platform"], "precision": fields["precision"],
            "original_filename": model_file.filename, "source_path": str(source),
            "calibration_path": str(calibration) if calibration else None,
            "work_dir": str(work), "artifacts": {}, "error": None,
        }
        with self.lock:
            self.tasks[task_id] = task
            self._save(task)
        self.executor.submit(self._run, task_id)
        return self.public(task)

    def create_vlm(self, fields, source, parameters, roots):
        task_id = uuid.uuid4().hex
        work = self.data_dir / task_id
        work.mkdir(parents=True)
        task = {
            "id": task_id, "status": "queued", "progress": 0, "message": "等待实验 VLM 转换",
            "created_at": utc_now(), "started_at": None, "finished_at": None,
            "vendor": "sophon", "architecture": "cosmos_reason2_2b", "backend": "bmodel",
            "platform": parameters["platform"], "precision": parameters["precision"],
            "original_filename": Path(source).name, "source_path": str(source),
            "input_kind": "hf_directory", "experimental": True, "parameters": parameters,
            "allowed_model_roots": [str(p) for p in roots], "calibration_path": None,
            "work_dir": str(work), "artifacts": {}, "error": None,
        }
        with self.lock:
            self.tasks[task_id] = task
            self._save(task)
        self.executor.submit(self._run, task_id)
        return self.public(task)

    def cancel(self, task_id):
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                raise KeyError(task_id)
            if task.get("input_kind") != "hf_directory":
                raise ValueError("现有 CV 后端尚未支持安全取消；此入口仅适用 VLM")
            if task["status"] in ("completed", "failed", "canceled"):
                raise ValueError("任务已经结束")
            (Path(task["work_dir"]) / "cancel.requested").touch()
            if task["status"] == "queued":
                self._update(task, status="canceled", message="排队任务已取消", finished_at=utc_now())
            else:
                self._update(task, status="canceling", message="正在取消并终止编译进程组")
            return self.public(task)

    def _run(self, task_id):
        from .vlm import ConversionCanceled
        with self.lock:
            task = self.tasks[task_id]
            if task["status"] == "canceled":
                return
            self._update(task, status="running", started_at=utc_now(), message="开始处理")
        try:
            artifacts = self.pipeline.run(task, lambda p, m: self._update(task, progress=p, message=m))
            with self.lock:
                if (Path(task["work_dir"]) / "cancel.requested").exists():
                    raise ConversionCanceled("转换已取消；未发布部分产物")
                self._update(task, status="completed", artifacts=artifacts, finished_at=utc_now())
        except ConversionCanceled as exc:
            self._update(task, status="canceled", message=str(exc), artifacts={}, finished_at=utc_now())
        except Exception as exc:
            self._update(task, status="failed", message=str(exc), error=traceback.format_exc(), finished_at=utc_now())

    def _update(self, task, **changes):
        with self.lock:
            task.update(changes)
            self._save(task)

    def _save(self, task):
        path = Path(task["work_dir"]) / "task.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def get(self, task_id):
        with self.lock:
            return self.public(self.tasks.get(task_id)) if task_id in self.tasks else None

    def list(self):
        with self.lock:
            return [self.public(t) for t in sorted(self.tasks.values(), key=lambda x: x["created_at"], reverse=True)]

    @staticmethod
    def public(task):
        if not task:
            return None
        result = {k: v for k, v in task.items() if k not in
                  {"source_path", "calibration_path", "work_dir", "error", "allowed_model_roots"}}
        if task.get("input_kind") == "hf_directory":
            from .backends.vlm_bmodel import log_tail
            result["log_tail"] = log_tail(Path(task["work_dir"]) / "conversion.log")
            result["artifacts"] = {key: True for key in task.get("artifacts", {})}
        return result
