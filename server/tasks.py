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
                if task["status"] in ("queued", "running"):
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

    def _run(self, task_id):
        task = self.tasks[task_id]
        self._update(task, status="running", started_at=utc_now(), message="开始处理")
        try:
            artifacts = self.pipeline.run(task, lambda p, m: self._update(task, progress=p, message=m))
            self._update(task, status="completed", artifacts=artifacts, finished_at=utc_now())
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
        return {k: v for k, v in task.items() if k not in {"source_path", "calibration_path", "work_dir", "error"}}
