"""Rebuild completed SDK archives with portable POSIX ZIP entry paths."""
from pathlib import Path
import shutil


root = Path(__file__).resolve().parents[1] / "data" / "tasks"
for task_dir in root.iterdir():
    stage = task_dir / "sdk"
    archives = list(task_dir.glob("*_sdk.zip"))
    if not stage.is_dir() or not archives:
        continue
    archive = archives[0]
    temporary_base = task_dir / f"{archive.stem}.portable"
    temporary = Path(shutil.make_archive(str(temporary_base), "zip", stage))
    temporary.replace(archive)
    print(f"rebuilt {archive}")
