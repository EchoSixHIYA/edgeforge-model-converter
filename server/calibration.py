"""Shared calibration samples shipped with the official RKNN Model Zoo."""
from pathlib import Path


def prepare_builtin_calibration(model_zoo, work):
    root = (Path(model_zoo) / "datasets" / "COCO").resolve()
    manifest = root / "coco_subset_20.txt"
    missing = "内置校准数据集不完整，请检查 RKNN Model Zoo 的 datasets/COCO/coco_subset_20.txt 和 subset 目录"
    if not manifest.is_file():
        raise RuntimeError(missing)
    images = []
    for name in manifest.read_text(encoding="utf-8").splitlines():
        name = name.strip()
        if not name:
            continue
        image = (root / name).resolve()
        if (root not in image.parents or not image.is_file() or not image.stat().st_size
                or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}):
            raise RuntimeError(missing)
        images.append(image)
    if len(images) != 20 or len(set(images)) != 20:
        raise RuntimeError(missing)
    dataset = Path(work).resolve() / "dataset.txt"
    dataset.write_text("\n".join(str(path) for path in images) + "\n", encoding="utf-8")
    return dataset, len(images)
