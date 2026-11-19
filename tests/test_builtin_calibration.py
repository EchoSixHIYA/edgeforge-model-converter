import pytest

from server.calibration import prepare_builtin_calibration


def test_builtin_calibration_resolves_shared_samples_without_copying(tmp_path):
    zoo = tmp_path / "zoo"
    root = zoo / "datasets" / "COCO"
    subset = root / "subset"
    subset.mkdir(parents=True)
    names = [f"subset/{i:04d}.jpg" for i in range(20)]
    for name in names:
        (root / name).write_bytes(b"sample")
    manifest = root / "coco_subset_20.txt"
    manifest.write_text("\n".join(names), encoding="utf-8")
    work = tmp_path / "task"
    work.mkdir()
    dataset, count = prepare_builtin_calibration(zoo, work)
    assert count == 20
    assert dataset.read_text(encoding="utf-8").splitlines() == [str((root / name).resolve()) for name in names]
    assert sorted(path.name for path in work.iterdir()) == ["dataset.txt"]

    stray = zoo / "stray.jpg"
    stray.write_bytes(b"sample outside dataset")
    manifest.write_text("\n".join(names[:-1] + [str(stray.resolve())]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="内置校准数据集不完整"):
        prepare_builtin_calibration(zoo, work)
