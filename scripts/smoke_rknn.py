"""Build a tiny ONNX model through the complete RKNN + SDK pipeline."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.pipeline import Pipeline


def main():
    root = ROOT
    work = Path(tempfile.mkdtemp(prefix="edgeforge-smoke-"))
    source = work / "source.onnx"
    weights = numpy_helper.from_array(np.ones((4, 3, 3, 3), dtype=np.float32) / 27, "weights")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["images", "weights"], ["conv"], pads=[1, 1, 1, 1]),
         helper.make_node("Relu", ["conv"], ["output"])],
        "edgeforge_smoke",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 32, 32])],
        [weights],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), source)
    task = {
        "vendor": "rockchip", "architecture": "mobilenet", "backend": "rknn", "platform": "rk3588", "precision": "fp", "internal_test_onnx": True,
        "source_path": str(source), "calibration_path": None, "work_dir": str(work),
    }
    artifacts = Pipeline(root).run(task, lambda p, m: print(f"{p:3d}% {m}", flush=True))
    print(json.dumps({k: {"path": v, "bytes": Path(v).stat().st_size} for k, v in artifacts.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
