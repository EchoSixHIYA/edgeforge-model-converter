from pathlib import Path

import pytest

from server.exporters import ArchitectureExporters


@pytest.mark.parametrize("architecture", ["yolov8", "yolo11"])
def test_ultralytics_rockchip_export_uses_rknn_friendly_branch(tmp_path, architecture):
    repo = tmp_path / "vendor" / "exporters" / ("yolov8" if architecture == "yolov8" else "yolo11")
    repo.mkdir(parents=True)
    source = tmp_path / "input.pt"
    source.write_bytes(b"weights")
    work = tmp_path / "work"
    work.mkdir()
    exporter = ArchitectureExporters(tmp_path)
    calls = []

    def fake_run(command, cwd):
        calls.append((command, cwd))
        source.with_suffix(".onnx").write_bytes(b"friendly onnx")

    exporter._run = fake_run
    output = exporter.export(source, work, architecture, "rockchip", lambda *_: None)

    command, cwd = calls[0]
    assert cwd == repo
    assert command[command.index("--format") + 1] == "rknn"
    assert command[command.index("--opset") + 1] == "12"
    assert output.read_bytes() == b"friendly onnx"


def test_acceptance_status_is_explicit_in_public_catalog(tmp_path):
    from server.catalog import BY_ID

    assert BY_ID["yolov8"]["postprocess_ready"] is True
    assert isinstance(BY_ID["yolov8"]["verified"], bool)


def test_mobilenet_accepts_only_pth_training_checkpoint(tmp_path):
    exporter = ArchitectureExporters(tmp_path)
    assert exporter.supports("mobilenet", ".pth", "rockchip")
    assert not exporter.supports("mobilenet", ".onnx", "rockchip")
    assert not exporter.supports("mobilenet", ".pt", "rockchip")
    assert exporter.supports("resnet", ".pth", "rockchip")
    assert exporter.supports("yolov6", ".pt", "rockchip")
    assert exporter.supports("yolov7", ".pt", "rockchip")
    assert exporter.supports("yolox", ".pth", "rockchip")
    assert exporter.supports("retinaface", ".pth", "rockchip")
    assert not exporter.supports("retinaface", ".onnx", "rockchip")
    assert exporter.supports("ppyoloe", ".pdparams", "rockchip")
    assert not exporter.supports("ppyoloe", ".onnx", "rockchip")
    assert exporter.supports("lprnet", ".pth", "rockchip")
    assert not exporter.supports("lprnet", ".onnx", "rockchip")
    assert exporter.supports("yamnet", ".zip", "rockchip")
    assert not exporter.supports("yamnet", ".onnx", "rockchip")
    assert exporter.supports("ppseg", ".pdparams", "rockchip")
    assert not exporter.supports("ppseg", ".onnx", "rockchip")
    assert exporter.supports("deeplabv3", ".pb", "rockchip")
    assert not exporter.supports("deeplabv3", ".onnx", "rockchip")
    assert exporter.supports("mobilesam", ".pt", "rockchip")
    assert not exporter.supports("mobilesam", ".onnx", "rockchip")
    assert exporter.supports("ppocr", ".zip", "rockchip")
    assert not exporter.supports("ppocr", ".onnx", "rockchip")


def test_retinaface_decodes_box_and_five_landmarks():
    import numpy as np
    from sdk_template.edgeforge_sdk.model import _postprocess_retinaface

    locations = np.zeros((1, 4200, 4), dtype=np.float32)
    probabilities = np.zeros((1, 4200, 2), dtype=np.float32)
    landmarks = np.zeros((1, 4200, 10), dtype=np.float32)
    probabilities[0, 0, 1] = .99

    boxes, scores, points = _postprocess_retinaface(
        [locations, probabilities, landmarks], (320, 320), .5, .4)

    assert boxes.shape == (1, 4)
    assert scores.tolist() == pytest.approx([.99])
    assert points.shape == (1, 5, 2)


def test_ppyoloe_decodes_its_nine_output_contract():
    import numpy as np
    from sdk_template.edgeforge_sdk.model import _postprocess_dfl

    outputs = []
    for branch, size in enumerate((20, 40, 80)):
        position = np.zeros((1, 68, size, size), dtype=np.float32)
        probabilities = np.zeros((1, 80, size, size), dtype=np.float32)
        if branch == 0:
            probabilities[0, 5, 0, 0] = .9
        score_sum = probabilities.sum(axis=1, keepdims=True).clip(0, 1)
        outputs.extend((position, probabilities, score_sum))

    boxes, classes, scores = _postprocess_dfl(
        outputs, (640, 640), 80, .25, .45, nms=True)

    assert boxes.shape == (1, 4)
    assert classes.tolist() == [5]
    assert scores.tolist() == pytest.approx([.9])


def test_lprnet_collapses_repeats_and_blank_tokens():
    import numpy as np
    from sdk_template.edgeforge_sdk.model import _postprocess_lprnet

    logits = np.full((1, 68, 8), -10.0, dtype=np.float32)
    # 京, 京, blank, A, A, blank, 1, blank -> 京A1
    for timestep, token in enumerate((0, 0, 67, 41, 41, 67, 32, 67)):
        logits[0, token, timestep] = 10.0

    results = _postprocess_lprnet([logits])

    assert len(results) == 1
    assert results[0].text == "京A1"
    assert results[0].token_ids == (0, 41, 32)
    assert results[0].score > .99


def test_yamnet_resamples_audio_and_averages_frame_scores():
    import numpy as np
    from sdk_template.edgeforge_sdk.model import _postprocess_yamnet, _preprocess_audio

    waveform = np.linspace(-1, 1, 8000, dtype=np.float32)
    tensor, context = _preprocess_audio((waveform, 8000), 16000, 48000)
    assert tensor.shape == (1, 48000)
    assert tensor.dtype == np.float32
    assert context["source_sample_rate"] == 8000
    assert tensor[0, 15999] == pytest.approx(1.0, abs=2e-4)
    assert np.count_nonzero(tensor[0, 16000:]) == 0

    labels = ["Silence", "Animal", "Music"]
    scores = np.asarray([[[.1, .8, .1], [.2, .6, .2]]], dtype=np.float32)
    results = _postprocess_yamnet(
        [np.zeros((1, 2, 1024), dtype=np.float32),
         np.zeros((1, 2, 64), dtype=np.float32), scores], labels, 2)
    assert [item.label for item in results] == ["Animal", "Music"]
    assert results[0].score == pytest.approx(.7)
    assert results[0].rank == 1


def test_ppseg_returns_original_size_semantic_mask():
    import numpy as np
    from sdk_template.edgeforge_sdk.model import _postprocess_semantic_mask

    logits = np.zeros((1, 3, 4, 5), dtype=np.float32)
    logits[:, 1, :2] = 2.0
    logits[:, 2, 2:] = 3.0
    result = _postprocess_semantic_mask(
        [logits], ["road", "person", "car"], (8, 10))

    assert result.mask.shape == (8, 10)
    assert result.class_ids == (1, 2)
    assert result.as_dict()["classes"][0]["label"] == "person"


def test_mobilesam_restores_best_prompt_mask_to_original_image(tmp_path):
    import json
    import numpy as np
    from sdk_template.edgeforge_sdk import InteractiveMask, MobileSAM

    (tmp_path / "encoder.rknn").write_bytes(b"encoder")
    (tmp_path / "decoder.rknn").write_bytes(b"decoder")
    (tmp_path / "config.json").write_text(json.dumps({
        "architecture": "mobilesam", "modality": "image+prompt",
        "task_type": "interactive_segmentation", "input_size": [448, 448],
        "model_files": {"encoder": "encoder.rknn", "decoder": "decoder.rknn"},
    }), encoding="utf-8")
    model = MobileSAM(tmp_path)
    tensor, context = model.preprocess(np.zeros((300, 600, 3), dtype=np.uint8))
    assert tensor.shape == (1, 448, 448, 3)
    assert context["input_shape"] == (224, 448)
    context.update({"point_coords": np.asarray([[10, 20], [100, 200]]),
                    "point_labels": np.asarray([2, 3])})
    scores = np.asarray([[.1, .9, .2, .3]], dtype=np.float32)
    masks = np.zeros((1, 4, 112, 112), dtype=np.float32)
    masks[0, 1, 10:80, 10:80] = 1.0
    result = model.postprocess([scores, masks], context)
    assert isinstance(result, InteractiveMask)
    assert result.mask.shape == (300, 600)
    assert result.mask.any()
    assert result.score == pytest.approx(.9)


def test_ppocr_ctc_removes_blank_and_repeated_tokens():
    import numpy as np
    from sdk_template.edgeforge_sdk.model import _ppocr_ctc

    probabilities = np.zeros((1, 7, 4), dtype=np.float32)
    for timestep, token in enumerate((0, 1, 1, 0, 2, 3, 0)):
        probabilities[0, timestep, token] = .9
    text, score = _ppocr_ctc(probabilities, ["纯", "臻", " "])
    assert text == "纯臻 "
    assert score == pytest.approx(.9)


def test_yolov8_pose_returns_seventeen_keypoints():
    import numpy as np
    from sdk_template.edgeforge_sdk.model import _postprocess_yolov8_pose

    outputs = [np.zeros((1, 65, size, size), dtype=np.float32) for size in (80, 40, 20)]
    # One strong candidate in the first branch.  Position logits of zero decode
    # to a deterministic DFL box and the final channel is a confidence logit.
    for output in outputs:
        output[:, 64] = -20.0
    outputs[0][0, 64, 0, 0] = 10.0
    keypoints = np.zeros((1, 17, 3, 8400), dtype=np.float32)
    keypoints[0, :, 0, 0] = np.arange(17)
    keypoints[0, :, 1, 0] = np.arange(17) + 20
    keypoints[0, :, 2, 0] = .9
    boxes, scores, points = _postprocess_yolov8_pose(
        outputs + [keypoints], (640, 640), .25, .45)

    assert boxes.shape == (1, 4)
    assert .99 < scores[0] <= 1.0
    assert points.shape == (1, 17, 3)
    assert points[0, 16].tolist() == pytest.approx([16, 36, .9])


def test_yolov8_obb_decodes_angle_and_uses_rotated_nms():
    import numpy as np
    from sdk_template.edgeforge_sdk.model import _postprocess_yolov8_obb

    outputs = [np.zeros((1, 79, size, size), dtype=np.float32) for size in (80, 40, 20)]
    for output in outputs:
        output[:, 64:] = -20.0
    outputs[0][0, 65, 0, 0] = 10.0
    angles = np.full((1, 1, 8400), .25, dtype=np.float32)
    rects, classes, scores = _postprocess_yolov8_obb(
        outputs + [angles], (640, 640), 15, .5, .4)

    assert rects.shape == (1, 5)
    assert rects[0, 4] == pytest.approx(0.0)
    assert classes.tolist() == [1]
    assert .99 < scores[0] <= 1.0


def test_yolov8_seg_returns_binary_instance_mask():
    import numpy as np
    from sdk_template.edgeforge_sdk.model import _postprocess_yolov8_seg

    outputs = []
    for branch, size in enumerate((80, 40, 20)):
        position = np.zeros((1, 64, size, size), dtype=np.float32)
        scores = np.zeros((1, 80, size, size), dtype=np.float32)
        if branch == 0:
            scores[0, 0, 0, 0] = .9
        auxiliary = np.ones((1, 1, size, size), dtype=np.float32)
        coefficients = np.zeros((1, 32, size, size), dtype=np.float32)
        if branch == 0:
            coefficients[0, 0, 0, 0] = 2.0
        outputs.extend((position, scores, auxiliary, coefficients))
    proto = np.zeros((1, 32, 160, 160), dtype=np.float32)
    proto[:, 0] = 1.0
    boxes, classes, scores, masks = _postprocess_yolov8_seg(
        outputs + [proto], (640, 640), 80, .25, .45)

    assert boxes.shape == (1, 4)
    assert classes.tolist() == [0]
    assert masks.shape == (1, 640, 640)
    assert masks.dtype == np.bool_
    assert masks[0].any()
