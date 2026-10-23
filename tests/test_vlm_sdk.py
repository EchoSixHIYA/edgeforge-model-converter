"""Offline SDK contracts. Fake device/processor fixtures are NOT real hardware tests."""
import json
import shutil
import subprocess
import sys
import threading
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from sdk_template.edgeforge_sdk import CosmosReason, Qwen3VL
from sdk_template.edgeforge_sdk import vlm
from server.vlm_sdk import package_vlm_sdk, validate_processor_assets

ROOT = Path(__file__).resolve().parents[1]


def model_config():
    return {"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl",
            "image_token_id": 151655, "video_token_id": 151656,
            "vision_start_token_id": 151652, "vision_end_token_id": 151653,
            "text_config": {"num_hidden_layers": 28, "hidden_size": 2048, "vocab_size": 151936,
                            "num_key_value_heads": 8, "head_dim": 128,
                            # Tiny base vocab is intentional for this synthetic tokenizer fixture.
                            "bos_token_id": 3, "eos_token_id": 151645},
            "vision_config": {"deepstack_visual_indexes": [5, 11, 17], "patch_size": 16,
                              "temporal_patch_size": 2, "spatial_merge_size": 2,
                              "num_position_embeddings": 2304, "in_channels": 3}}


def manifest():
    return {"runtime": "sophgo-llm-tpu-qwen3-vl", "lmhead_with_topk": False,
            "runtime_source": {"commit": vlm.RUNTIME_REVISION}, "sequence_length": 32,
            "max_input_length": 32, "max_pixels": 4096, "model_config": model_config()}


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "model"
    processor = root / "processor"
    processor.mkdir(parents=True)
    data = {"config.json": model_config(),
            "generation_config.json": {"eos_token_id": [151645, 151643], "do_sample": True,
                                       "temperature": .7, "top_p": .8, "top_k": 20, "repetition_penalty": 1.0},
            "tokenizer_config.json": {}, "preprocessor_config.json": {"do_resize": True},
            "video_preprocessor_config.json": {"do_resize": True},
            "chat_template.json": {"chat_template": "synthetic fixture"},
            "tokenizer.json": {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2},
                                         "merges": ["a b"]},
                               "added_tokens": [{"id": x} for x in [151645, 151643, 151655, 151656, 151652, 151653]]}}
    for name, value in data.items():
        (processor / name).write_text(json.dumps(value))
    (root / "model.bmodel").write_bytes(b"fixture-not-a-real-bmodel")
    (root / "config.json").write_text(json.dumps({"architecture": "cosmos_reason2_2b", "tokenizer_directory": "processor"}))
    (root / "compilation_manifest.json").write_text(json.dumps(manifest()))
    return root


class FakeTokenizer:
    def decode(self, tokens, **kwargs):
        return ":".join(str(x) for x in tokens)

    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 151655, "<|video_pad|>": 151656,
                "<|vision_start|>": 151652, "<|vision_end|>": 151653}[token]


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []
        self.ids = [1, 2, 3]
        self.video_processor = types.SimpleNamespace(size={"shortest_edge": 4096, "longest_edge": 25165824})
        self.output = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "formatted"

    def __call__(self, **kwargs):
        assert kwargs["return_tensors"] == "pt"  # Real fast processors reject np.
        self.calls.append(kwargs)
        return self.output or {"input_ids": np.array([self.ids])}


class FakeNative:
    SEQLEN = MAX_INPUT_LENGTH = 32
    MAX_PIXELS = 4096
    MAX_PATCHES = 16
    NUM_LAYERS = 28
    HIDDEN_SIZE = 2048
    VIT_DIMS = 1536
    num_deepstack = 3
    support_history = lmhead_with_topk = False

    def __init__(self):
        self.events = []
        self.tokens = [9, 10, 151643]
        self.history_length = 0

    def init(self, *args):
        self.events.append("init")

    def deinit(self):
        self.events.append("deinit")

    def clear_history(self):
        self.events.append("clear")
        self.history_length = 0
        self.output = iter(self.tokens)

    def configure_generation(self, *args):
        self.generation = args

    def set_seed(self, seed):
        self.seed = seed

    def forward_embed(self, ids):
        self.events.append("embed")
        self.history_length = len(ids)

    def forward_vit(self, *args):
        self.events.append(("vit", args[-1]))

    def forward_first(self, pos):
        self.events.append("prefill")
        self.prefill_positions = pos
        self.history_length += 1
        return next(self.output)

    def forward_next(self, pos):
        self.events.append("decode")
        self.history_length += 1
        return next(self.output)


@pytest.fixture
def runtime(bundle, monkeypatch):
    processor, native = FakeProcessor(), FakeNative()
    monkeypatch.setattr(vlm, "_load_processor", lambda _: processor)
    monkeypatch.setattr(vlm, "_load_native", lambda: native)
    model = CosmosReason(bundle)
    yield model, native, processor
    model.close()


def test_import_does_not_import_heavy_vlm_dependencies():
    script = "import sys; from sdk_template.edgeforge_sdk import CosmosReason; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; assert 'edgeforge_qwen3_vl' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def test_complete_assets_required(bundle):
    path = bundle / "processor"
    validate_processor_assets(path)
    (path / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="Incomplete tokenizer"):
        validate_processor_assets(path)


def test_partial_tokenizer_rejected(bundle):
    path = bundle / "processor" / "tokenizer.json"
    data = json.loads(path.read_text())
    del data["model"]["vocab"]["b"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="truncated"):
        validate_processor_assets(path.parent)


def test_vocab_merges_alternative(bundle):
    path = bundle / "processor"
    (path / "tokenizer.json").unlink()
    (path / "vocab.json").write_text(json.dumps({str(x): x for x in [0, 1, 2, 151645, 151643, 151655, 151656, 151652, 151653]}))
    (path / "merges.txt").write_text("#version: 0.2\na b\n")
    validate_processor_assets(path)


def test_sdk_archive_includes_assets_and_one_model(bundle, tmp_path):
    task = {"architecture": "cosmos_reason2_2b", "platform": "bm1684x", "precision": "w4bf16"}
    archive = package_vlm_sdk(ROOT, task, bundle / "processor", bundle / "model.bmodel", manifest(), tmp_path)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert [name for name in names if name.endswith(".bmodel")] == ["model/model.bmodel"]
        for required in ["model/processor/tokenizer.json", "model/processor/chat_template.json",
                         "model/processor/generation_config.json", "model/compilation_manifest.json",
                         "runtime/qwen3_vl/chat.cpp", "runtime/qwen3_vl/sampling.h", "runtime/qwen3_vl/LICENSE",
                         "runtime/qwen3_vl/build.sh", "VLM_README.md", "example.py"]:
            assert required in names
        assert not any("rknn-toolkit-lite2" in name for name in names)
        assert not any("__pycache__" in name for name in names)
        info = json.loads(zf.read("manifest.json"))
        assert info["sdk"]["transformers"] == "4.57.1"
        assert info["validation_status"]["hardware_inference_verified"] is False
        assert "model.bmodel" in info["asset_sha256"]
        assert json.loads(zf.read("model/processor/config.json")) == model_config()


@pytest.mark.parametrize("key,value", [("lmhead_with_topk", True), ("sequence_length", 0), ("runtime", "sail")])
def test_package_rejects_wrong_manifest(bundle, tmp_path, key, value):
    data = manifest(); data[key] = value
    with pytest.raises(ValueError):
        package_vlm_sdk(ROOT, {}, bundle / "processor", bundle / "model.bmodel", data, tmp_path)


def test_generation_preserves_sampling_and_both_eos(runtime):
    model, native, _ = runtime
    for eos in [151643, 151645]:
        native.tokens = [9, eos]
        result = model.generate("hello", generation={"seed": 12})
        assert result.token_ids == (9,)
        assert result.finish_reason == "eos"
        assert native.generation == (True, .7, .8, 20, 1.0)
        assert native.seed == 12
    assert native.events.count("clear") == 2


def test_first_eos_no_decode(runtime):
    model, native, _ = runtime
    native.tokens = [151643]
    assert model.predict("hello").token_ids == ()
    assert "decode" not in native.events


def test_max_new_tokens_no_extra_decode(runtime):
    model, native, _ = runtime
    result = model.generate("hello", max_new_tokens=1)
    assert result.token_ids == (9,) and result.finish_reason == "length"
    assert "decode" not in native.events


def test_sequence_capacity_stops_before_native_overflow(runtime):
    model, native, processor = runtime
    processor.ids = list(range(31))
    result = model.generate("hello", max_new_tokens=10)
    assert result.token_ids == (9,) and "decode" not in native.events
    processor.ids = list(range(32))
    before = len(native.events)
    with pytest.raises(ValueError, match="compiled prefill"):
        model.generate("hello")
    assert len(native.events) == before


@pytest.mark.parametrize("ids", [[], [-1, 2], [151936]])
def test_bad_token_ids_rejected_before_native(runtime, ids):
    model, native, processor = runtime
    processor.ids = ids
    with pytest.raises(ValueError, match="token IDs"):
        model.generate("hello")
    assert "embed" not in native.events


@pytest.mark.parametrize("options", [{"top_p": 0}, {"temperature": float("nan")}, {"top_k": -1},
                                      {"repetition_penalty": 0}, {"do_sample": "yes"}, {"seed": -1},
                                      {"num_beams": 2}])
def test_generation_invalid_options(runtime, options):
    with pytest.raises(ValueError):
        runtime[0].generate("hello", generation=options)


def test_original_resize_settings_not_overridden(runtime, tmp_path, monkeypatch):
    # Do not require Pillow in the server/CV test environment.
    class FakeImage:
        size = (32, 32)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def convert(self, mode): return self
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=types.SimpleNamespace(open=lambda _: FakeImage())))
    model, native, processor = runtime
    path = tmp_path / "frame.png"; path.write_bytes(b"fake-image-for-unit-test")
    processor.output = {"input_ids": [[1, 151652, 151655, 151653, 2]], "image_grid_thw": [[1, 2, 2]],
                        "pixel_values": np.zeros((4, 1536))}
    model.generate("hello", image=path, max_new_tokens=1)
    assert "do_resize" not in processor.calls[-1]
    assert "images_kwargs" not in processor.calls[-1]
    assert native.events.index(("vit", 2)) < native.events.index("prefill")
    model.preprocess("hello", image=path, min_pixels=1024, max_pixels=4096)
    assert processor.calls[-1]["images_kwargs"] == {"min_pixels": 1024, "max_pixels": 4096}


def test_video_temporal_groups_and_timestamp_metadata(runtime, monkeypatch, tmp_path):
    model, native, processor = runtime
    path = tmp_path / "clip.mp4"; path.write_bytes(b"fake")
    audit = {"fps": 30, "frames_indices": [0, 15, 60], "timestamps": [0., .5, 2.],
             "temporal_timestamps": [.25, 2.], "padded_last_frame": True}
    monkeypatch.setattr(vlm, "_decode_video", lambda *a, **kw: (np.zeros((3, 32, 32, 3)), object(), dict(audit)))
    processor.output = {"input_ids": [[1, 151652, 151656, 151653, 20, 151652, 151656, 151653, 2]],
                        "video_grid_thw": [[2, 2, 2]], "pixel_values_videos": np.zeros((8, 1536))}
    result = model.generate("hello", video=path, num_frames=3, max_new_tokens=1)
    assert [("vit", 2), ("vit", 6)] == [x for x in native.events if isinstance(x, tuple)]
    assert result.media_metadata["frames_indices"] == [0, 15, 60]
    assert processor.calls[-1]["videos_kwargs"]["do_sample_frames"] is False
    assert "do_resize" not in processor.calls[-1]["videos_kwargs"]
    assert processor._edgeforge_frame_times is None


def test_patch_order_interpolation_and_mrope():
    positions, indices, weights = vlm.vision_positions(4, 4)
    assert positions[:8].tolist() == [[0,0], [0,1], [1,0], [1,1], [0,2], [0,3], [1,2], [1,3]]
    assert indices.shape == weights.shape == (16, 4)
    assert np.allclose(weights.sum(axis=1), 1)
    assert indices.min() == 0 and indices.max() == 2303
    result = vlm.rope_positions(np.arange(10), [(2, 2, 2)])
    assert result[:, :2].tolist() == [[0,1]] * 3
    assert result[:, 2:6].tolist() == [[2,2,2,2], [2,2,3,3], [2,3,2,3]]
    assert result[:, 6:].tolist() == [[4,5,6,7]] * 3


def test_actual_timestamps_odd_padding_and_vfr():
    assert vlm.temporal_timestamps([0, 10, 40], 20) == [.25, 2.]
    assert vlm.temporal_timestamps([0, 10, 40], 20, timestamps=[0, .6, 2.3]) == [.3, 2.3]
    with pytest.raises(ValueError):
        vlm.temporal_timestamps([0], 0)


def test_close_is_idempotent_and_preprocess_closed(runtime):
    model, native, _ = runtime
    model.close(); model.close()
    assert native.events.count("deinit") == 1
    with pytest.raises(RuntimeError, match="closed"):
        model.preprocess("hello")


def test_wrong_native_abi_dependency_error(monkeypatch):
    monkeypatch.setattr(vlm.importlib, "import_module", lambda _: types.SimpleNamespace(EDGEFORGE_ABI=0))
    with pytest.raises(RuntimeError, match="Incompatible"):
        vlm._load_native()


def test_native_missing_actionable_error(monkeypatch):
    def missing(_):
        raise ImportError("not available")
    monkeypatch.setattr(vlm.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="SAIL alone is insufficient"):
        vlm._load_native()


def test_native_sampling_cpp(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable; host sampler not tested")
    source = tmp_path / "sample_test.cpp"
    source.write_text(r'''
#include "sampling.h"
#include <cassert>
int main() {
  std::mt19937 rng(42), other(42);
  std::vector<int> history = {0, 0};
  assert(edgeforge_sample({4, -2, 3}, history, 2, false, .7, .8, 20, 2, rng) == 2);
  assert(edgeforge_sample({-1, -1.5}, history, 2, false, 1, 1, 0, 2, rng) == 1);
  assert(edgeforge_sample({0, 5, 1}, {}, 0, true, .7, .8, 1, 1, rng) == 1);
  assert(edgeforge_sample({0, 5, 1}, {}, 0, true, 1, .0001, 0, 1, rng) == 1);
  // Top-p 0.5 keeps only the first token at an exact 0.5 mass boundary.
  for (int i=0; i<20; ++i) assert(edgeforge_sample({0,0}, {},0,true,1,.5,0,1,rng)==0);
  rng.seed(123); other.seed(123);
  for (int i=0; i<100; ++i)
    assert(edgeforge_sample({1,2,3}, {},0,true,.7,.95,0,1,rng) ==
           edgeforge_sample({1,2,3}, {},0,true,.7,.95,0,1,other));
  bool threw = false;
  try { edgeforge_sample({NAN}, {},0,true,1,1,0,1,rng); } catch (...) { threw = true; }
  assert(threw); threw=false;
  try { edgeforge_sample({1}, {},0,true,0,1,0,1,rng); } catch (...) { threw = true; }
  assert(threw);
  assert(edgeforge_fp16(0x3c00) == 1.0f);
  assert(edgeforge_fp16(0xc000) == -2.0f);
  assert(edgeforge_bf16(0x3f80) == 1.0f);
  assert(std::isinf(edgeforge_fp16(0x7c00)));
}
''')
    binary = tmp_path / "sample_test"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "sdk_template/runtime/qwen3_vl"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_video_decode_preserves_source_indices_fps_and_pts(monkeypatch, tmp_path):
    # Real decoder contract exercised through fake PyAV frames, no PyAV installed.
    class Metadata:
        def __init__(self, **kw): self.__dict__.update(kw)
    class Frame:
        def __init__(self, index): self.index, self.time = index, [2., 2.04, 2.11, 2.13, 2.20][index]
        def to_ndarray(self, format): return np.full((32, 32, 3), self.index, dtype=np.uint8)
    class Container:
        streams = types.SimpleNamespace(video=[types.SimpleNamespace(average_rate=25, width=32, height=32)])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def decode(self, **kw): return iter(Frame(i) for i in range(5))
    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(open=lambda _: Container()))
    monkeypatch.setitem(sys.modules, "transformers.video_utils", types.SimpleNamespace(VideoMetadata=Metadata))
    calls = []
    def sample(metadata, num_frames=None, fps=None):
        calls.append((metadata.fps, num_frames, fps))
        assert isinstance(num_frames, int)
        return np.linspace(0, metadata.total_num_frames - 1, num_frames).round().astype(int)
    processor = types.SimpleNamespace(sample_frames=sample)
    frames, metadata, audit = vlm._decode_video(tmp_path / "fake.mp4", processor, num_frames=3.0)
    assert frames[:, 0, 0, 0].tolist() == [0, 2, 4]
    assert metadata.frames_indices == [0, 2, 4]
    assert metadata.fps == 25 and calls == [(25., 3, None)]
    assert np.allclose(audit["timestamps"], [0, .11, .2])
    assert np.allclose(audit["temporal_timestamps"], [.055, .2])
    assert audit["padded_last_frame"] is True


def test_video_no_guessed_fps(monkeypatch, tmp_path):
    class Container:
        streams = types.SimpleNamespace(video=[types.SimpleNamespace(average_rate=None)])
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(open=lambda _: Container()))
    monkeypatch.setitem(sys.modules, "transformers.video_utils", types.SimpleNamespace(VideoMetadata=object))
    with pytest.raises(ValueError, match="no reliable FPS"):
        vlm._decode_video(tmp_path / "fake.mp4", object())


def test_runtime_rejects_wrong_revision_before_dependency_load(bundle, monkeypatch):
    data = manifest(); data["runtime_source"]["commit"] = "wrong"
    (bundle / "compilation_manifest.json").write_text(json.dumps(data))
    monkeypatch.setattr(vlm, "_load_processor", lambda _: pytest.fail("must fail before loading dependencies"))
    with pytest.raises(ValueError, match="revision"):
        CosmosReason(bundle)


def test_runtime_shape_mismatch_closes_native(bundle, monkeypatch):
    native = FakeNative(); native.MAX_PIXELS = 8192
    monkeypatch.setattr(vlm, "_load_processor", lambda _: FakeProcessor())
    monkeypatch.setattr(vlm, "_load_native", lambda: native)
    with pytest.raises(ValueError, match="MAX_PIXELS"):
        CosmosReason(bundle)
    assert native.events == ["init", "deinit"]


def test_overbudget_visual_grid_rejected_before_device(runtime):
    model, native, _ = runtime
    inputs = {"image_grid_thw": [[1, 6, 6]], "pixel_values": np.zeros((36, 1536))}
    with pytest.raises(ValueError, match="compiled maximum"):
        model._visual_inputs(inputs, np.array([1]), True, False)
    assert "embed" not in native.events


def test_required_shape_validation_precedes_native_reads():
    body = (ROOT / "sdk_template/runtime/qwen3_vl/chat.cpp").read_text()
    assert 'require_net("vit", 5, 4)' in body
    assert '{patches, 4, 1}, "ViT interpolation weights"' in body
    assert '{1, 1, patches, patches}, "ViT attention mask"' in body
    assert '{patches / 4, 2048}, "ViT/DeepStack output"' in body
    assert 'shape(net_lm->stages[0].output_shapes[0], {1, 151936}' in body
    assert 'catch (...) { deinit(); throw; }' in body


def test_async_native_launch_is_fenced_before_dependent_reads():
    body = (ROOT / "sdk_template/runtime/qwen3_vl/chat.cpp").read_text()
    launch = body.split("void Qwen3_VL::net_launch(", 1)[1].split("void Qwen3_VL::net_launch_decode", 1)[0]
    assert launch.index("bmrt_launch_tensor_ex") < launch.index("bm_thread_sync(bm_handle)")


def test_processor_tensors_are_converted_at_native_boundary(runtime):
    model, _, processor = runtime
    class Tensor:
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.array([[1, 2, 3]], dtype=np.int64)
    processor.output = {"input_ids": Tensor()}
    prepared = model.preprocess("hello")
    assert processor.calls[-1]["return_tensors"] == "pt"
    assert prepared.input_ids.dtype == np.int32
    assert prepared.input_ids.flags.c_contiguous
