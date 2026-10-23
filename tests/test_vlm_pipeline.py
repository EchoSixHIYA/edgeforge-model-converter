"""No weights/toolchain/board required. Mocks below NEVER validate a compilation."""
import io
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

import pytest

from server import vlm
from server.backends.vlm_bmodel import VlmBModelBackend
from server.pipeline import Pipeline
from server.tasks import TaskManager
from server.vlm_worker import apply_controls, validate_compiler_tools, validate_host_runtime
from server.web import create_app

ROOT = Path(__file__).resolve().parents[1]
HEADER = {"X-EdgeForge-Local-Import": "1"}


@pytest.mark.parametrize("missing_tool", ["cmake", "make", "gcc", "g++"])
def test_preflight_detects_missing_ppl_jit_build_tools(monkeypatch, missing_tool):
    monkeypatch.setattr("server.vlm_worker.shutil.which",
                        lambda name: None if name == missing_tool else "/mock/bin/" + name)
    with pytest.raises(RuntimeError) as error:
        validate_compiler_tools()
    assert missing_tool in str(error.value)


def test_compiler_rejects_old_glibc_before_native_tools_crash(monkeypatch):
    monkeypatch.setattr("server.vlm_worker.sys.platform", "linux")
    monkeypatch.setattr("server.vlm_worker.platform.libc_ver", lambda: ("glibc", "2.31"))
    with pytest.raises(RuntimeError, match="Ubuntu 22.04"):
        validate_host_runtime()


def test_compiler_accepts_supported_glibc(monkeypatch):
    monkeypatch.setattr("server.vlm_worker.sys.platform", "linux")
    monkeypatch.setattr("server.vlm_worker.platform.libc_ver", lambda: ("glibc", "2.35"))
    validate_host_runtime()


def test_vlm_schema_and_manifest_are_explicit():
    params = vlm.parameters({})
    assert params == {"platform": "bm1684x", "precision": "w4bf16", "seq_length": 2048,
                      "max_pixels": 147456, "jobs": 1}
    manifest = vlm.compilation_manifest(params, {"model_config": {}, "config_sha256": "hash"})
    assert manifest["visual"]["max_visual_tokens"] == 144
    assert manifest["graph_contract"]["vit_output_shapes"] == [[144, 2048]] * 4
    assert manifest["graph_contract"]["vit_input_shapes"][-1] == [1, 1, 576, 576]
    assert manifest["lmhead_with_topk"] is False
    assert manifest["compile"]["do_sample"] is True
    assert "block_cache_27" in manifest["graph_contract"]["required_graphs"]
    assert manifest["verified"] is False


@pytest.mark.parametrize("bad", [
    {"platform": "bm1684"}, {"precision": "auto"}, {"precision": "f16"}, {"jobs": 4},
    {"jobs": True}, {"seq_length": 255}, {"seq_length": 2049}, {"seq_length": "2048"},
    {"max_pixels": 147457}, {"max_pixels": 0}, {"max_pixels": 1048576, "seq_length": 256},
    {"only_mlir": True}, {"dry_run": True}, {"architecture": "qwen_image"},
])
def test_vlm_rejects_invalid_or_nonexistent_upstream_options(bad):
    with pytest.raises(ValueError):
        vlm.parameters(bad)


def test_directory_allowlist_and_no_pickle_or_links(tmp_path):
    root = tmp_path / "allowed"
    model = root / "hf"
    model.mkdir(parents=True)
    assert vlm.resolve_model_dir(str(model), [root.resolve()]) == model
    with pytest.raises(ValueError):
        vlm.resolve_model_dir(str(model), [])
    with pytest.raises(ValueError):
        vlm.resolve_model_dir(str(tmp_path), [root])
    with pytest.raises(ValueError):
        vlm.resolve_model_dir("../hf", [root])
    link = model / "evil.json"
    link.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="符号链接"):
        vlm.resolve_model_dir(str(model), [root])
    link.unlink()
    (model / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(ValueError, match="pickle"):
        vlm.resolve_model_dir(str(model), [root])


def test_weights_absent_or_shard_traversal_fail_before_compile(tmp_path):
    with pytest.raises(ValueError, match="真实"):
        vlm.validate_weights(tmp_path)
    (tmp_path / "one.safetensors").write_bytes(b"invalid")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../escape.safetensors"}}))
    with pytest.raises(ValueError, match="不安全"):
        vlm.validate_weights(tmp_path)


def test_safetensors_header_is_bounded_and_checks_offsets(tmp_path):
    path = tmp_path / "weight.safetensors"
    header = {"x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"0000")
    assert vlm.safetensors_headers([path])["x"]["shape"] == [2]
    path.write_bytes(struct.pack("<Q", 2**63))
    with pytest.raises(ValueError, match="头部长度"):
        vlm.safetensors_headers([path])
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"0")
    with pytest.raises(ValueError, match="数据区"):
        vlm.safetensors_headers([path])


def test_official_serial_adapter_controls_export_and_compiler_not_omp_only():
    events = []
    class OfficialLike:
        def __init__(self):
            self.max_workers = 99
            self.debug = False
        def gen_all_mlir(self):
            events.append(("export_serial", self.debug))
        def compile_all(self):
            events.append(("parallel_jobs", self.max_workers))
    apply_controls(OfficialLike, 1)
    converter = OfficialLike()
    assert converter.max_workers == 1
    converter.gen_all_mlir()
    assert converter.debug is False
    converter.compile_all()
    assert events == [("export_serial", True), ("parallel_jobs", 1)]


def test_pipeline_hf_path_never_calls_single_onnx(monkeypatch, tmp_path):
    pipeline = Pipeline(ROOT)
    monkeypatch.setattr(pipeline, "_run_vlm", lambda task, update: {"mock_routing_only": True})
    monkeypatch.setattr(pipeline, "_to_onnx", lambda *a: pytest.fail("CV ONNX route must not be used"))
    assert pipeline.run({"input_kind": "hf_directory"}, lambda *a: None) == {"mock_routing_only": True}


def make_app(tmp_path, monkeypatch):
    model = tmp_path / "models" / "cosmos"
    model.mkdir(parents=True)
    app = create_app({"TESTING": True, "DATA_DIR": str(tmp_path / "tasks"), "MODEL_ROOTS": [str(model.parent)]})
    manager = app.extensions["task_manager"]
    monkeypatch.setattr(manager.executor, "submit", lambda *a, **kw: None)
    return app, model, manager


def test_local_vlm_api_rejects_remote_csrf_and_rebinding(tmp_path, monkeypatch):
    app, model, _ = make_app(tmp_path, monkeypatch)
    client = app.test_client()
    payload = {"model_dir": str(model), "acknowledge_experimental": True}
    assert client.post("/api/vlm/tasks", json=payload).status_code == 403
    assert client.post("/api/vlm/tasks", json=payload, headers=HEADER,
                       environ_base={"REMOTE_ADDR": "10.0.0.2"}).status_code == 403
    assert client.post("/api/vlm/tasks", json=payload, headers={**HEADER, "Host": "evil.example"}).status_code == 403
    assert client.post("/api/vlm/tasks", json=payload, headers={**HEADER, "Origin": "https://evil.example"}).status_code == 403


def test_mock_preflight_and_explicit_experimental_submit(tmp_path, monkeypatch):
    app, model, manager = make_app(tmp_path, monkeypatch)
    monkeypatch.setattr(vlm, "preflight", lambda *a: {"ready": True, "errors": [], "warnings": [], "resources": {}, "model": {}})
    client = app.test_client()
    payload = {"model_dir": str(model), "seq_length": 2048, "max_pixels": 147456}
    assert client.post("/api/vlm/preflight", json=payload, headers=HEADER).get_json()["ready"]
    assert client.post("/api/vlm/tasks", json=payload, headers=HEADER).status_code == 400
    payload["acknowledge_experimental"] = True
    response = client.post("/api/vlm/tasks", json=payload, headers=HEADER)
    assert response.status_code == 202
    task = response.get_json()
    assert task["input_kind"] == "hf_directory" and task["experimental"]
    assert "source_path" not in task and "allowed_model_roots" not in task
    assert task["parameters"]["jobs"] == 1
    assert manager.tasks[task["id"]]["source_path"] == str(model)
    # No fabricated completion: queued stays queued because this test stubs queue submission.
    assert task["status"] == "queued"


def test_vlm_tasks_and_logs_are_not_visible_to_lan(tmp_path, monkeypatch):
    app, model, manager = make_app(tmp_path, monkeypatch)
    task = manager.create_vlm({}, model, vlm.parameters({}), [model.parent])
    task_id = task["id"]
    (Path(manager.tasks[task_id]["work_dir"]) / "conversion.log").write_text("private compiler log")
    client = app.test_client()
    remote = {"REMOTE_ADDR": "10.0.0.2"}
    assert client.get("/api/tasks", environ_base=remote).get_json() == []
    for suffix in ("", "/events", "/download/log", "/download/model", "/download/package"):
        assert client.get(f"/api/tasks/{task_id}{suffix}", environ_base=remote).status_code == 403
    assert client.get(f"/api/tasks/{task_id}/download/log").status_code == 200
    assert client.get(f"/api/tasks/{task_id}/download/model").status_code == 404
    canceled = client.post(f"/api/tasks/{task_id}/cancel", headers=HEADER)
    assert canceled.get_json()["status"] == "canceled"
    assert client.get(f"/api/tasks/{task_id}/events").status_code == 200
    assert client.post(f"/api/tasks/{task_id}/cancel", headers=HEADER).status_code == 409


def test_missing_compiler_does_not_queue_or_succeed(tmp_path, monkeypatch):
    app, model, manager = make_app(tmp_path, monkeypatch)
    monkeypatch.setattr(vlm, "toolchain_probe", lambda root: {"ready": False, "errors": ["missing official compiler"]})
    response = app.test_client().post("/api/vlm/tasks", headers=HEADER,
                                     json={"model_dir": str(model), "acknowledge_experimental": True})
    assert response.status_code == 400
    assert "missing official compiler" in response.get_json()["errors"]
    assert not manager.list()


def test_cancellation_state_is_not_failed_or_completed(tmp_path, monkeypatch):
    class CancelPipeline:
        def run(self, task, update):
            raise vlm.ConversionCanceled("requested")
    manager = TaskManager(tmp_path, CancelPipeline())
    monkeypatch.setattr(manager.executor, "submit", lambda *a: None)
    task = manager.create_vlm({}, tmp_path, vlm.parameters({}), [tmp_path])
    manager._run(task["id"])
    assert manager.get(task["id"])["status"] == "canceled"
    assert manager.get(task["id"])["artifacts"] == {}
    manager.executor.shutdown()


def test_mutated_source_cannot_reach_official_worker(tmp_path):
    source, work = tmp_path / "source", tmp_path / "work"
    source.mkdir(); work.mkdir()
    f = source / "config.json"; f.write_text("{}")
    identity = [int(getattr(f.stat(), attr)) for attr in ("st_dev", "st_ino", "st_size", "st_mtime_ns")]
    f.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed after preflight"):
        VlmBModelBackend(ROOT).convert(source, work, vlm.parameters({}), {}, {}, lambda *a: None, {"config.json": identity})


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="Linux /proc and POSIX process groups are required to verify descendant termination")
def test_cancel_kills_sigterm_ignoring_descendant_even_if_leader_exits(tmp_path):
    marker = tmp_path / "child_ready"
    child_code = "import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);open(%r,'w').write(str(os.getpid()));time.sleep(60)" % str(marker)
    parent_code = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',%r]);time.sleep(60)" % child_code
    process = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        child_pid = int(marker.read_text())
        VlmBModelBackend._terminate(process)
        assert process.poll() is not None
        # A killed orphan can briefly be a zombie until container init reaps it.
        stat = Path(f"/proc/{child_pid}/stat")
        def child_state():
            try:
                return stat.read_text().split()[2]
            except (FileNotFoundError, ProcessLookupError):
                return None  # Successful exit can race any /proc exists/read.
        deadline = time.monotonic() + 2
        while child_state() not in {None, "Z"} and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_state() in {None, "Z"}
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


@pytest.mark.parametrize("outcome", ["success", "missing", "small", "missing_info", "missing_graph"])
def test_mock_official_process_requires_real_artifact_contract(tmp_path, monkeypatch, outcome):
    """Test orchestration with tiny fixture bytes, NOT actual BModel compilation."""
    source, work = tmp_path / "source", tmp_path / "work"
    source.mkdir(); work.mkdir()
    params = vlm.parameters({})
    manifest = vlm.compilation_manifest(params, {"model_config": {}, "config_sha256": "test"})
    seen = {}
    class FakeProcess:
        returncode = 0
        pid = -1
        def __init__(self, command, **kwargs):
            seen.update(command=command, kwargs=kwargs)
            target = work / "compiled"
            if outcome != "missing":
                (target / "fixture.bmodel").write_bytes(b"fixture-not-a-real-bmodel" * (1 if outcome == "small" else 4))
            if outcome != "missing_info":
                names = manifest["graph_contract"]["required_graphs"]
                if outcome == "missing_graph":
                    names = [x for x in names if x != "block_1"]
                (target / "model.log").write_text("\n".join(names))
        def poll(self):
            return 0
    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    backend = VlmBModelBackend(ROOT)
    if outcome == "success":
        result = backend.convert(source, work, params, {"python": sys.executable}, manifest,
                                 lambda *a: None, {})
        assert result == work / "model.bmodel"
        assert manifest["graph_contract"]["model_tool_names_checked"] is True
    else:
        with pytest.raises(RuntimeError):
            backend.convert(source, work, params, {"python": sys.executable}, manifest,
                            lambda *a: None, {})
        assert not (work / "model.bmodel").exists()
    assert seen["kwargs"]["start_new_session"] is True
    assert seen["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert seen["kwargs"]["env"]["HF_HUB_OFFLINE"] == "1"
    assert "--jobs" in seen["command"]
    assert "--dry_run" not in seen["command"]
    assert "--only_mlir" not in seen["command"]


def test_staging_ignores_new_assets_index_and_weights_not_in_preflight_snapshot(tmp_path, monkeypatch):
    """Only staging is inspected; no model load/export/compiler is executed."""
    source, work = tmp_path / "source", tmp_path / "work"
    source.mkdir(); work.mkdir()
    original = {"config.json": b"{}", "model.safetensors": b"snapshot-weight-fixture"}
    for name, content in original.items():
        (source / name).write_bytes(content)
    identities = {name: [int(getattr((source / name).stat(), attr)) for attr in
                         ("st_dev", "st_ino", "st_size", "st_mtime_ns")]
                  for name in original}
    # These files appear AFTER the validated snapshot; their names are otherwise
    # valid candidates. The newly added index must not change compiler loading.
    (source / "processor_config.json").write_text('{"auto_map":{"processor":"unvalidated"}}')
    (source / "model.safetensors.index.json").write_text('{"weight_map":{"x":"new.safetensors"}}')
    (source / "new.safetensors").write_bytes(b"unvalidated-weight-fixture")
    def inspect_stage_without_executing(command, **kwargs):
        stage = work / "hf_source"
        assert {p.name for p in stage.iterdir()} == set(original)
        assert (stage / "config.json").read_bytes() == original["config.json"]
        assert (stage / "model.safetensors").resolve() == source / "model.safetensors"
        raise RuntimeError("staging inspected; compiler intentionally not executed")
    monkeypatch.setattr(subprocess, "Popen", inspect_stage_without_executing)
    with pytest.raises(RuntimeError, match="staging inspected"):
        VlmBModelBackend(ROOT).convert(source, work, vlm.parameters({}), {"python": sys.executable},
                                      {}, lambda *a: None, identities)


def memory_config_fixture():
    """Config geometry only, not weights and not a hardware acceptance fixture."""
    return {
        "model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"],
        "tie_word_embeddings": True,
        "text_config": {"hidden_size": 2048, "num_hidden_layers": 28, "num_attention_heads": 16,
                        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 6144,
                        "vocab_size": 151936, "tie_word_embeddings": True, "dtype": "bfloat16",
                        "rope_scaling": {"mrope_section": [24, 20, 20], "mrope_interleaved": True}},
        "vision_config": {"depth": 24, "hidden_size": 1024, "num_heads": 16,
                          "intermediate_size": 4096, "out_hidden_size": 2048, "patch_size": 16,
                          "temporal_patch_size": 2, "spatial_merge_size": 2,
                          "num_position_embeddings": 2304, "in_channels": 3,
                          "deepstack_visual_indexes": [5, 11, 17]},
    }


def test_device_estimate_uses_mixed_precision_components_not_total_params_divided_by_two():
    estimate = vlm.device_memory_estimate(vlm.parameters({"seq_length": 1024, "max_pixels": 65536}),
                                           memory_config_fixture())
    assert estimate["status"] == "analytical_not_measured"
    assert estimate["parameter_counts"] == {
        "decoder_quantizable_matrices": 1409286144, "decoder_and_final_norms": 123904,
        "embedding": 311164928, "vision_including_deepstack": 406957056,
    }
    parts = estimate["weight_components_bytes"]
    assert parts["decoder_matrix_payload"] == 704643072
    assert parts["decoder_quantization_metadata"] == 66060288
    assert parts["embedding"] == 622329856
    assert parts["vision_including_deepstack"] == 813914112
    shared, separate = estimate["scenarios"]
    assert shared["weight_bytes"] == 2207195136
    assert separate["weight_bytes"] == 2829524992
    assert shared["known_component_sum_bytes"] == 2341412864
    assert separate["known_component_sum_bytes"] == 2963742720
    assert estimate["fit_assessed"] is False
    for key in ("peak_memory_bytes", "tpu_heap_available_bytes", "board_system_total_bytes", "board_system_available_bytes"):
        assert estimate[key] is None
    assert estimate["unknown_overheads"] and estimate["warnings"]


@pytest.mark.parametrize("length,kv_mib,extra_mib", [(512, 56, 8), (1024, 112, 16), (2048, 224, 32)])
def test_device_kv_and_explicit_buffers_scale_with_sequence_not_host_ram(length, kv_mib, extra_mib):
    params = vlm.parameters({"seq_length": length, "max_pixels": 65536})
    estimate = vlm.device_memory_estimate(params, memory_config_fixture())
    assert estimate["kv_cache_bytes"] == kv_mib * 1024**2
    assert estimate["explicit_device_buffers_bytes"] == extra_mib * 1024**2
    assert estimate["fit_assessed"] is False


@pytest.mark.parametrize("precision,bits,metadata", [("w4bf16", 4, 66060288), ("w8bf16", 8, 66060288), ("bf16", 16, 0)])
def test_device_estimate_precision_and_quantization_metadata(precision, bits, metadata):
    estimate = vlm.device_memory_estimate(vlm.parameters({"precision": precision}), memory_config_fixture())
    assert estimate["weight_components_bytes"]["decoder_matrix_payload"] == 1409286144 * bits // 8
    assert estimate["weight_components_bytes"]["decoder_quantization_metadata"] == metadata
    assert estimate["weight_components_bytes"]["vision_including_deepstack"] == 406957056 * 2


def test_preflight_retains_config_memory_estimate_but_missing_weights_never_pass(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(memory_config_fixture()))
    monkeypatch.setattr(vlm, "toolchain_probe", lambda root: {"ready": True, "errors": []})
    report = vlm.preflight(ROOT, str(model), vlm.parameters({}), [tmp_path], tmp_path / "work")
    assert report["ready"] is False and report["errors"]
    assert report["device_memory_estimate"]["status"] == "analytical_not_measured"
    assert report["device_memory_estimate"]["fit_assessed"] is False


def test_host_preflight_ready_is_not_device_memory_fit_assessment(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    cfg = memory_config_fixture()
    (model / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(vlm, "toolchain_probe", lambda root: {"ready": True, "errors": []})
    monkeypatch.setattr(vlm, "inspect_model", lambda path: {"weight_bytes": 1, "model_config": cfg})
    monkeypatch.setattr(vlm, "resource_report", lambda *a: {
        "memory_scope": "host_compiler_not_target_board", "available_memory_bytes": 64 * 1024**3,
        "disk_free_bytes": 128 * 1024**3, "estimated_work_disk_bytes": 20 * 1024**3,
        "minimum_memory_bytes": 6 * 1024**3,
    })
    report = vlm.preflight(ROOT, str(model), vlm.parameters({}), [tmp_path], tmp_path / "work")
    assert report["ready"] is True  # Mocked compiler admission only, not board fit.
    assert report["device_memory_estimate"]["fit_assessed"] is False
    assert report["device_memory_estimate"]["tpu_heap_available_bytes"] is None
    assert any("TPU heap" in warning for warning in report["warnings"])


def test_manifest_records_device_estimate_without_changing_default_preset():
    params = vlm.parameters({})
    assert params["seq_length"] == 2048 and params["max_pixels"] == 147456
    manifest = vlm.compilation_manifest(params, {"model_config": memory_config_fixture(), "config_sha256": "fixture"})
    assert manifest["device_memory_estimate"]["kv_cache_bytes"] == 224 * 1024**2
    assert manifest["device_memory_estimate"]["fit_assessed"] is False
    assert manifest["board_validated"] is False
    unknown = vlm.device_memory_estimate(params, {})
    assert unknown["status"] == "unavailable" and unknown["scenarios"] == []
