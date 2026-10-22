"""Portable Cosmos/Qwen3-VL SDK packaging, deliberately separate from CV SDKs."""
import hashlib
import json
import os
import shutil
from pathlib import Path

RUNTIME_REVISION = "2866d403dae89c4164d8b96c8c615d7f67693e6d"
REQUIRED_ASSETS = ("config.json", "tokenizer_config.json", "preprocessor_config.json",
                   "video_preprocessor_config.json", "generation_config.json")
OPTIONAL_ASSETS = ("tokenizer.json", "vocab.json", "merges.txt", "chat_template.json",
                   "chat_template.jinja", "special_tokens_map.json", "added_tokens.json",
                   "processor_config.json", "LICENSE", "LICENSE.txt", "NOTICE", "README.md")


def _read_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Missing or invalid VLM asset: {}".format(path)) from exc
    if not isinstance(data, dict):
        raise ValueError("VLM JSON asset must be an object: {}".format(path))
    return data


def validate_processor_assets(model_dir):
    """Fail closed: small config JSONs do not constitute a usable tokenizer."""
    model_dir = Path(model_dir)
    config = {name: _read_json(model_dir / name) for name in REQUIRED_ASSETS}
    fast = model_dir / "tokenizer.json"
    vocab_file, merges_file = model_dir / "vocab.json", model_dir / "merges.txt"
    if fast.is_file():
        tokenizer = _read_json(fast)
        tokenizer_model = tokenizer.get("model", {})
        vocab = tokenizer_model.get("vocab")
        if not isinstance(vocab, dict) or not vocab or tokenizer_model.get("type") != "BPE":
            raise ValueError("tokenizer.json must contain the complete Qwen BPE vocabulary, not just tokenizer_config.json.")
        if not isinstance(tokenizer_model.get("merges"), list) or not tokenizer_model["merges"]:
            raise ValueError("tokenizer.json is missing its BPE merge table.")
        ids = set(vocab.values())
        ids.update(item.get("id") for item in tokenizer.get("added_tokens", []) if isinstance(item, dict))
    elif vocab_file.is_file() and merges_file.is_file():
        vocab = _read_json(vocab_file)
        if not vocab or not merges_file.read_text(encoding="utf-8").strip():
            raise ValueError("Qwen tokenizer requires nonempty vocab.json and merges.txt.")
        ids = set(vocab.values())
    else:
        raise ValueError("Incomplete tokenizer: export requires full tokenizer.json OR both vocab.json and merges.txt; "
                         "the six configuration JSON files alone cannot produce a runnable SDK.")
    tokenizer_cfg = config["tokenizer_config.json"]
    ids.update(int(key) for key in tokenizer_cfg.get("added_tokens_decoder", {}))
    hf = config["config.json"]
    generation = config["generation_config.json"]
    eos = generation.get("eos_token_id", hf.get("text_config", {}).get("eos_token_id"))
    eos = [eos] if isinstance(eos, int) else eos
    if not isinstance(eos, list) or not eos or any(not isinstance(x, int) for x in eos):
        raise ValueError("generation_config.json requires valid EOS IDs.")
    base_size = hf.get("text_config", {}).get("bos_token_id", generation.get("bos_token_id"))
    if not isinstance(base_size, int) or base_size <= 0 or not set(range(base_size)).issubset(ids):
        raise ValueError("Tokenizer is truncated: missing base vocabulary IDs below bos_token_id.")
    required_ids = eos + [hf[key] for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")]
    if any(token not in ids for token in required_ids):
        raise ValueError("Tokenizer vocabulary/added tokens are incomplete: required EOS/vision token IDs are missing.")
    if not ((model_dir / "chat_template.json").is_file() or (model_dir / "chat_template.jinja").is_file()
            or tokenizer_cfg.get("chat_template")):
        raise ValueError("Missing original chat template (chat_template.json/.jinja or tokenizer_config.chat_template).")
    return config


def package_vlm_sdk(root, task, model_dir, bmodel_path, manifest, work):
    """Package actual compiler output + complete local HF assets; never download/fake them."""
    root, model_dir, bmodel_path, work = map(Path, (root, model_dir, bmodel_path, work))
    assets = validate_processor_assets(model_dir)
    if isinstance(manifest, (str, Path)):
        manifest = _read_json(Path(manifest))
    manifest = dict(manifest)
    if manifest.get("runtime") != "sophgo-llm-tpu-qwen3-vl" or manifest.get("lmhead_with_topk") is not False:
        raise ValueError("VLM SDK requires the official multi-graph runtime with raw lm_head logits (--do_sample).")
    revision = manifest.get("runtime_source", {}).get("commit", manifest.get("runtime_revision"))
    if revision != RUNTIME_REVISION:
        raise ValueError("Compilation manifest does not match the packaged official runtime revision.")
    for key in ("sequence_length", "max_input_length", "max_pixels"):
        if not isinstance(manifest.get(key), int) or manifest[key] <= 0:
            raise ValueError("Compilation manifest requires positive " + key)
    if manifest.get("model_config") != assets["config.json"]:
        raise ValueError("Compilation manifest model_config differs from the exported processor configuration.")
    if not bmodel_path.is_file() or bmodel_path.stat().st_size == 0:
        raise ValueError("Cannot package an absent/empty compiled BModel.")
    stage = work / "sdk"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(root / "sdk_template", stage,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build", "rknn-toolkit-lite2"))
    bundle = stage / "model"
    processor = bundle / "processor"
    processor.mkdir(parents=True)
    copied = []
    for name in REQUIRED_ASSETS + OPTIONAL_ASSETS:
        path = model_dir / name
        if path.is_file():
            shutil.copy2(path, processor / name)
            copied.append(name)
    try:
        os.link(bmodel_path, bundle / "model.bmodel")
    except OSError:
        shutil.copy2(bmodel_path, bundle / "model.bmodel")
    config = {"architecture": task.get("architecture", "cosmos_reason2_2b"),
              "vendor": "sophon", "backend": "bmodel", "runtime": manifest["runtime"],
              "platform": task.get("platform", manifest.get("chip", "bm1684x")),
              "precision": task.get("precision", manifest.get("precision")),
              "modality": "text+image+video", "task_type": "generation", "result_type": "GenerationResult",
              "model_file": "model.bmodel", "tokenizer_directory": "processor",
              "sdk_class": "CosmosReason", "experimental": True}
    (bundle / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["sdk"] = {"class": "CosmosReason", "native_module": "edgeforge_qwen3_vl", "native_abi": 1,
                       "python": ">=3.10", "transformers": "4.57.1", "torch": "2.9.0", "torchvision": "0.24.0",
                       "pybind11": "2.13.6", "av": "16.0.1", "batch_size": 1,
                       "kv_history_across_calls": False,
                       "generation": ["greedy", "temperature", "top_p", "top_k", "repetition_penalty", "seed", "eos_token_id", "max_new_tokens"],
                       "sampling_implementation": "EdgeForge host sampler over raw lm_head logits; not bitwise HF RNG parity",
                       "libsophon": "Matching official runtime + development SDK + board/PCIe driver required; no unverified minimum version claimed"}
    manifest["processor_assets"] = copied
    manifest["validation_status"] = {"compiler_output_present": True,
                                     "runtime_unit_tests_are_not_hardware_validation": True,
                                     "hardware_inference_verified": False}
    # All archived files are local task assets; no credentials or source weights.
    manifest["asset_sha256"] = {
        str(path.relative_to(bundle)): _sha256(path)
        for path in sorted(bundle.rglob("*")) if path.is_file()
    }
    (bundle / "compilation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # The multi-GB BModel has one canonical copy in the archive: model/.
    shutil.copy2(stage / "vlm_example.py", stage / "example.py")
    (stage / "使用说明.txt").write_text(
        "Cosmos/Qwen3-VL SDK（实验性；尚未完成真实权重/板端推理验收）\n"
        "请先阅读 VLM_README.md；需要匹配的 libsophon C++ runtime/dev + 驱动，不是仅安装 SAIL。\n"
        "bash install.sh --vlm 安装隔离 VLM 依赖并构建官方衍生 native runtime。\n"
        "python example.py --model model --prompt 'Describe the scene' --image image.jpg --max-pixels 147456\n"
        "模型、完整 processor/tokenizer、generation/chat 配置和编译清单位于 model/。\n", encoding="utf-8")
    archive = work / "{}_{}_{}_sdk.zip".format(config["architecture"], config["platform"], config["precision"])
    shutil.make_archive(str(archive.with_suffix("")), "zip", stage)
    return archive


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
