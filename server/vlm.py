"""Opt-in, local-only Cosmos Reason 2 2B -> official TPU-MLIR VLM route.

No Torch import, model deserialization, download, or compilation happens here.
Preflight is evidence about local files, not proof of successful compilation.
"""
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ARCHITECTURE = "cosmos_reason2_2b"
TPU_MLIR_VERSION = "1.28.1"
TPU_MLIR_COMMIT = "43676b3fcf07f44158e63baaefe6c0e809d16890"
RUNTIME_REVISION = "2866d403dae89c4164d8b96c8c615d7f67693e6d"
PRECISIONS = ("w4bf16", "w8bf16", "bf16")
ASSET_NAMES = (
    "config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
    "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json",
    "chat_template.json", "chat_template.jinja", "special_tokens_map.json", "added_tokens.json",
    "processor_config.json", "LICENSE", "LICENSE.txt", "NOTICE", "README.md",
)
REQUIRED_ASSETS = ("config.json", "tokenizer_config.json", "preprocessor_config.json",
                   "video_preprocessor_config.json", "generation_config.json")


class ConversionCanceled(RuntimeError):
    pass


def model_roots(value=None):
    value = os.environ.get("EDGEFORGE_MODEL_ROOTS", "") if value is None else value
    if isinstance(value, (list, tuple)):
        return [Path(p).expanduser().resolve() for p in value]
    return [Path(p).expanduser().resolve() for p in str(value).split(os.pathsep) if p]


def resolve_model_dir(value, roots):
    if not roots:
        raise ValueError("本地目录导入未开启；管理员须设置 EDGEFORGE_MODEL_ROOTS（允许读取的模型根目录）")
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError("model_dir 必须是允许根目录内的绝对路径")
    path = Path(value).resolve(strict=True)
    if not path.is_dir() or not any(path == r or r in path.parents for r in roots):
        raise ValueError("model_dir 不在 EDGEFORGE_MODEL_ROOTS 允许的目录范围内")
    # A materialized HF local-dir is expected, not a cache snapshot with escaping links.
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"模型目录不接受符号链接；请使用完整的 HF local-dir: {item.name}")
        if not item.is_dir() and not item.is_file():
            raise ValueError(f"模型目录含非普通文件: {item.name}")
        if item.suffix.lower() in {".py", ".bin", ".pt", ".pth", ".pkl", ".pickle"}:
            raise ValueError(f"只接受原始 safetensors 与配置，不接受可执行代码或 pickle 权重: {item.name}")
    return path


def parameters(data):
    if not isinstance(data, dict):
        raise ValueError("请求必须是 JSON 对象")
    allowed = {"model_dir", "architecture", "vendor", "platform", "precision", "seq_length",
               "max_pixels", "jobs", "acknowledge_experimental"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError("未知 VLM 参数: " + ", ".join(unknown))
    if data.get("architecture", ARCHITECTURE) != ARCHITECTURE or data.get("vendor", "sophon") != "sophon":
        raise ValueError("此入口只接受 Cosmos Reason 2 2B / Sophon")
    result = {"platform": data.get("platform", "bm1684x"), "precision": data.get("precision", "w4bf16"),
              "seq_length": data.get("seq_length", 2048), "max_pixels": data.get("max_pixels", 147456),
              "jobs": data.get("jobs", 1)}
    if result["platform"] != "bm1684x":
        raise ValueError("此实验链路仅接入 BM1684X，不支持 BM1684/RKNN")
    if result["precision"] not in PRECISIONS:
        raise ValueError("原始 BF16 模型必须显式选择 w4bf16、w8bf16 或 bf16；不支持 auto")
    for key in ("seq_length", "max_pixels", "jobs"):
        if type(result[key]) is not int:
            raise ValueError(f"{key} 必须为整数")
    if not 256 <= result["seq_length"] <= 8192 or result["seq_length"] % 128:
        raise ValueError("实验序列长度须为 256–8192 范围内的 128 倍数")
    if not 65536 <= result["max_pixels"] <= 1048576 or result["max_pixels"] % 1024:
        raise ValueError("max_pixels 须为 65536–1048576 范围内的 1024 倍数（像素预算，不是边长）")
    if result["max_pixels"] // 1024 + 64 >= result["seq_length"]:
        raise ValueError("视觉 token 预算过大；序列还须保留文本与生成空间")
    if result["jobs"] != 1:
        raise ValueError("当前受控实验只允许 jobs=1（同时限制导出与 GNU parallel 编译）")
    return result


def read_json(path, max_bytes=32 * 1024 * 1024):
    if not path.is_file() or not 0 < path.stat().st_size <= max_bytes:
        raise ValueError(f"缺失、为空或过大的 JSON 文件: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path.name}")
    return value


def validate_config(config):
    expected = {
        "model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"],
        "tie_word_embeddings": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Cosmos Reason 2 2B 配置不匹配: {key}")
    if config.get("auto_map") or config.get("quantization_config"):
        raise ValueError("不接受自定义 remote code 或预量化配置；此链路只接入原始 BF16")
    t, v = config.get("text_config", {}), config.get("vision_config", {})
    for key, value in {"hidden_size": 2048, "num_hidden_layers": 28, "num_attention_heads": 16,
                       "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 6144,
                       "vocab_size": 151936, "tie_word_embeddings": True}.items():
        if t.get(key) != value:
            raise ValueError(f"Cosmos Reason 2 2B text_config 不匹配: {key}")
    if t.get("dtype", t.get("torch_dtype")) != "bfloat16":
        raise ValueError("此配置链路要求原始 bfloat16 权重")
    rope = t.get("rope_scaling", {})
    if rope.get("mrope_section") != [24, 20, 20] or rope.get("mrope_interleaved") is not True:
        raise ValueError("要求 interleaved MRoPE [24,20,20]")
    for key, value in {"depth": 24, "hidden_size": 1024, "num_heads": 16, "intermediate_size": 4096,
                       "out_hidden_size": 2048, "patch_size": 16, "temporal_patch_size": 2,
                       "spatial_merge_size": 2, "num_position_embeddings": 2304,
                       "in_channels": 3, "deepstack_visual_indexes": [5, 11, 17]}.items():
        if v.get(key) != value:
            raise ValueError(f"Cosmos Reason 2 2B vision_config 不匹配: {key}")


def safetensors_headers(paths):
    """Read bounded headers only; verify offsets and floating point source dtypes."""
    tensors = {}
    for path in paths:
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"截断的 safetensors: {path.name}")
            length = struct.unpack("<Q", raw)[0]
            if not 2 <= length <= min(32 * 1024 * 1024, size - 8):
                raise ValueError(f"不合法的 safetensors 头部长度: {path.name}")
            header = json.loads(stream.read(length))
        if not isinstance(header, dict):
            raise ValueError("safetensors 头部不是对象")
        for key, item in header.items():
            if key == "__metadata__":
                continue
            if key in tensors:
                raise ValueError(f"重复 tensor: {key}")
            if not isinstance(item, dict) or item.get("dtype") not in {"BF16", "F16", "F32"}:
                raise ValueError(f"不支持的预量化/非法 tensor: {key}")
            shape, offsets = item.get("shape"), item.get("data_offsets")
            if not isinstance(shape, list) or not all(type(n) is int and n > 0 for n in shape):
                raise ValueError(f"非法 tensor shape: {key}")
            if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(n) is int for n in offsets):
                raise ValueError(f"非法 tensor offsets: {key}")
            count = 1
            for n in shape:
                count *= n
            element_bytes = 4 if item["dtype"] == "F32" else 2
            if not 0 <= offsets[0] <= offsets[1] <= size - 8 - length or offsets[1] - offsets[0] != count * element_bytes:
                raise ValueError(f"截断或非法 tensor 数据区: {key}")
            tensors[key] = {"shape": shape, "dtype": item["dtype"], "file": path.name}
    return tensors


def validate_weights(path):
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError("缺少真实 .safetensors 权重；仅配置文件不能开始转换")
    index_path = path / "model.safetensors.index.json"
    index = read_json(index_path) if index_path.exists() else None
    if index:
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("无效的 safetensors 分片索引")
        for name in set(mapping.values()):
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
                raise ValueError("safetensors 索引含不安全分片路径")
            if not (path / name).is_file():
                raise ValueError(f"缺失 safetensors 分片: {name}")
    tensors = safetensors_headers(files)
    if index and any(k not in tensors or tensors[k]["file"] != v for k, v in index["weight_map"].items()):
        raise ValueError("safetensors 索引与实际 tensor 头部不一致")
    required = {"model.language_model.embed_tokens.weight": [151936, 2048],
                "model.language_model.norm.weight": [2048],
                "model.visual.patch_embed.proj.weight": [1024, 3, 2, 16, 16]}
    for layer in range(28):
        prefix = f"model.language_model.layers.{layer}."
        for name, shape in {
            "input_layernorm.weight": [2048], "post_attention_layernorm.weight": [2048],
            "self_attn.q_proj.weight": [2048, 2048], "self_attn.k_proj.weight": [1024, 2048],
            "self_attn.v_proj.weight": [1024, 2048], "self_attn.o_proj.weight": [2048, 2048],
            "self_attn.q_norm.weight": [128], "self_attn.k_norm.weight": [128],
            "mlp.gate_proj.weight": [6144, 2048], "mlp.up_proj.weight": [6144, 2048],
            "mlp.down_proj.weight": [2048, 6144],
        }.items():
            required[prefix + name] = shape
    for name, shape in required.items():
        if name not in tensors or tensors[name]["shape"] != shape:
            raise ValueError(f"缺失或 shape 不匹配的官方映射 tensor: {name}")
    for i in range(3):
        if not any(k.startswith(f"model.visual.deepstack_merger_list.{i}.") for k in tensors):
            raise ValueError(f"缺少 DeepStack merger {i} 权重")
    for i in range(24):
        if not any(k.startswith(f"model.visual.blocks.{i}.") for k in tensors):
            raise ValueError(f"缺少视觉层 {i} 权重")
    return {"weight_bytes": sum(p.stat().st_size for p in files), "weight_files": [p.name for p in files],
            "tensor_count": len(tensors), "header_validation": "partial_shapes_and_full_offsets_not_accuracy"}


def inspect_model(path):
    for name in REQUIRED_ASSETS:
        read_json(path / name)
    if not (path / "tokenizer.json").is_file() and not all((path / p).is_file() for p in ("vocab.json", "merges.txt")):
        raise ValueError("缺少 tokenizer.json 或 vocab.json + merges.txt；tokenizer_config 不能替代词表")
    if not any((path / p).is_file() for p in ("chat_template.json", "chat_template.jinja")):
        raise ValueError("缺少原始 chat_template.json / chat_template.jinja")
    for name in ASSET_NAMES:
        f = path / name
        if f.exists() and (not f.is_file() or f.stat().st_size == 0):
            raise ValueError(f"无效模型配套文件: {name}")
        if f.exists() and name.endswith(".json") and name not in {"tokenizer.json", "vocab.json"}:
            if read_json(f).get("auto_map"):
                raise ValueError("不接受模型包中的自定义 remote code")
    if any((path / name).stat().st_size > 128 * 1024**2 for name in ASSET_NAMES if (path / name).is_file()):
        raise ValueError("模型 tokenizer/processor 配套文件超过 128 MiB 安全上限")
    from .vlm_sdk import validate_processor_assets
    validate_processor_assets(path)
    config = read_json(path / "config.json")
    validate_config(config)
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        p = read_json(path / name)
        if p.get("processor_class") != "Qwen3VLProcessor" or any(p.get(k) != v for k, v in
                {"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2}.items()):
            raise ValueError(f"处理器配置与 Qwen3-VL 不匹配: {name}")
    result = validate_weights(path)
    identity_files = [*result["weight_files"], *[p for p in ASSET_NAMES if (path / p).is_file()]]
    if (path / "model.safetensors.index.json").is_file():
        identity_files.append("model.safetensors.index.json")
    result["file_identities"] = {name: [int(getattr((path / name).stat(), attr)) for attr in
                                              ("st_dev", "st_ino", "st_size", "st_mtime_ns")]
                                 for name in identity_files}
    result.update(model_type=config["model_type"], config_sha256=hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
                  model_config=config, asset_files=[p for p in ASSET_NAMES if (path / p).is_file()])
    return result


def toolchain_probe(root):
    python = os.environ.get("EDGEFORGE_LLM_PYTHON", sys.executable)
    worker = Path(root) / "server" / "vlm_worker.py"
    result = {"ready": False, "version": None, "python": python, "entrypoint": None, "errors": []}
    try:
        call = subprocess.run([python, str(worker), "--probe"], capture_output=True, text=True, timeout=30)
        if call.returncode:
            raise RuntimeError((call.stderr or call.stdout)[-2000:])
        result.update(json.loads(call.stdout.strip().splitlines()[-1]))
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as exc:
        result["errors"] = [f"TPU-MLIR VLM 环境未就绪: {exc}"]
    return result


def resource_report(work, weight_bytes):
    available = None
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
        available = next(int(l.split()[1]) * 1024 for l in lines if l.startswith("MemAvailable:"))
        # Respect a container memory limit, not only the host's MemAvailable.
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit.isdigit():
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            available = min(available, max(0, int(limit) - used))
    except (OSError, StopIteration, ValueError):
        pass
    parent = Path(work)
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    # Deliberately conservative workspace estimate: float intermediates, per-graph
    # and combined models, SDK staging + ZIP. Not a measured compiler peak.
    estimate = max(20 * 1024**3, weight_bytes * 6)
    return {"available_memory_bytes": available, "disk_free_bytes": free,
            "weight_bytes": weight_bytes, "estimated_work_disk_bytes": estimate,
            "minimum_memory_bytes": 6 * 1024**3, "estimate_is_measured": False}


def preflight(root, model_dir, params, roots, work):
    report = {"ready": False, "errors": [], "warnings": [], "parameters": params,
              "model": {}, "resources": {}, "toolchain": toolchain_probe(root)}
    try:
        path = resolve_model_dir(model_dir, roots)
        report["model"] = inspect_model(path)
        report["resources"] = resource_report(work, report["model"]["weight_bytes"])
        resources = report["resources"]
        if resources["disk_free_bytes"] < resources["estimated_work_disk_bytes"]:
            report["errors"].append("工作盘空间低于保守预估；需要至少 %.1f GiB 可用空间" % (resources["estimated_work_disk_bytes"] / 1024**3))
        if resources["available_memory_bytes"] is not None and resources["available_memory_bytes"] < resources["minimum_memory_bytes"]:
            report["errors"].append("可用内存不足 6 GiB 的最低预检阈值；这不是实测峰值保证")
        report["warnings"].append("RAM/磁盘预估不是实测峰值；即使预检通过仍可能 OOM。导出和编译均串行。")
        report["warnings"].append("仅核验配置、权重头部和依赖；未证明编译正确性、量化质量或 BM1684X 板端推理。")
        report["warnings"].append("max_pixels 是每次视觉调用的像素预算；视频还受时间分组、帧分辨率和总序列 token 上限约束。")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["errors"].append(str(exc))
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", str(Path(work).resolve())):
        report["errors"].append("官方编译器内部使用 shell；任务工作目录必须只含 ASCII 字母、数字、_、-、/、.（请设置 DATA_DIR）")
    report["errors"].extend(report["toolchain"].get("errors", []))
    report["ready"] = not report["errors"] and report["toolchain"].get("ready", False)
    return report


def compilation_manifest(params, model):
    length, pixels = params["seq_length"], params["max_pixels"]
    patches, tokens = pixels // 256, pixels // 1024
    graphs = ["embedding", "embedding_cache", "vit", "add", "lm_head", "greedy_head", "sample_head"]
    graphs += [f"block_{i}" for i in range(28)] + [f"block_cache_{i}" for i in range(28)]
    return {
        "schema_version": 1, "architecture": ARCHITECTURE, "model_family": "qwen3_vl",
        "runtime": "sophgo-llm-tpu-qwen3-vl", "runtime_revision": RUNTIME_REVISION,
        "runtime_source": {"repository": "https://github.com/sophgo/LLM-TPU", "commit": RUNTIME_REVISION},
        "compiler": {"name": "tpu-mlir", "version": TPU_MLIR_VERSION, "commit": TPU_MLIR_COMMIT},
        "experimental": True, "verified": False, "board_validated": False,
        "chip": params["platform"], "precision": params["precision"], "model_file": "model.bmodel",
        "sequence_length": length, "max_input_length": length, "max_pixels": pixels,
        "lmhead_with_topk": False, "model_config": model["model_config"],
        "source_config_sha256": model["config_sha256"],
        "compile": {"seq_length": length, "max_input_length": length, "max_pixels": pixels,
                    "jobs": 1, "q_group_size": 64, "batch": 1, "num_device": 1,
                    "dynamic": False, "do_sample": True, "lmhead_with_topk": False,
                    "embedding_disk": False, "use_block_with_kv": False},
        "visual": {"patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
                   "max_patches": patches, "max_visual_tokens": tokens,
                   "budget_scope": "per_visual_call_and_total_sequence", "max_shape": None},
        "graph_contract": {"required_graphs": graphs, "source": "expected_from_pinned_converter_not_board_validated",
                           "vit_input_shapes": [[patches, 1536], [patches, 2], [patches, 4], [patches, 4, 1], [1, 1, patches, patches]],
                           "vit_output_shapes": [[tokens, 2048]] * 4,
                           "lm_head_output_shape": [1, 151936]},
    }
