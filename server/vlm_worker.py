#!/usr/bin/env python3
"""Run the pinned official entrypoint in an isolated compiler interpreter.

The adapter changes concurrency and remote-code policy only. It does not fork
or replace Qwen3-VL graph export, quantization, compilation, or model combine.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

VERSION = "1.28.1"


def validate_compiler_tools():
    # ViT PPL kernels are compiled on demand, after graph export. Check their
    # host build tools up front rather than failing deep inside ppl_jit.sh.
    tools = ("model_deploy.py", "model_tool", "tpuc-opt", "parallel", "cmake", "make", "gcc", "g++")
    missing = [name for name in tools if not shutil.which(name)]
    if missing:
        raise RuntimeError("Missing official compiler prerequisites: " + ", ".join(missing))


def validate_host_runtime():
    # The pinned wheel bundles glibc 2.35. Mixing it with the older Ubuntu
    # 20.04 loader can crash even /usr/bin/perl (and thus GNU parallel).
    if sys.platform == "linux":
        name, version = platform.libc_ver()
        if name == "glibc" and version:
            numbers = tuple(int(part) for part in version.split(".")[:2])
            if numbers < (2, 35):
                raise RuntimeError(
                    f"TPU-MLIR {VERSION} requires glibc >= 2.35 (Ubuntu 22.04+); "
                    f"this host has glibc {version}. Use Ubuntu-22.04 or the official compiler container.")


def initialize_toolchain():
    validate_host_runtime()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("TPU-MLIR 1.28.1 wheel requires Python 3.10; set EDGEFORGE_LLM_PYTHON to its isolated venv")
    actual = importlib.metadata.version("tpu_mlir")
    if actual != VERSION:
        raise RuntimeError(f"Expected tpu_mlir=={VERSION}, found {actual}")
    if importlib.metadata.version("transformers") != "5.3.0":
        raise RuntimeError("Pinned compiler environment requires transformers==5.3.0")
    # Official package initialization configures sys.path, PATH and TPUC_ROOT.
    # Do NOT call its console wrapper: that spawns another Python process and
    # would discard our explicit concurrency controls.
    with contextlib.redirect_stdout(io.StringIO()):
        import tpu_mlir
    base = Path(tpu_mlir.__file__).resolve().parent
    entry = base / "python" / "tools" / "llm_convert.py"
    if not entry.is_file():
        raise RuntimeError(f"Official llm_convert.py not found in installed wheel: {entry}")
    validate_compiler_tools()
    result = subprocess.run(["parallel", "--version"], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(
            f"GNU parallel version probe failed (exit={result.returncode}): "
            f"{result.stderr.strip() or 'no diagnostic output; check host runtime/library compatibility'}")
    if "GNU parallel" not in result.stdout:
        raise RuntimeError("GNU parallel is required, not another program named parallel")
    importlib.metadata.version("torch")
    return entry


def apply_controls(converter_class, jobs):
    """Version-specific control of BOTH official exporter and GNU parallel."""
    if jobs != 1:
        raise ValueError("Only jobs=1 is supported")
    original_init = converter_class.__init__
    original_gen = converter_class.gen_all_mlir
    original_compile = converter_class.compile_all

    def controlled_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # execute_tasks() constructs GNU parallel -j from this exact field.
        self.max_workers = jobs

    def record_stage(self, name):
        out_dir = getattr(self, "out_dir", None)
        if out_dir:
            path = Path(out_dir).parent / "compiler_stage.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"stage": name}), encoding="utf-8")
            temporary.replace(path)

    def serial_gen(self):
        record_stage(self, "export")
        print("EDGEFORGE_STAGE:export (official serial export path)", flush=True)
        previous = self.debug
        # v1.28.1 explicitly uses a serial loop in debug mode. Restore the
        # setting so compile flags/cleanup retain their original semantics.
        self.debug = True
        try:
            return original_gen(self)
        finally:
            self.debug = previous

    def controlled_compile(self):
        self.max_workers = jobs
        record_stage(self, "compile")
        print("EDGEFORGE_STAGE:compile (GNU parallel -j 1)", flush=True)
        return original_compile(self)

    converter_class.__init__ = controlled_init
    converter_class.gen_all_mlir = serial_gen
    converter_class.compile_all = controlled_compile


def run(args, entry):
    from transformers import AutoConfig
    from llm.LlmConverter import LlmConverter
    from llm.Qwen3VLConverter import Qwen3VLConverter  # noqa: F401 - fail before loading weights
    apply_controls(LlmConverter, args.jobs)
    original_config = AutoConfig.from_pretrained

    def safe_config(cls, *positional, **kwargs):
        kwargs.update(trust_remote_code=False, local_files_only=True)
        config = original_config(*positional, **kwargs)
        if config.model_type != "qwen3_vl":
            raise ValueError("Expected qwen3_vl, not a text-to-image model")
        return config

    AutoConfig.from_pretrained = classmethod(safe_config)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    sys.argv = [str(entry), "-m", args.model_path, "-o", args.out_dir, "-s", str(args.seq_length),
                "-q", args.precision, "-g", "64", "-c", "bm1684x", "--num_device", "1",
                "--num_core", "1", "-b", "1", "--max_pixels", str(args.max_pixels),
                "--max_input_length", str(args.seq_length), "--do_sample"]
    print(json.dumps({"official_entrypoint": str(entry), "argv": sys.argv, "jobs": 1,
                      "entrypoint_sha256": hashlib.sha256(entry.read_bytes()).hexdigest()}), flush=True)
    runpy.run_path(str(entry), run_name="__main__")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--out-dir")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=147456)
    parser.add_argument("--precision", choices=("w4bf16", "w8bf16", "bf16"), default="w4bf16")
    parser.add_argument("--jobs", type=int, choices=(1,), default=1)
    args = parser.parse_args(argv)
    entry = initialize_toolchain()
    # LD_LIBRARY_PATH must be present when the native loader starts. The
    # official console wrapper also launches a fresh interpreter after setup.
    if os.environ.get("_EDGEFORGE_TPU_ENV_READY") != "1":
        environment = dict(os.environ, _EDGEFORGE_TPU_ENV_READY="1")
        environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], environment)
    # Check actual native/Python imports, not only package metadata.
    import pymlir  # noqa: F401
    import torch
    from llm.Qwen3VLConverter import Qwen3VLConverter  # noqa: F401
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding  # noqa: F401
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 4):
        raise RuntimeError("transformers 5.3.0 requires torch >= 2.4; compiler recommendation is CPU torch 2.4.0")
    if args.probe:
        print(json.dumps({"ready": True, "version": VERSION, "python": sys.executable,
                          "entrypoint": str(entry), "errors": []}))
        return
    if not args.model_path or not args.out_dir:
        parser.error("--model-path and --out-dir are required")
    run(args, entry)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VLM worker failed: {exc}", file=sys.stderr)
        raise
