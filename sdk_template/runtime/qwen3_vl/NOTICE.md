# Third-party source and local changes

`chat.cpp` is derived from Sophgo LLM-TPU:
https://github.com/sophgo/LLM-TPU/blob/2866d403dae89c4164d8b96c8c615d7f67693e6d/models/Qwen3_VL/python_demo/chat.cpp

`upstream_pipeline.py` is retained unmodified for source comparison, **not executed**:
https://github.com/sophgo/LLM-TPU/blob/2866d403dae89c4164d8b96c8c615d7f67693e6d/models/Qwen3_VL/python_demo/pipeline.py

License: accompanying `LICENSE` (BSD-2-Clause, except third-party components).
Preserve this notice and license when redistributing source or binaries. User-supplied
Cosmos weights/tokenizer have their own NVIDIA/Hugging Face license terms; their
source LICENSE/NOTICE are copied if present. This SDK does not grant weight access.

EdgeForge modifications to native runtime:
- Uniquely named `edgeforge_qwen3_vl` module with pinned ABI/revision markers.
- Raw logits are read from the lm_head's own allocated buffer (not the potentially
  smaller embedding buffer). Host sampler in `sampling.h` implements greedy,
  temperature, top-k/top-p and repetition penalty, and optional random seed.
- Graph/profile and exact static input/output shape/dtype checks before tensor access;
  batch-one Cosmos-Reason2-2B profile. Native initialization failures clean up resources.
- Fresh-call KV reset and exposed model dimensions for manifest verification.
- Prefill, decode, device KV movement, all four ViT outputs and three DeepStack
  additions retain the pinned Sophgo implementation.
- CMake/build.sh adapt the official BMRT/bmlib + pybind11 build for an active venv.

Python SDK adapts the upstream patch ordering, learned-position interpolation and
mRoPE algorithm. It uses the source HF processor, not the demo's 1 FPS/resolution
constants, and preserves source indices/FPS and decoded presentation timestamps.
HF processing remains in the separately installed transformers dependency under
its upstream Apache-2.0 license; no HF implementation is vendored here.

Not validated here: building against a real libsophon SDK, real model compilation,
real device inference or numerical parity against the source model. Host tests
are not a substitute. This is experimental software, not a certification.
