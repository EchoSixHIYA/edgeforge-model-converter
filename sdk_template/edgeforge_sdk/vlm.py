"""Qwen3-VL/Cosmos callable runtime, using Sophgo's pinned multi-graph C++ engine.

Heavy processor/runtime dependencies are loaded only when a VLM is constructed.
No generic ``sail.Engine`` fallback is valid for this autoregressive graph bundle.
"""
import importlib
import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

RUNTIME_REVISION = "2866d403dae89c4164d8b96c8c615d7f67693e6d"
TRANSFORMERS_VERSION = "4.57.1"


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Missing or invalid model asset: {}".format(path)) from exc


def _numpy(value, dtype=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=dtype)


def temporal_timestamps(frame_indices, fps, temporal_patch_size=2, timestamps=None):
    """Average real frame times per temporal patch; duplicate the final odd frame."""
    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("Video requires a positive actual source FPS; it is never guessed.")
    values = (list(timestamps) if timestamps is not None else
              [float(i) / fps for i in frame_indices])
    if not values or len(values) != len(frame_indices):
        raise ValueError("Video timestamps and sampled source frame indices must align.")
    if any(not math.isfinite(t) for t in values) or any(b < a for a, b in zip(values, values[1:])):
        raise ValueError("Video timestamps must be finite and nondecreasing.")
    while len(values) % temporal_patch_size:
        values.append(values[-1])
    return [(values[i] + values[i + temporal_patch_size - 1]) / 2
            for i in range(0, len(values), temporal_patch_size)]


def vision_positions(height, width, merge_size=2, grid_side=48):
    """Official LLM-TPU patch order and bilinear learned-position interpolation."""
    if height <= 0 or width <= 0 or height % merge_size or width % merge_size:
        raise ValueError("Vision patch grid must be positive and divisible by spatial_merge_size.")
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    def ordered(x):
        return x.reshape(height // merge_size, merge_size, width // merge_size,
                         merge_size).transpose(0, 2, 1, 3).reshape(-1)
    rotary = np.stack([ordered(rows), ordered(cols)], axis=1).astype(np.int32)
    hr = np.linspace(0, grid_side - 1, height, dtype=np.float32)
    wr = np.linspace(0, grid_side - 1, width, dtype=np.float32)
    h0, w0 = hr.astype(np.int32), wr.astype(np.int32)
    h1, w1 = np.minimum(h0 + 1, grid_side - 1), np.minimum(w0 + 1, grid_side - 1)
    dh, dw = hr - h0, wr - w0
    indices = [h0[:, None] * grid_side + w0, h0[:, None] * grid_side + w1,
               h1[:, None] * grid_side + w0, h1[:, None] * grid_side + w1]
    weights = [(1 - dh[:, None]) * (1 - dw), (1 - dh[:, None]) * dw,
               dh[:, None] * (1 - dw), dh[:, None] * dw]
    return rotary, np.stack([ordered(x) for x in indices], axis=1).astype(np.int32), \
        np.stack([ordered(x) for x in weights], axis=1).astype(np.float32)


def rope_positions(input_ids, visual_spans):
    """Qwen3-VL three-axis mRoPE; video time is already represented by text timestamps.

    visual_spans: (first visual-pad offset, merged height, merged width), one per
    image/temporal video group. All remaining tokens (including timestamps) are text.
    """
    ids = np.asarray(input_ids).reshape(-1)
    result = np.empty((3, len(ids)), dtype=np.int32)
    cursor, position = 0, 0
    for offset, height, width in visual_spans:
        count = height * width
        if offset < cursor or offset + count > len(ids):
            raise ValueError("Visual token spans overlap or exceed prompt length.")
        text_len = offset - cursor
        result[:, cursor:offset] = np.arange(position, position + text_len)[None, :]
        position += text_len
        result[0, offset:offset + count] = position
        result[1, offset:offset + count] = position + np.repeat(np.arange(height), width)
        result[2, offset:offset + count] = position + np.tile(np.arange(width), height)
        position += max(height, width)
        cursor = offset + count
    result[:, cursor:] = np.arange(position, position + len(ids) - cursor)[None, :]
    return result


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: tuple
    finish_reason: str
    prompt_tokens: int
    media_metadata: dict = field(default_factory=dict)

    def as_dict(self):
        return {"text": self.text, "token_ids": list(self.token_ids),
                "finish_reason": self.finish_reason, "prompt_tokens": self.prompt_tokens,
                "media_metadata": self.media_metadata}


@dataclass
class PreparedVLMInput:
    input_ids: np.ndarray
    position_ids: np.ndarray
    visual_inputs: list
    media_metadata: dict


def _load_processor(path):
    try:
        import transformers
        from transformers import Qwen3VLProcessor
        import torch  # noqa: F401 -- needed by the official image/video processors
        import torchvision  # noqa: F401
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Cosmos requires the VLM extras (Python >=3.10, CPU torch/torchvision, "
                           "transformers==4.57.1, Pillow and PyAV). Run install.sh --vlm "
                           "in the exported SDK; see VLM_README.md.") from exc
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError("This SDK pins transformers=={}; found {}. Use an isolated VLM environment."
                           .format(TRANSFORMERS_VERSION, transformers.__version__))

    class TimestampProcessor(Qwen3VLProcessor):
        # HF 4.57.1 computes timestamps from indices/FPS. Preserve decoded PTS for
        # variable-frame-rate videos as well, without forging frame indices/FPS.
        def _calculate_timestamps(self, indices, video_fps, merge_size=2):
            times = getattr(self, "_edgeforge_frame_times", None)
            return temporal_timestamps(indices, video_fps, 2, times)

    return TimestampProcessor.from_pretrained(str(path), local_files_only=True,
                                             trust_remote_code=False)


def _load_native():
    try:
        module = importlib.import_module("edgeforge_qwen3_vl")
    except ImportError as exc:
        raise RuntimeError("Missing Sophgo native VLM extension edgeforge_qwen3_vl. "
                           "Build runtime/qwen3_vl with the SOPHON libsophon development SDK "
                           "on the target host; follow VLM_README.md. SAIL alone is insufficient.") from exc
    if getattr(module, "EDGEFORGE_ABI", None) != 1 or getattr(module, "UPSTREAM_REVISION", None) != RUNTIME_REVISION:
        raise RuntimeError("Incompatible native VLM extension: rebuild the version shipped with this SDK.")
    return module.Qwen3_VL()


def _decode_video(path, video_processor, *, fps=None, num_frames=None, max_frames=None):
    try:
        import av
        from transformers.video_utils import VideoMetadata
    except ImportError as exc:
        raise RuntimeError("Video inference requires PyAV and transformers==4.57.1 (VLM extras).") from exc
    if fps is not None and num_frames is not None:
        raise ValueError("Specify fps OR num_frames, not both.")
    for name, value in (("fps", fps), ("num_frames", num_frames), ("max_frames", max_frames)):
        if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
            raise ValueError(name + " must be positive.")
    # Two passes keep unselected full-resolution frames out of RAM. First pass
    # establishes exact decoded count and presentation times, not container guesses.
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or 0)
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Video has no reliable FPS; provide a valid timed video (no 24/1 FPS fallback).")
        times = []
        for frame in container.decode(video=0):
            if frame.time is None:
                raise ValueError("Video frame has no presentation timestamp; remux to a timed video first.")
            times.append(float(frame.time))
        if not times:
            raise ValueError("Video contains no decodable frames.")
        origin = times[0]
        times = [value - origin for value in times]
        metadata = VideoMetadata(total_num_frames=len(times), fps=source_fps,
                                 width=stream.width, height=stream.height,
                                 duration=times[-1] + 1 / source_fps, video_backend="pyav")
    if num_frames is not None and (int(num_frames) != num_frames or num_frames > len(times)):
        raise ValueError("num_frames must be an integer no larger than the source frame count.")
    if max_frames is not None and int(max_frames) != max_frames:
        raise ValueError("max_frames must be an integer.")
    num_frames = int(num_frames) if num_frames is not None else None
    indices = video_processor.sample_frames(metadata, num_frames=num_frames, fps=fps)
    if max_frames is not None and len(indices) > max_frames:
        indices = np.linspace(0, len(times) - 1, int(max_frames)).round().astype(int)
    indices = [int(value) for value in indices]
    selected = set(indices)
    frames = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in selected:
                frames[index] = frame.to_ndarray(format="rgb24")
            if index >= max(indices):
                break
    if set(frames) != selected:
        raise ValueError("Video changed or failed to decode during sampling.")
    selected_times = [times[index] for index in indices]
    # A one-frame clip needs one repeated frame before HF smart_resize, which
    # otherwise rejects T < temporal_patch_size. All repeated metadata is explicit.
    padded = len(indices) == 1
    if padded:
        indices.append(indices[-1])
        selected_times.append(selected_times[-1])
    metadata.frames_indices = indices.copy()
    audit = {"fps": source_fps, "frames_indices": indices.copy(),
             "timestamps": selected_times, "total_num_frames": len(times),
             "temporal_timestamps": temporal_timestamps(indices, source_fps, 2, selected_times),
             "temporal_patch_size": 2, "padded_last_frame": padded or bool(len(indices) % 2),
             "requested_fps": fps, "requested_num_frames": num_frames, "max_frames": max_frames}
    return np.stack([frames[index] for index in indices]), metadata, audit


class Qwen3VL:
    """Batch-one text/image/video generation. Each call starts a fresh KV cache.

    Conversation history can be included as text/messages explicitly; implicit
    cross-call KV reuse is intentionally unsupported by this static graph profile.
    Instances serialize calls because the native engine owns mutable KV buffers.
    """
    def __init__(self, model=None, device_id=0):
        self.model_dir = Path(model) if model is not None else Path.cwd() / "model"
        self.config = _json(self.model_dir / "config.json")
        self.manifest = _json(self.model_dir / "compilation_manifest.json")
        if self.config.get("architecture") not in {"cosmos_reason2_2b", "qwen3_vl"}:
            raise ValueError("Not an EdgeForge Cosmos/Qwen3-VL model bundle.")
        if self.manifest.get("runtime") != "sophgo-llm-tpu-qwen3-vl":
            raise ValueError("Bundle does not declare the supported official multi-graph runtime.")
        revision = self.manifest.get("runtime_source", {}).get("commit", self.manifest.get("runtime_revision"))
        if revision != RUNTIME_REVISION:
            raise ValueError("Bundle runtime revision does not match this pinned SDK.")
        if self.manifest.get("lmhead_with_topk") is not False:
            raise ValueError("Bundle must expose raw lm_head logits (compile with --do_sample).")
        hf_dir = self.model_dir / self.config.get("tokenizer_directory", "processor")
        self.model_config = _json(hf_dir / "config.json")
        if self.model_config != self.manifest.get("model_config"):
            raise ValueError("Source processor model_config differs from compilation manifest.")
        self.generation_config = _json(hf_dir / "generation_config.json")
        self.processor = _load_processor(hf_dir)
        self.tokenizer = self.processor.tokenizer
        self._lock = threading.RLock()
        self._closed = True
        self._native = _load_native()
        bmodel = self.model_dir / self.config.get("model_file", "model.bmodel")
        if not bmodel.is_file():
            raise ValueError("Missing compiled model: {}".format(bmodel))
        self._native.init(int(device_id), str(bmodel))
        self._closed = False
        try:
            self._validate_runtime()
        except Exception:
            self.close()
            raise

    @classmethod
    def load(cls, model=None, **kwargs):
        return cls(model, **kwargs)

    def _validate_runtime(self):
        expected = self.manifest
        for key, attr in (("sequence_length", "SEQLEN"), ("max_input_length", "MAX_INPUT_LENGTH"),
                          ("max_pixels", "MAX_PIXELS")):
            if int(expected[key]) != int(getattr(self._native, attr)):
                raise ValueError("BModel {}={} differs from compilation manifest {}."
                                 .format(attr, getattr(self._native, attr), expected[key]))
        text, vision = self.model_config["text_config"], self.model_config["vision_config"]
        for attr, value in (("NUM_LAYERS", text["num_hidden_layers"]), ("HIDDEN_SIZE", text["hidden_size"]),
                            ("num_deepstack", len(vision["deepstack_visual_indexes"])),
                            ("VIT_DIMS", vision["in_channels"] * vision["temporal_patch_size"] * vision["patch_size"] ** 2)):
            if int(getattr(self._native, attr)) != int(value):
                raise ValueError("BModel {} does not match source HF model configuration.".format(attr))
        for token, key in (("<|image_pad|>", "image_token_id"), ("<|video_pad|>", "video_token_id"),
                           ("<|vision_start|>", "vision_start_token_id"), ("<|vision_end|>", "vision_end_token_id")):
            if self.tokenizer.convert_tokens_to_ids(token) != self.model_config[key]:
                raise ValueError("Processor special-token IDs do not match source model configuration.")
        if vision["num_position_embeddings"] != 2304 or vision["deepstack_visual_indexes"] != [5, 11, 17]:
            raise ValueError("Pinned runtime requires grid_side48 and DeepStack indexes [5,11,17].")
        if self._native.support_history or self._native.lmhead_with_topk:
            raise ValueError("Unsupported graph profile: this SDK requires static no-history prefill and raw logits.")
        if (vision["patch_size"], vision["temporal_patch_size"], vision["spatial_merge_size"]) != (16, 2, 2):
            raise ValueError("Pinned Qwen3-VL runtime requires patch16 / temporal2 / merge2.")

    def preprocess(self, prompt, **kwargs):
        with self._lock:
            return self._preprocess(prompt, **kwargs)

    def _preprocess(self, prompt, *, image=None, video=None, fps=None, num_frames=None,
                   max_frames=None, min_pixels=None, max_pixels=None, video_total_pixels=None,
                   system=None):
        """Use original HF normalization/resize defaults unless explicitly overridden.

        max_pixels is a per-image/per-temporal-group resolution budget. For video,
        video_total_pixels is the HF video processor's aggregate T*H*W budget.
        Neither automatically truncates prompt/frames to fit the compiled model.
        """
        if self._closed:
            raise RuntimeError("Model is closed.")
        if image is not None and video is not None:
            raise ValueError("Use one image OR one video per request (batch size 1).")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty text.")
        if video is None and any(x is not None for x in (fps, num_frames, max_frames, video_total_pixels)):
            raise ValueError("Video sampling options require video=...")
        content, kwargs, audit = [], {}, {}
        for name, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels),
                            ("video_total_pixels", video_total_pixels)):
            if value is not None and (int(value) != value or value <= 0):
                raise ValueError(name + " must be a positive integer.")
        if min_pixels is not None and max_pixels is not None and min_pixels > max_pixels:
            raise ValueError("min_pixels cannot exceed max_pixels.")
        if image is not None:
            from PIL import Image
            path = Path(image)
            if not path.is_file():
                raise ValueError("Image does not exist: {}".format(path))
            with Image.open(path) as source:
                picture = source.convert("RGB")
            content.append({"type": "image", "image": str(path)})
            kwargs["images"] = [picture]
            image_kwargs = {}
            if min_pixels is not None:
                image_kwargs["min_pixels"] = min_pixels
            if max_pixels is not None:
                image_kwargs["max_pixels"] = max_pixels
            if image_kwargs:
                kwargs["images_kwargs"] = image_kwargs
            audit = {"type": "image", "source_size": list(picture.size),
                     "min_pixels": min_pixels, "max_pixels": max_pixels}
        if video is not None:
            path = Path(video)
            if not path.is_file():
                raise ValueError("Video does not exist: {}".format(path))
            frames, metadata, audit = _decode_video(path, self.processor.video_processor,
                                                    fps=fps, num_frames=num_frames, max_frames=max_frames)
            content.append({"type": "video", "video": str(path)})
            self.processor._edgeforge_frame_times = audit["timestamps"]
            # Sampling was already performed using the official configured sampler;
            # disable only re-sampling, not image normalization or spatial resizing.
            video_kwargs = {"video_metadata": [metadata], "do_sample_frames": False}
            if any(x is not None for x in (min_pixels, max_pixels, video_total_pixels)):
                size = dict(self.processor.video_processor.size)
                padded_frames = (len(frames) + 1) // 2 * 2
                if min_pixels is not None:
                    size["shortest_edge"] = min_pixels * padded_frames
                if max_pixels is not None:
                    size["longest_edge"] = max_pixels * padded_frames
                if video_total_pixels is not None:
                    size["longest_edge"] = min(size["longest_edge"], video_total_pixels)
                if size["shortest_edge"] > size["longest_edge"]:
                    raise ValueError("Video pixel budget is below configured minimum; set an explicit min_pixels too.")
                video_kwargs["size"] = size
            kwargs.update(videos=[frames], videos_kwargs=video_kwargs)
            audit.update(type="video", min_pixels=min_pixels, max_pixels=max_pixels,
                         video_total_pixels=video_total_pixels)
        content.append({"type": "text", "text": prompt})
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": content})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            inputs = self.processor(text=[text], return_tensors="np", **kwargs)
        finally:
            self.processor._edgeforge_frame_times = None
        ids = _numpy(inputs["input_ids"], np.int32)
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Only batch-one prompts are supported.")
        ids = ids[0]
        vocab_size = self.model_config["text_config"]["vocab_size"]
        if len(ids) == 0 or np.any(ids < 0) or np.any(ids >= vocab_size):
            raise ValueError("Processor emitted empty/out-of-vocabulary token IDs.")
        if len(ids) > self._native.MAX_INPUT_LENGTH or len(ids) >= self._native.SEQLEN:
            raise ValueError("Prompt has {} tokens; compiled prefill limit is {}, sequence limit {}. "
                             "Reduce frames/resolution/prompt or recompile; input is never silently truncated."
                             .format(len(ids), self._native.MAX_INPUT_LENGTH, self._native.SEQLEN))
        visual_inputs, spans = self._visual_inputs(inputs, ids, image is not None, video is not None)
        audit["grids"] = [entry[4].reshape(-1).tolist() for entry in visual_inputs]
        return PreparedVLMInput(ids, rope_positions(ids, spans), visual_inputs, audit)

    def _visual_inputs(self, inputs, ids, has_image, has_video):
        if not has_image and not has_video:
            return [], []
        key = "image" if has_image else "video"
        grids = _numpy(inputs[key + "_grid_thw"], np.int32)
        if grids.shape != (1, 3):
            raise ValueError("Exactly one image or video is supported per call.")
        t, h, w = [int(x) for x in grids[0]]
        if has_image and t != 1:
            raise ValueError("Image processor must produce one temporal patch.")
        pixels = _numpy(inputs["pixel_values" if has_image else "pixel_values_videos"], np.float32)
        if pixels.shape != (t * h * w, int(self._native.VIT_DIMS)):
            raise ValueError("Pixel patch shape does not match temporal2 patch16 runtime.")
        if h * w > self._native.MAX_PATCHES:
            raise ValueError("Visual frame has {} patches; compiled maximum {}. Set explicit max_pixels "
                             "or recompile. Processor settings are not silently overridden."
                             .format(h * w, self._native.MAX_PATCHES))
        pad_id = self.model_config[key + "_token_id"]
        start_id = self.model_config["vision_start_token_id"]
        end_id = self.model_config["vision_end_token_id"]
        offsets = [i + 1 for i in range(len(ids) - 1) if ids[i] == start_id and ids[i + 1] == pad_id]
        if len(offsets) != t:
            raise ValueError("Vision-start tokens do not match temporal patch groups.")
        count = h * w // 4
        if np.count_nonzero(ids == pad_id) != t * count:
            raise ValueError("Visual pad token count does not match the processor grid.")
        rotary, pos_idx, pos_weight = vision_positions(h, w)
        results, spans = [], []
        for i, offset in enumerate(offsets):
            if offset + count >= len(ids) or not np.all(ids[offset:offset + count] == pad_id) or ids[offset + count] != end_id:
                raise ValueError("Malformed visual token span.")
            results.append((pixels[i * h * w:(i + 1) * h * w], rotary, pos_idx, pos_weight,
                            np.array([[1, h, w]], dtype=np.int32), offset))
            spans.append((offset, h // 2, w // 2))
        return results, spans

    def _generation_options(self, overrides):
        supported = {"do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "eos_token_id", "seed"}
        unknown = set(overrides) - supported
        if unknown:
            raise ValueError("Unsupported generation settings: " + ", ".join(sorted(unknown)))
        cfg = dict(self.generation_config)
        cfg.update(overrides)
        eos = cfg.get("eos_token_id", self.model_config["text_config"].get("eos_token_id"))
        eos = [eos] if isinstance(eos, int) else eos
        if not eos or any(not isinstance(i, int) or i < 0 for i in eos):
            raise ValueError("generation_config must contain valid eos_token_id(s).")
        sampling = cfg.get("do_sample", False)
        temperature, top_p = float(cfg.get("temperature", 1.0)), float(cfg.get("top_p", 1.0))
        top_k, penalty = cfg.get("top_k", 0), float(cfg.get("repetition_penalty", 1.0))
        if not isinstance(sampling, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("do_sample must be boolean and top_k a nonnegative integer.")
        if not all(math.isfinite(x) for x in (temperature, top_p, penalty)) or temperature <= 0 or not 0 < top_p <= 1 or penalty <= 0:
            raise ValueError("Require temperature>0, 0<top_p<=1, repetition_penalty>0.")
        seed = cfg.get("seed")
        if seed is not None and (not isinstance(seed, int) or not 0 <= seed <= 2 ** 32 - 1):
            raise ValueError("seed must be an unsigned 32-bit integer.")
        return eos, sampling, temperature, top_p, top_k, penalty, seed

    def generate(self, prompt, *, max_new_tokens=256, generation=None, **media_options):
        with self._lock:
            if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be a positive integer.")
            eos, sampling, temperature, top_p, top_k, penalty, seed = self._generation_options(generation or {})
            prepared = self.preprocess(prompt, **media_options)
            self._native.clear_history()
            self._native.configure_generation(sampling, temperature, top_p, top_k, penalty)
            if seed is not None:
                self._native.set_seed(seed)
            self._native.forward_embed(prepared.input_ids)
            for visual_input in prepared.visual_inputs:
                self._native.forward_vit(*visual_input)
            token = int(self._native.forward_first(prepared.position_ids))
            max_position = int(prepared.position_ids.max())
            output, finish = [], "length"
            available = min(max_new_tokens, int(self._native.SEQLEN) - len(prepared.input_ids))
            while len(output) < available:
                if token in eos:
                    finish = "eos"
                    break
                output.append(token)
                if len(output) >= available:
                    break
                max_position += 1
                token = int(self._native.forward_next(np.array([max_position] * 3, dtype=np.int32)))
            decoded = self.tokenizer.decode(output, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            return GenerationResult(decoded, tuple(output), finish, len(prepared.input_ids), prepared.media_metadata)

    def predict(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def close(self):
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                if not getattr(self, "_closed", True):
                    self._closed = True
                    self._native.deinit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


class CosmosReason(Qwen3VL):
    """NVIDIA Cosmos-Reason2-2B with its exported processor and generation defaults."""
