import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import onnx
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules
from transformers import CLIPTextModelWithProjection


class TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        features = self.model(input_ids=input_ids).text_embeds
        return features / features.norm(p=2, dim=-1, keepdim=True)


class ClipSequenceEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final

    def forward(self, input_ids):
        features = self.token_embedding(input_ids).type(self.positional_embedding.dtype)
        features = features + self.positional_embedding
        features = features.permute(1, 0, 2)
        features = self.transformer(features)
        features = features.permute(1, 0, 2)
        return self.ln_final(features).type(self.positional_embedding.dtype)


class DetectorEncoder(torch.nn.Module):
    """Keep the Rockchip deployment switches while preserving dynamic outputs."""

    def __init__(self, deployed):
        super().__init__()
        self.deployed = deployed

    def forward(self, images, texts):
        base = self.deployed.baseModel
        if hasattr(base.backbone, "forward_image"):
            image_features = base.backbone.forward_image(images)
        else:
            image_features = base.backbone.image_model(images)
        if getattr(base, "with_neck", False):
            if getattr(base, "mm_neck", False):
                neck_features = base.neck(image_features, texts)
            else:
                neck_features = base.neck(image_features)
        else:
            neck_features = image_features
        head_features = base.bbox_head(neck_features, texts)
        outputs = []
        for class_scores, box_distances in zip(*head_features):
            outputs.extend((class_scores.to(dtype=torch.float32).sigmoid(), box_distances))
        return tuple(outputs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--tokenizer-output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--clip-runtime", choices=("hf", "openai"), default="hf")
    return parser.parse_args()


def main():
    args = parse_args()
    repo, source, assets = Path(args.repo), Path(args.model), Path(args.assets)
    labels = [line.strip() for line in Path(args.labels).read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if len(labels) != 80:
        raise ValueError(f"YOLO-World requires the official 80-class prompt list, got {len(labels)}")

    checkpoint = torch.load(source, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    prefix = "backbone.text_model.model."
    text_state = {key[len(prefix):]: value for key, value in state.items()
                  if key.startswith(prefix)}
    if not text_state:
        raise ValueError("checkpoint does not contain the YOLO-World CLIP text tower")

    with tempfile.TemporaryDirectory(prefix="edgeforge-yolo-world-") as temporary:
        local_assets = Path(temporary) / "clip"
        shutil.copytree(assets, local_assets)
        torch.save(text_state, local_assets / "pytorch_model.bin")

        stem_channels = int(state.get("backbone.image_model.stem.conv.weight").shape[0])
        variant_by_stem = {32: "s", 48: "m", 64: "l", 80: "x", 96: "xl"}
        variant = variant_by_stem.get(stem_channels)
        if not variant:
            raise ValueError(
                f"unsupported YOLO-World V2 stem width {stem_channels}; "
                "expected one of 32/48/64/80/96")
        config_path = repo / "configs" / "pretrain" / (
            f"yolo_world_v2_{variant}_vlpan_bn_2e-3_100e_4x8gpus_"
            "obj365v1_goldg_train_lvis_minival.py")
        config = Config.fromfile(config_path)
        config.model.backbone.text_model.model_name = str(local_assets)
        register_all_modules(init_default_scope=True)
        import_modules_from_strings(**config.custom_imports)
        model = MODELS.build(config.model)
        load_checkpoint(model, str(source), map_location="cpu")
        model.to("cpu")
        model.eval()
        model.reparameterize([labels])

        sys.path.insert(0, str(repo / "deploy"))
        from easydeploy.model import DeployModel, MMYOLOBackend
        detector = DetectorEncoder(
            DeployModel(model, MMYOLOBackend.ONNXRUNTIME, postprocess_cfg=None))
        detector.eval()
        image = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
        texts = model.text_feats.detach().to(dtype=torch.float32)
        with torch.no_grad():
            outputs = detector(image, texts)
        output_names = [f"output_{index}" for index in range(len(outputs))]
        torch.onnx.export(
            detector, (image, texts), args.detector,
            input_names=["images", "texts"], output_names=output_names,
            opset_version=12, do_constant_folding=True)
        detector_model = onnx.load(args.detector)
        static_outputs = (
            (1, len(labels), 80, 80), (1, 4, 80, 80),
            (1, len(labels), 40, 40), (1, 4, 40, 40),
            (1, len(labels), 20, 20), (1, 4, 20, 20),
        )
        for output, shape in zip(detector_model.graph.output, static_outputs):
            tensor_shape = output.type.tensor_type.shape
            tensor_shape.ClearField("dim")
            for value in shape:
                tensor_shape.dim.add().dim_value = int(value)
        onnx.checker.check_model(detector_model)
        onnx.save(detector_model, args.detector)

        sequence_length = int(args.sequence_length)
        if args.clip_runtime == "openai":
            import clip
            openai_model, _ = clip.load("ViT-B/32", device="cpu")
            openai_model.eval()
            np_projection = openai_model.text_projection.detach().cpu().numpy().astype("float32")
            npy_path = Path(args.metadata).with_name("text_projection_512_512.npy")
            import numpy as np
            np.save(npy_path, np_projection)
            bpe_src = Path(clip.__file__).with_name("bpe_simple_vocab_16e6.txt.gz")
            if bpe_src.is_file():
                shutil.copy2(bpe_src, Path(args.metadata).with_name("bpe_simple_vocab_16e6.txt.gz"))
            text_encoder = ClipSequenceEncoder(openai_model).eval()
            input_ids = torch.zeros((1, sequence_length), dtype=torch.int64)
            opset = 13
        else:
            text_model = CLIPTextModelWithProjection.from_pretrained(local_assets).eval()
            text_encoder = TextEncoder(text_model).eval()
            input_ids = torch.full((1, sequence_length), 49407, dtype=torch.long)
            opset = 18
        torch.onnx.export(
            text_encoder, input_ids, args.text, input_names=["input_ids"],
            output_names=["text_features"], opset_version=opset, do_constant_folding=True)

    tokenizer_output = Path(args.tokenizer_output)
    tokenizer_output.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "special_tokens_map.json"):
        path = assets / name
        if path.is_file():
            shutil.copy2(path, tokenizer_output / name)
    Path(args.metadata).write_text(json.dumps({
        "model_variant": f"yolo_world_v2_{variant}",
        "input_size": [640, 640],
        "num_classes": 80,
        "sequence_length": int(args.sequence_length),
        "pad_token_id": 49407,
        "token_pad_id": 0 if args.clip_runtime == "openai" else 49407,
        "text_projection": "text_projection_512_512.npy" if args.clip_runtime == "openai" else None,
        "labels": labels,
        "model_files": {"detector": "detector.rknn", "text": "text.rknn"},
        "tokenizer_directory": "tokenizer",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
