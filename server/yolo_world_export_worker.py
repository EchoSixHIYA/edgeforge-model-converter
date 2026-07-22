import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
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


class DetectorEncoder(torch.nn.Module):
    """Keep the Rockchip deployment switches while preserving dynamic outputs."""

    def __init__(self, deployed):
        super().__init__()
        self.deployed = deployed

    def forward(self, images, texts):
        base = self.deployed.baseModel
        image_features = base.backbone.image_model(images)
        neck_features = base.neck(image_features, texts)
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

        config_path = repo / "configs" / "pretrain" / (
            "yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_"
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

        text_model = CLIPTextModelWithProjection.from_pretrained(local_assets).eval()
        text_encoder = TextEncoder(text_model).eval()
        input_ids = torch.full((1, 20), 49407, dtype=torch.long)
        torch.onnx.export(
            text_encoder, input_ids, args.text, input_names=["input_ids"],
            output_names=["text_features"], opset_version=18, do_constant_folding=True)

    tokenizer_output = Path(args.tokenizer_output)
    tokenizer_output.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "special_tokens_map.json"):
        path = assets / name
        if path.is_file():
            shutil.copy2(path, tokenizer_output / name)
    Path(args.metadata).write_text(json.dumps({
        "model_variant": "yolo_world_v2_s",
        "input_size": [640, 640],
        "num_classes": 80,
        "sequence_length": 20,
        "pad_token_id": 49407,
        "labels": labels,
        "model_files": {"detector": "detector.rknn", "text": "text.rknn"},
        "tokenizer_directory": "tokenizer",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
