"""Export official openai/clip-vit-base-patch32 PyTorch weights to two ONNX graphs."""
import argparse
import json
import shutil
from pathlib import Path

import onnx
import torch
import torch.nn.functional as functional
from transformers import CLIPConfig, CLIPModel


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--assets", required=True)
parser.add_argument("--image", required=True)
parser.add_argument("--text", required=True)
parser.add_argument("--tokenizer-output", required=True)
parser.add_argument("--metadata", required=True)
args = parser.parse_args()

assets = Path(args.assets).resolve()
required = ("config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.json", "merges.txt")
missing = [name for name in required if not (assets / name).is_file()]
if missing:
    raise RuntimeError(f"CLIP exporter assets are missing: {missing}; rerun scripts/setup_wsl.sh")

config = CLIPConfig.from_pretrained(str(assets), local_files_only=True)
config._attn_implementation = "eager"
config.text_config._attn_implementation = "eager"
config.vision_config._attn_implementation = "eager"
state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
state_dict.pop("text_model.embeddings.position_ids", None)
state_dict.pop("vision_model.embeddings.position_ids", None)
model = CLIPModel(config)
model.load_state_dict(state_dict, strict=True)
model.eval()


class ImageEncoder(torch.nn.Module):
    def __init__(self, clip):
        super().__init__(); self.clip = clip

    def forward(self, pixel_values):
        return functional.normalize(self.clip.get_image_features(pixel_values=pixel_values), dim=-1)


class TextEncoder(torch.nn.Module):
    def __init__(self, clip):
        super().__init__(); self.clip = clip

    def forward(self, input_ids):
        return functional.normalize(self.clip.get_text_features(input_ids=input_ids), dim=-1)


torch.onnx.export(
    ImageEncoder(model), torch.randn(1, 3, 224, 224), args.image,
    input_names=["pixel_values"], output_names=["image_embeds"],
    opset_version=18, do_constant_folding=True,
)
torch.onnx.export(
    TextEncoder(model), torch.full((1, 20), 49407, dtype=torch.long), args.text,
    input_names=["input_ids"], output_names=["text_embeds"],
    opset_version=18, do_constant_folding=True,
)
for artifact in (args.image, args.text):
    onnx.checker.check_model(onnx.load(artifact))

tokenizer_output = Path(args.tokenizer_output).resolve()
tokenizer_output.mkdir(parents=True, exist_ok=True)
for name in required[1:]:
    shutil.copy2(assets / name, tokenizer_output / name)
Path(args.metadata).write_text(json.dumps({
    "input_layout": "nhwc", "sequence_length": 20, "pad_token_id": 49407,
    "logit_scale": 4.605170249938965,
    "model_files": {"image": "image.rknn", "text": "text.rknn"},
    "tokenizer_directory": "tokenizer",
}, ensure_ascii=False, indent=2), encoding="utf-8")
