import argparse
import json
from pathlib import Path

import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, input_values):
        return self.model(input_values).logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()
    assets = Path(args.assets)
    config = Wav2Vec2Config.from_pretrained(assets)
    config._attn_implementation = "eager"
    model = Wav2Vec2ForCTC(config)
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) - {"wav2vec2.masked_spec_embed"} or unexpected:
        raise ValueError(f"unexpected Wav2Vec2 checkpoint keys: missing={missing}, unexpected={unexpected}")
    model.eval()
    sample_count = 20 * 16000
    torch.onnx.export(
        Logits(model).eval(), torch.zeros(1, sample_count), args.output,
        input_names=["input"], output_names=["output"], opset_version=12,
        do_constant_folding=True)
    vocab = json.loads((assets / "vocab.json").read_text(encoding="utf-8"))
    labels = [None] * (max(vocab.values()) + 1)
    for token, index in vocab.items():
        labels[index] = token
    Path(args.metadata).write_text(json.dumps({
        "sample_rate": 16000, "sample_count": sample_count,
        "chunk_seconds": 20, "labels": labels,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
