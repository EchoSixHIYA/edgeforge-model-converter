"""Export the SOPHGO YOLO-World v2 CLIP text tower with the OpenAI CLIP runtime.

This worker intentionally runs in a Torch 1.13 environment.  SOPHGO's
YOLO_world_v2 notes call out that Torch 2.x SDPA export can compile into a
degraded text encoder under TPU-MLIR.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import clip


class ClipSequenceEncoder(nn.Module):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--bpe-output", required=True)
    args = parser.parse_args()

    model, _ = clip.load("ViT-B/32", device="cpu")
    model.eval()
    np.save(args.projection, model.text_projection.detach().cpu().numpy().astype(np.float32))
    bpe = Path(clip.__file__).with_name("bpe_simple_vocab_16e6.txt.gz")
    if bpe.is_file():
        shutil.copy2(bpe, args.bpe_output)

    encoder = ClipSequenceEncoder(model).eval()
    dummy = torch.zeros((1, 77), dtype=torch.int64)
    torch.onnx.export(
        encoder, dummy, args.output,
        input_names=["input_ids"], output_names=["text_features"],
        opset_version=13, do_constant_folding=True,
    )

    import onnx
    from onnxsim import simplify
    simplified, ok = simplify(onnx.load(args.output))
    if not ok:
        raise RuntimeError("onnxsim failed for YOLO-World CLIP text encoder")
    onnx.checker.check_model(simplified)
    onnx.save(simplified, args.output)


if __name__ == "__main__":
    main()
