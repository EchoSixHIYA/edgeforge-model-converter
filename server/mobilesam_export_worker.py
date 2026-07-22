"""Export an original MobileSAM checkpoint into RKNN-Zoo compatible ONNX graphs."""
import argparse
import json
import sys
from pathlib import Path

import onnx
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--encoder", required=True)
parser.add_argument("--decoder", required=True)
parser.add_argument("--metadata", required=True)
args = parser.parse_args()

repo = Path(args.repo).resolve()
sys.path.insert(0, str(repo))
from mobile_sam import sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel

sam = sam_model_registry["vit_t"](checkpoint=args.model)
sam.eval()

encoder = Path(args.encoder).resolve()
encoder.parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(
    sam.image_encoder, torch.randn(1, 3, 448, 448), str(encoder),
    input_names=["images"], output_names=["image_embeddings"],
    opset_version=16, do_constant_folding=True,
)

decoder_wrapper = SamOnnxModel(
    model=sam, return_single_mask=False, use_stability_score=False,
    return_extra_metrics=False,
)
decoder_wrapper.eval()
decoder_temp = Path(args.decoder).with_suffix(".full.onnx").resolve()
decoder_inputs = {
    "image_embeddings": torch.randn(1, 256, 28, 28),
    "point_coords": torch.tensor([[[190.0, 70.0], [460.0, 280.0]]]),
    "point_labels": torch.tensor([[2.0, 3.0]]),
    "mask_input": torch.zeros(1, 1, 112, 112),
    "has_mask_input": torch.zeros(1),
    "orig_im_size": torch.tensor([448.0, 448.0]),
}
torch.onnx.export(
    decoder_wrapper, tuple(decoder_inputs.values()), str(decoder_temp),
    input_names=list(decoder_inputs),
    output_names=["masks", "iou_predictions", "low_res_masks"],
    opset_version=16, do_constant_folding=True,
)
onnx.utils.extract_model(
    str(decoder_temp), str(Path(args.decoder).resolve()),
    ["image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input"],
    ["iou_predictions", "low_res_masks"],
)
decoder_temp.unlink()
for artifact in (encoder, Path(args.decoder).resolve()):
    onnx.checker.check_model(onnx.load(str(artifact)))

Path(args.metadata).write_text(json.dumps({
    "input_layout": "nhwc", "prompt_count": 2,
    "model_files": {"encoder": "encoder.rknn", "decoder": "decoder.rknn"},
    "upstream_repository": "https://github.com/airockchip/MobileSAM",
}, ensure_ascii=False, indent=2), encoding="utf-8")
