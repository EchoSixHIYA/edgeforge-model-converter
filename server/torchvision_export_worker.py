"""Export supported TorchVision training checkpoints to RKNN-friendly ONNX."""
import argparse
import json
from pathlib import Path

import torch
from torchvision.models import mobilenet_v2, resnet50


def _checkpoint(path):
    # weights_only blocks executable pickle payloads while allowing tensors and
    # simple metadata such as the class-name list used by the generated SDK.
    value = torch.load(path, map_location="cpu", weights_only=True)
    labels = []
    if isinstance(value, dict):
        labels = value.get("labels") or value.get("names") or []
        state = value.get("state_dict") or value.get("model_state_dict") or value
    else:
        state = value
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint must contain a non-empty state_dict")
    normalized = {}
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        key = str(key)
        normalized[key[7:] if key.startswith("module.") else key] = tensor
    state = normalized
    if not state:
        raise ValueError("checkpoint does not contain tensor weights")
    if isinstance(labels, dict):
        labels = [str(labels[key]) for key in sorted(labels, key=lambda item: int(item))]
    elif isinstance(labels, (list, tuple)):
        labels = [str(item) for item in labels]
    else:
        labels = []
    return state, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True, choices=["mobilenet_v2", "resnet50"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    state, labels = _checkpoint(args.model)
    model = mobilenet_v2(weights=None) if args.architecture == "mobilenet_v2" else resnet50(weights=None)
    model.load_state_dict(state, strict=True)
    model.eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, torch.zeros(1, 3, 224, 224), str(output), opset_version=12,
        input_names=["input"], output_names=["output"], do_constant_folding=True,
    )
    Path(args.metadata).write_text(
        json.dumps({"labels": labels, "checkpoint_type": "torchvision_state_dict"},
                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
