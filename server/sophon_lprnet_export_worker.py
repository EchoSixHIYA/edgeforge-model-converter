"""Export an LPRNet checkpoint using the official sophon-demo network."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("edgeforge_sophon_lprnet", args.network)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(args.model, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError("LPRNet checkpoint must contain a PyTorch state dict")
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}

    # sophon-demo's reference network uses the same 68-class Chinese plate
    # alphabet as the RKNN integration, but exposes build_lprnet() instead of
    # the constructor used by the RKNN Model Zoo network.
    model = module.build_lprnet(lpr_max_len=8, phase=False, class_num=68, dropout_rate=0)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            torch.zeros(1, 3, 24, 94),
            args.output,
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )
    Path(args.metadata).write_text(json.dumps({
        "model_variant": "lprnet_chinese_68",
        "input_size": [94, 24],
        "characters": [
            "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
            "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
            "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4",
            "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H",
            "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W",
            "X", "Y", "Z", "I", "O", "-",
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
