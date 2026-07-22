"""Create the reproducible ResNet-50 checkpoint used for acceptance."""
from pathlib import Path

import torch
from torchvision.models import ResNet50_Weights, resnet50


def main():
    output = Path("data/smoke/resnet50.pth")
    output.parent.mkdir(parents=True, exist_ok=True)
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    torch.save({"state_dict": model.state_dict(), "labels": weights.meta["categories"]}, output)
    print(output, output.stat().st_size)


if __name__ == "__main__":
    main()
