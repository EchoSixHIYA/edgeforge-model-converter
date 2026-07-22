"""Create the reproducible MobileNet V2 checkpoint used for acceptance."""
from pathlib import Path

import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def main():
    output = Path("data/smoke/mobilenet_v2.pth")
    output.parent.mkdir(parents=True, exist_ok=True)
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    torch.save({"state_dict": model.state_dict(), "labels": weights.meta["categories"]}, output)
    print(output, output.stat().st_size)


if __name__ == "__main__":
    main()
