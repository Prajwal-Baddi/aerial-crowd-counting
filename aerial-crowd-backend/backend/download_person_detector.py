"""Download the official COCO person detector once; API startup stays offline."""
from pathlib import Path

import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights


def main():
    root = Path(__file__).resolve().parents[2]
    directory = root / "checkpoints/person_detection"
    directory.mkdir(parents=True, exist_ok=True)
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    torch.hub.load_state_dict_from_url(
        weights.url, model_dir=str(directory), map_location="cpu",
        progress=True, check_hash=True, weights_only=True,
    )
    print(f"Detector weights ready in {directory}")


if __name__ == "__main__":
    main()
