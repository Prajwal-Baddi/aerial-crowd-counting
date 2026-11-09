import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset_loader import VisDroneCrowdDataset
from model import CrowdCounter

DATASET_ROOT = "dataset/VisDrone2019-DET-train"
SPLIT_FILE = "residual_phase2_split.json"
COARSE_CHECKPOINT = "residual_phase2_coarse.pth"
CACHE_FILE = "residual_phase2_cache.pt"
BATCH_SIZE = 1


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads(Path(SPLIT_FILE).read_text(encoding="utf-8"))
    dataset = VisDroneCrowdDataset(root=DATASET_ROOT, image_size=(320, 320))

    if len(dataset) != split["dataset_size"]:
        raise RuntimeError(
            f"Dataset size changed: split expects {split['dataset_size']}, "
            f"loader found {len(dataset)}."
        )

    checkpoint = load_checkpoint(COARSE_CHECKPOINT, device)
    model = CrowdCounter(
        dropout_rate=checkpoint.get("dropout_rate", 0.25)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cache_indices = (
        split["refiner_train"]
        + split["refiner_validation"]
        + split["final_test"]
    )
    loader = DataLoader(
        Subset(dataset, cache_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    cache = {}
    refiner_indices = set(
        split["refiner_train"] + split["refiner_validation"]
    )
    cursor = 0
    for images, gt_density, counts in loader:
        image_index = cache_indices[cursor]
        images = images.to(device)
        coarse = model(images).squeeze(0).cpu().float()

        if coarse.shape != (1, 80, 80):
            raise RuntimeError(
                f"Image {image_index}: expected coarse [1,80,80], "
                f"got {tuple(coarse.shape)}"
            )

        # Ground-truth density maps are cached only for refiner train/validation.
        # The final test partition is kept out of the training cache.
        entry = {"coarse": coarse.contiguous()}
        if image_index in refiner_indices:
            target = gt_density.squeeze(0).cpu().float()
            if target.shape != (1, 80, 80):
                raise RuntimeError(
                    f"Image {image_index}: expected target [1,80,80], "
                    f"got {tuple(target.shape)}"
                )
            entry["target"] = target.contiguous()
            entry["count"] = float(counts.item())
        cache[str(image_index)] = entry
        cursor += 1
        if cursor % 100 == 0 or cursor == len(cache_indices):
            print(f"Cached {cursor}/{len(cache_indices)} images")

    torch.save({
        "cache": cache,
        "split_file": SPLIT_FILE,
        "coarse_checkpoint": COARSE_CHECKPOINT,
    }, CACHE_FILE)
    print(f"Saved density cache: {CACHE_FILE}")
    print(f"Cached images: {len(cache)}")


if __name__ == "__main__":
    main()
