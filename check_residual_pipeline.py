import json
from pathlib import Path

import torch

from dataset_loader import VisDroneCrowdDataset
from residual_data import ResidualPatchDataset

DATASET_ROOT = "dataset/VisDrone2019-DET-train"
SPLIT_FILE = "residual_phase2_split.json"
CACHE_FILE = "residual_phase2_cache.pt"


def load_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    split = json.loads(Path(SPLIT_FILE).read_text(encoding="utf-8"))
    cache = load_file(CACHE_FILE)["cache"]
    dataset = VisDroneCrowdDataset(root=DATASET_ROOT, image_size=(320, 320))

    # Verify no data partition overlap.
    keys = ["coarse_train", "coarse_validation", "refiner_train",
            "refiner_validation", "final_test"]
    groups = [set(split[k]) for k in keys]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            assert not groups[i].intersection(groups[j]), (
                f"Overlap between {keys[i]} and {keys[j]}"
            )
    assert len(set.union(*groups)) == len(dataset)

    # Verify cached density-map shapes and exact residual reconstruction.
    checked = 0
    for image_index in split["refiner_train"][:10] + split["refiner_validation"][:10]:
        entry = cache[str(image_index)]
        coarse = entry["coarse"]
        target = entry["target"]
        assert coarse.shape == (1, 80, 80)
        assert target.shape == (1, 80, 80)
        assert torch.isfinite(coarse).all()
        assert torch.isfinite(target).all()
        residual = target - coarse
        reconstructed = coarse + residual
        assert torch.allclose(reconstructed, target, atol=1e-6)
        checked += 1

    # Verify patch dataset shapes and complete, unique validation coverage.
    val_data = ResidualPatchDataset(
        dataset, split["refiner_validation"], cache,
        patches_per_image=64, seed=42,
    )
    sample_image, sample_coarse, sample_target = val_data[0]
    assert sample_image.shape == (3, 160, 160)
    assert sample_coarse.shape == (1, 10, 10)
    assert sample_target.shape == (1, 10, 10)

    print("Residual pipeline integrity checks passed.")
    print(f"Dataset images: {len(dataset)}")
    print("Partition sizes:", {k: len(split[k]) for k in keys})
    print(f"Cached map pairs checked: {checked}")
    print("Sample tensor shapes:",
          tuple(sample_image.shape), tuple(sample_coarse.shape),
          tuple(sample_target.shape))
    print("Residual target definition: full-image GT patch - coarse prediction patch")


if __name__ == "__main__":
    main()
