
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import ErrorAwareCrowdDataset


def check_density_generation():
    """Test Gaussian normalization and boundary handling."""

    # Create a temporary manifest and image.
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        image_dir = root / "images"
        annotation_dir = root / "annotations"
        processed_dir = root / "processed"

        image_dir.mkdir()
        annotation_dir.mkdir()
        processed_dir.mkdir()

        image_path = image_dir / "sample.jpg"

        Image.fromarray(
            np.zeros((512, 512, 3), dtype=np.uint8)
        ).save(image_path)

        points = [
            [100.0, 100.0],   # Interior
            [0.0, 0.0],       # Top-left boundary
            [511.0, 511.0],   # Bottom-right boundary
            [256.0, 256.0],   # Center
        ]

        record = {
            "split": "test",
            "image": "images/sample.jpg",
            "original_width": 512,
            "original_height": 512,
            "padded_width": 512,
            "padded_height": 512,
            "crop_x": 0,
            "crop_y": 0,
            "crop_size": 512,
            "density_size": 64,
            "density_stride": 8,
            "gaussian_sigma": 3.0,
            "points": points,
            "count": len(points),
        }

        manifest = processed_dir / "test.jsonl"

        with manifest.open("w", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        dataset = ErrorAwareCrowdDataset(
            manifest_path=manifest,
            dataset_root=root,
            crop_size=512,
            density_stride=8,
            gaussian_sigma=3.0,
        )

        image, density, count = dataset[0]

        assert image.shape == (3, 512, 512)
        assert density.shape == (1, 64, 64)
        assert count.shape == torch.Size([])

        assert torch.isfinite(image).all()
        assert torch.isfinite(density).all()

        assert image.min() >= 0
        assert image.max() <= 1

        assert density.min() >= 0

        # Four pedestrians should contribute four counts.
        assert abs(density.sum().item() - 4.0) < 1e-5

        assert count.item() == 4.0

        # Check empty crop handling.
        empty_density = dataset._generate_density_map([])

        assert empty_density.shape == (1, 64, 64)
        assert empty_density.sum().item() == 0.0

        print("Synthetic density generation checks passed.")


def check_real_dataset():
    """Check actual prepared VisDrone samples."""

    project_root = Path(__file__).resolve().parent.parent

    dataset_root = (
        project_root / "dataset" / "VisDrone2019-DET-train"
    )

    manifest_path = (
        project_root / "error_aware" / "processed" / "train.jsonl"
    )

    dataset = ErrorAwareCrowdDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )

    print(f"Training crops available: {len(dataset)}")

    indices = sorted(
        set([
            0,
            len(dataset) // 4,
            len(dataset) // 2,
            3 * len(dataset) // 4,
            len(dataset) - 1,
        ])
    )

    for index in indices:
        image, density, count = dataset[index]

        assert image.shape == (3, 512, 512)
        assert density.shape == (1, 64, 64)

        assert torch.isfinite(image).all()
        assert torch.isfinite(density).all()

        assert density.min() >= 0

        density_count = density.sum().item()
        expected_count = count.item()

        assert abs(density_count - expected_count) < 1e-4, (
            f"Count mismatch at index {index}: "
            f"density={density_count}, "
            f"expected={expected_count}"
        )

        print(
            f"Sample {index:6d} | "
            f"Pedestrians: {expected_count:5.0f} | "
            f"Density sum: {density_count:8.4f}"
        )

    print("Real dataset checks passed.")


if __name__ == "__main__":
    check_density_generation()
    check_real_dataset()

    print("\nAll data pipeline checks passed.")
