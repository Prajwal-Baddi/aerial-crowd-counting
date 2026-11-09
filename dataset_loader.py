
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


class VisDroneCrowdDataset(Dataset):
    """
    VisDrone pedestrian-counting dataset.

    Uses category 1 (pedestrian) only.
    Category 2 (people/group) is excluded because its
    individual pedestrian count is unknown.

    Images: 3 x 320 x 320
    Density targets: 1 x 80 x 80
    """

    def __init__(
        self,
        root="dataset/VisDrone2019-DET-train",
        image_size=(320, 320),
        sigma=1.0,
    ):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.annotation_dir = self.root / "annotations"
        self.image_size = image_size
        self.sigma = sigma

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir.resolve()}"
            )

        if not self.annotation_dir.exists():
            raise FileNotFoundError(
                f"Annotation directory not found: "
                f"{self.annotation_dir.resolve()}"
            )

        self.image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and (self.annotation_dir / f"{path.stem}.txt").exists()
        )

        if not self.image_paths:
            raise RuntimeError(
                "No matching images and annotation files were found."
            )

        if image_size[0] % 4 != 0 or image_size[1] % 4 != 0:
            raise ValueError(
                "Both image dimensions must be divisible by 4."
            )

        self.output_width = image_size[0] // 4
        self.output_height = image_size[1] // 4

        print(f"Loaded dataset index: {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def _read_points(
        self,
        annotation_path,
        original_width,
        original_height,
    ):
        points = []

        with annotation_path.open(
            "r", encoding="utf-8-sig"
        ) as file:
            for line in file:
                fields = [
                    value.strip()
                    for value in line.strip().split(",")
                ]

                # Ignore trailing commas in some annotation rows.
                while fields and fields[-1] == "":
                    fields.pop()

                if len(fields) != 8:
                    continue

                try:
                    x, y, width, height = map(float, fields[:4])
                    category = int(fields[5])
                except ValueError:
                    continue

                # VisDrone category 1: pedestrian.
                if category != 1:
                    continue

                if width <= 0 or height <= 0:
                    continue

                # Center in original image coordinates.
                center_x = x + width / 2.0
                center_y = y + height / 2.0

                # Scale to the resized image coordinates.
                resized_x = (
                    center_x * self.image_size[0] / original_width
                )
                resized_y = (
                    center_y * self.image_size[1] / original_height
                )

                if not (
                    0 <= resized_x < self.image_size[0]
                    and 0 <= resized_y < self.image_size[1]
                ):
                    continue

                points.append((resized_x, resized_y))

        return points

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        annotation_path = (
            self.annotation_dir / f"{image_path.stem}.txt"
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size

            resized_image = image.resize(
                self.image_size,
                Image.Resampling.BILINEAR,
            )

            image_array = (
                np.asarray(resized_image, dtype=np.float32) / 255.0
            )

        points = self._read_points(
            annotation_path,
            original_width,
            original_height,
        )

        # Build target density directly at output resolution.
        density = np.zeros(
            (self.output_height, self.output_width),
            dtype=np.float32,
        )

        for center_x, center_y in points:
            grid_x = center_x / 4.0
            grid_y = center_y / 4.0

            pixel_x = min(
                int(grid_x), self.output_width - 1
            )
            pixel_y = min(
                int(grid_y), self.output_height - 1
            )

            density[pixel_y, pixel_x] += 1.0

        # Spread each point into a Gaussian while preserving
        # the total number of pedestrians.
        if points:
            density = gaussian_filter(
                density,
                sigma=self.sigma,
            )

            density_sum = float(density.sum())

            if density_sum > 0:
                density *= len(points) / density_sum

        image_tensor = torch.from_numpy(
            image_array
        ).permute(2, 0, 1)

        density_tensor = torch.from_numpy(
            density.copy()
        ).unsqueeze(0)

        count_tensor = torch.tensor(
            float(len(points)),
            dtype=torch.float32,
        )

        return image_tensor, density_tensor, count_tensor
