
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


class VisDronePatchDataset(Dataset):
    """
    Samples original-resolution crops from VisDrone images.

    Each image is divided into an 8 x 8 grid.
    Each crop produces a 10 x 10 density target.
    """

    def __init__(
        self,
        base_dataset,
        image_indices,
        patches_per_image=8,
        crop_size=160,
        sigma=1.0,
        seed=42,
    ):
        self.base_dataset = base_dataset
        self.image_indices = list(image_indices)
        self.patches_per_image = patches_per_image
        self.crop_size = crop_size
        self.sigma = sigma
        self.seed = seed
        self.epoch = 0

        self.patch_grid = 8
        self.target_size = 10

    def __len__(self):
        return len(self.image_indices) * self.patches_per_image

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _read_points(self, annotation_path):
        points = []

        with annotation_path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                fields = [v.strip() for v in line.strip().split(",")]

                while fields and fields[-1] == "":
                    fields.pop()

                if len(fields) != 8:
                    continue

                try:
                    x, y, width, height = map(float, fields[:4])
                    category = int(fields[5])
                except ValueError:
                    continue

                if category != 1 or width <= 0 or height <= 0:
                    continue

                center_x = x + width / 2.0
                center_y = y + height / 2.0

                points.append((center_x, center_y))

        return points

    def __getitem__(self, index):
        image_slot = index // self.patches_per_image
        sample_slot = index % self.patches_per_image

        image_index = self.image_indices[image_slot]
        image_path = self.base_dataset.image_paths[image_index]

        annotation_path = (
            self.base_dataset.annotation_dir / f"{image_path.stem}.txt"
        )

        rng = random.Random(
            self.seed
            + self.epoch * 1_000_003
            + image_index * 101
            + sample_slot
        )

        patch_index = rng.randrange(64)
        patch_row = patch_index // 8
        patch_col = patch_index % 8

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size

            left = round(patch_col * original_width / 8)
            right = round((patch_col + 1) * original_width / 8)
            top = round(patch_row * original_height / 8)
            bottom = round((patch_row + 1) * original_height / 8)

            crop = image.crop((left, top, right, bottom))
            crop = crop.resize(
                (self.crop_size, self.crop_size),
                Image.Resampling.BILINEAR,
            )

            image_array = np.asarray(crop, dtype=np.float32) / 255.0

        points = self._read_points(annotation_path)

        crop_width = right - left
        crop_height = bottom - top

        density = np.zeros(
            (self.target_size, self.target_size),
            dtype=np.float32,
        )

        patch_count = 0

        for center_x, center_y in points:
            if not (left <= center_x < right and top <= center_y < bottom):
                continue

            local_x = center_x - left
            local_y = center_y - top

            grid_x = local_x * self.target_size / crop_width
            grid_y = local_y * self.target_size / crop_height

            pixel_x = min(int(grid_x), self.target_size - 1)
            pixel_y = min(int(grid_y), self.target_size - 1)

            density[pixel_y, pixel_x] += 1.0
            patch_count += 1

        if patch_count > 0:
            density = gaussian_filter(density, sigma=self.sigma)
            density_sum = float(density.sum())

            if density_sum > 0:
                density *= patch_count / density_sum

        image_tensor = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .contiguous()
        )

        density_tensor = torch.from_numpy(density.copy()).unsqueeze(0)
        count_tensor = torch.tensor(float(patch_count), dtype=torch.float32)

        return image_tensor, density_tensor, count_tensor
