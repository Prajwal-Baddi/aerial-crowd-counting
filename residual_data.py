import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ResidualPatchDataset(Dataset):
    """
    Patch dataset built from aligned full-image maps.

    The target is sliced from the full-image ground-truth map. It is NOT
    regenerated independently per crop, so Gaussian mass at patch borders
    remains consistent with the coarse map.
    """
    def __init__(
        self,
        base_dataset,
        image_indices,
        cache,
        patches_per_image=16,
        crop_size=160,
        seed=42,
    ):
        self.base_dataset = base_dataset
        self.image_indices = list(image_indices)
        self.cache = cache
        self.patches_per_image = int(patches_per_image)
        self.crop_size = int(crop_size)
        self.seed = int(seed)
        self.epoch = 0
        self.grid_size = 8
        self.patch_size = 10

        if not 1 <= self.patches_per_image <= 64:
            raise ValueError("patches_per_image must be between 1 and 64.")
        missing = [i for i in self.image_indices if str(i) not in self.cache]
        if missing:
            raise KeyError(f"Cache is missing {len(missing)} image indices.")

    def __len__(self):
        return len(self.image_indices) * self.patches_per_image

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _patch_index(self, image_index, slot):
        # Sample distinct patches within an image for each epoch.
        # With patches_per_image=64, validation visits all 64 patches once.
        rng = random.Random(
            self.seed + self.epoch * 1_000_003 + image_index * 101
        )
        selected = rng.sample(range(64), k=self.patches_per_image)
        return selected[slot]

    def __getitem__(self, item):
        image_slot = item // self.patches_per_image
        sample_slot = item % self.patches_per_image
        image_index = self.image_indices[image_slot]
        patch_index = self._patch_index(image_index, sample_slot)
        row, col = divmod(patch_index, 8)

        image_path = self.base_dataset.image_paths[image_index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            left = round(col * width / 8)
            right = round((col + 1) * width / 8)
            top = round(row * height / 8)
            bottom = round((row + 1) * height / 8)
            crop = image.crop((left, top, right, bottom))
            crop = crop.resize(
                (self.crop_size, self.crop_size), Image.Resampling.BILINEAR
            )
            image_array = np.asarray(crop, dtype=np.float32) / 255.0

        entry = self.cache[str(image_index)]
        y0, y1 = row * 10, (row + 1) * 10
        x0, x1 = col * 10, (col + 1) * 10
        coarse_patch = entry["coarse"][:, y0:y1, x0:x1].clone()
        target_patch = entry["target"][:, y0:y1, x0:x1].clone()

        if coarse_patch.shape != (1, 10, 10) or target_patch.shape != (1, 10, 10):
            raise RuntimeError("Incorrect density patch dimensions.")

        image_tensor = (
            torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        )
        return image_tensor, coarse_patch, target_patch
