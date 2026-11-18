import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Density maps sum to pedestrian counts. Stage 2 uses count units directly.
DEN_SCALE = 1.0


class ErrorAwareCrowdDataset(Dataset):
    def __init__(
        self, manifest_path, dataset_root="dataset/VisDrone2019-DET-train",
        crop_size=512, density_stride=8, gaussian_sigma=3.0,
        return_metadata=False,
    ):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.crop_size = crop_size
        self.density_stride = density_stride
        self.gaussian_sigma = gaussian_sigma
        self.return_metadata = return_metadata

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        if crop_size % density_stride:
            raise ValueError("crop_size must be divisible by density_stride")
        if gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")

        self.density_size = crop_size // density_stride
        self.records = []
        with self.manifest_path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
                self.records.append(record)

        if not self.records:
            raise RuntimeError(f"No crop records found in {self.manifest_path}")

        for i, record in enumerate(self.records):
            if record["crop_size"] != crop_size:
                raise ValueError(f"Crop size mismatch at record {i}")
            if record["density_stride"] != density_stride:
                raise ValueError(f"Density stride mismatch at record {i}")
            if len(record["points"]) != record["count"]:
                raise ValueError(f"Point/count mismatch at record {i}")

    def __len__(self):
        return len(self.records)

    def _load_image_crop(self, record):
        path = self.dataset_root / record["image"]
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        x0, y0 = int(record["crop_x"]), int(record["crop_y"])
        with Image.open(path) as image:
            crop = image.convert("RGB").crop(
                (x0, y0, x0 + self.crop_size, y0 + self.crop_size)
            )
        if crop.size != (self.crop_size, self.crop_size):
            raise RuntimeError(f"Unexpected crop dimensions: {crop.size}")
        array = np.asarray(crop, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1).copy())

    def _generate_density_map(self, points):
        density = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        sigma = self.gaussian_sigma
        radius = int(np.ceil(3.0 * sigma))

        for point in points:
            cx, cy = float(point[0]), float(point[1])
            if not (0 <= cx < self.crop_size and 0 <= cy < self.crop_size):
                raise ValueError(f"Point outside crop boundaries: {point}")
            x0, x1 = max(0, int(np.floor(cx))-radius), min(
                self.crop_size, int(np.floor(cx))+radius+1
            )
            y0, y1 = max(0, int(np.floor(cy))-radius), min(
                self.crop_size, int(np.floor(cy))+radius+1
            )
            xs = np.arange(x0, x1, dtype=np.float32)
            ys = np.arange(y0, y1, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            kernel = np.exp(-((gx-cx)**2 + (gy-cy)**2)/(2.0*sigma**2))
            total = float(kernel.sum())
            if total <= 0:
                raise RuntimeError("Gaussian normalization failed")
            density[y0:y1, x0:x1] += kernel / total

        stride, size = self.density_stride, self.density_size
        density = density.reshape(size, stride, size, stride).sum(axis=(1, 3))
        return torch.from_numpy(density.copy()).unsqueeze(0)

    def __getitem__(self, index):
        record = self.records[index]
        image = self._load_image_crop(record)
        density = self._generate_density_map(record["points"])
        count = torch.tensor(float(record["count"]), dtype=torch.float32)
        if self.return_metadata:
            meta = {
                "image": record["image"], "crop_x": record["crop_x"],
                "crop_y": record["crop_y"], "split": record["split"],
            }
            return image, density, count, meta
        return image, density, count


def get_loader(
    data_dir="error_aware/processed", split="train", batch_size=1,
    shuffle=None, num_workers=0,
    dataset_root="dataset/VisDrone2019-DET-train", return_metadata=False,
):
    files = {
        "train": "train_balanced.jsonl",
        "validation": "validation.jsonl",
        "val": "validation.jsonl",
        "test": "test.jsonl",
    }
    if split not in files:
        raise ValueError(f"Unknown split {split!r}; choose {list(files)}")
    if shuffle is None:
        shuffle = split == "train"
    dataset = ErrorAwareCrowdDataset(
        Path(data_dir) / files[split], dataset_root=dataset_root,
        crop_size=512, density_stride=8, gaussian_sigma=3.0,
        return_metadata=return_metadata,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=False, persistent_workers=(num_workers > 0),
    )
    return loader
