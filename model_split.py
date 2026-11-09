import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset_loader import VisDroneCrowdDataset
from model import CrowdCounter

SEED = 42
DATASET_ROOT = "dataset/VisDrone2019-DET-train"
CHECKPOINT = "residual_phase2_coarse.pth"
SPLIT_FILE = "residual_phase2_split.json"
BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-4
COUNT_LOSS_WEIGHT = 0.05
DROPOUT_RATE = 0.25
NUM_WORKERS = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_split(dataset_size):
    """65% coarse-train, 5% coarse-selection, 17% refiner-train,
    3% refiner-validation, 10% final test. No image overlaps."""
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(dataset_size).tolist()

    n_coarse_train = int(0.65 * dataset_size)
    n_coarse_val = int(0.05 * dataset_size)
    n_refiner_train = int(0.17 * dataset_size)
    n_refiner_val = int(0.03 * dataset_size)

    a = n_coarse_train
    b = a + n_coarse_val
    c = b + n_refiner_train
    d = c + n_refiner_val

    split = {
        "seed": SEED,
        "dataset_size": dataset_size,
        "coarse_train": indices[:a],
        "coarse_validation": indices[a:b],
        "refiner_train": indices[b:c],
        "refiner_validation": indices[c:d],
        "final_test": indices[d:],
    }

    # Fail early if the partitions overlap or omit samples.
    groups = [set(split[k]) for k in (
        "coarse_train", "coarse_validation", "refiner_train",
        "refiner_validation", "final_test"
    )]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if groups[i] & groups[j]:
                raise RuntimeError("Split partitions overlap.")
    if len(set.union(*groups)) != dataset_size:
        raise RuntimeError("Split does not cover the full dataset.")
    return split


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    errors = []
    for images, _, counts in loader:
        images = images.to(device, non_blocking=True)
        counts = counts.to(device, non_blocking=True).view(-1)
        predictions = model(images)
        predicted_counts = predictions.sum(dim=(1, 2, 3))
        errors.extend((predicted_counts - counts).cpu().tolist())
    errors = np.asarray(errors, dtype=np.float64)
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = VisDroneCrowdDataset(root=DATASET_ROOT, image_size=(320, 320))
    split = create_split(len(dataset))
    Path(SPLIT_FILE).write_text(json.dumps(split, indent=2), encoding="utf-8")

    train_loader = DataLoader(
        Subset(dataset, split["coarse_train"]),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    selection_loader = DataLoader(
        Subset(dataset, split["coarse_validation"]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = CrowdCounter(dropout_rate=DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_mae = float("inf")

    print(
        "Split sizes:",
        {k: len(split[k]) for k in (
            "coarse_train", "coarse_validation", "refiner_train",
            "refiner_validation", "final_test"
        )}
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_samples = 0
        start = time.time()

        for images, densities, counts in train_loader:
            images = images.to(device, non_blocking=True)
            densities = densities.to(device, non_blocking=True)
            counts = counts.to(device, non_blocking=True).view(-1)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            density_loss = F.mse_loss(predictions, densities)
            predicted_counts = predictions.sum(dim=(1, 2, 3))
            count_loss = F.l1_loss(predicted_counts, counts)
            loss = density_loss + COUNT_LOSS_WEIGHT * count_loss
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        val_mae, val_rmse = evaluate(model, selection_loader, device)
        train_loss = total_loss / max(total_samples, 1)
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"loss={train_loss:.5f} | selection MAE={val_mae:.4f} | "
            f"RMSE={val_rmse:.4f} | time={time.time()-start:.1f}s"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mae": best_mae,
                "val_rmse": val_rmse,
                "dropout_rate": DROPOUT_RATE,
                "architecture": "multiscale_mc_dropout_phase2",
                "seed": SEED,
            }, CHECKPOINT)
            print(f"Saved best coarse checkpoint: {CHECKPOINT}")

    print(f"Done. Best coarse-selection MAE: {best_mae:.4f}")
    print(f"Split file: {SPLIT_FILE}")


if __name__ == "__main__":
    main()
