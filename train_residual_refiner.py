import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import VisDroneCrowdDataset
from residual_data import ResidualPatchDataset
from residual_model import ResidualPatchRefiner

SEED = 42
DATASET_ROOT = "dataset/VisDrone2019-DET-train"
SPLIT_FILE = "residual_phase2_split.json"
CACHE_FILE = "residual_phase2_cache.pt"
BEST_CHECKPOINT = "residual_patch_refiner_best.pth"
LAST_CHECKPOINT = "residual_patch_refiner_last.pth"

BATCH_SIZE = 8
EPOCHS = 12
PATCHES_PER_IMAGE = 16
LEARNING_RATE = 1e-4
COUNT_LOSS_WEIGHT = 0.05
NUM_WORKERS = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_file(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_epoch(model, loader, optimizer, device, train):
    model.train(train)
    total_loss = total_density = total_count = 0.0
    total_abs_patch_error = total_sq_patch_error = 0.0
    total_abs_coarse_error = 0.0
    total_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, coarse, targets in loader:
            images = images.to(device, non_blocking=True)
            coarse = coarse.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            residual = model(images, coarse)
            refined = torch.clamp(coarse + residual, min=0.0)

            density_loss = nn.functional.mse_loss(refined, targets)
            refined_counts = refined.sum(dim=(1, 2, 3))
            target_counts = targets.sum(dim=(1, 2, 3))
            count_loss = nn.functional.l1_loss(refined_counts, target_counts)
            loss = density_loss + COUNT_LOSS_WEIGHT * count_loss

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            errors = refined_counts - target_counts
            coarse_counts = coarse.sum(dim=(1, 2, 3))
            coarse_errors = coarse_counts - target_counts
            batch = images.size(0)
            total_loss += loss.item() * batch
            total_density += density_loss.item() * batch
            total_count += count_loss.item() * batch
            total_abs_patch_error += errors.abs().sum().item()
            total_sq_patch_error += errors.square().sum().item()
            total_abs_coarse_error += coarse_errors.abs().sum().item()
            total_samples += batch

    n = max(total_samples, 1)
    return {
        "loss": total_loss / n,
        "density_loss": total_density / n,
        "count_loss": total_count / n,
        "coarse_patch_mae": total_abs_coarse_error / n,
        "refined_patch_mae": total_abs_patch_error / n,
        "patch_rmse": (total_sq_patch_error / n) ** 0.5,
    }


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    split = json.loads(Path(SPLIT_FILE).read_text(encoding="utf-8"))
    cache_file = load_file(CACHE_FILE, "cpu")
    cache = cache_file["cache"]
    dataset = VisDroneCrowdDataset(root=DATASET_ROOT, image_size=(320, 320))

    train_data = ResidualPatchDataset(
        dataset, split["refiner_train"], cache,
        patches_per_image=PATCHES_PER_IMAGE, seed=SEED,
    )
    val_data = ResidualPatchDataset(
        dataset, split["refiner_validation"], cache,
        patches_per_image=64, seed=SEED + 100_000,
    )

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    print(f"Refiner training images: {len(split['refiner_train'])}")
    print(f"Refiner validation images: {len(split['refiner_validation'])}")
    print(f"Training patches per epoch: {len(train_data)}")
    print(f"Validation patches: {len(val_data)}")

    model = ResidualPatchRefiner().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_mae = float("inf")

    for epoch in range(EPOCHS):
        train_data.set_epoch(epoch)
        start = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, True)
        val_metrics = run_epoch(model, val_loader, None, device, False)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"train loss={train_metrics['loss']:.6f} | "
            f"val coarse patch MAE={val_metrics['coarse_patch_mae']:.6f} | "
            f"val refined patch MAE={val_metrics['refined_patch_mae']:.6f} | "
            f"val patch RMSE={val_metrics['patch_rmse']:.6f} | "
            f"time={time.time()-start:.1f}s"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_coarse_patch_mae": val_metrics["coarse_patch_mae"],
            "validation_refined_patch_mae": val_metrics["refined_patch_mae"],
            "validation_patch_rmse": val_metrics["patch_rmse"],
            "architecture": "residual_patch_refiner_4ch_signed",
            "seed": SEED,
            "split_file": SPLIT_FILE,
            "cache_file": CACHE_FILE,
        }
        torch.save(checkpoint, LAST_CHECKPOINT)

        if val_metrics["refined_patch_mae"] < best_mae:
            best_mae = val_metrics["refined_patch_mae"]
            torch.save(checkpoint, BEST_CHECKPOINT)
            print(f"Saved best checkpoint: {BEST_CHECKPOINT}")

    print(f"Training complete. Best validation patch MAE: {best_mae:.6f}")
    print("Reminder: patch MAE is not full-image MAE. Use the test evaluator.")


if __name__ == "__main__":
    main()
