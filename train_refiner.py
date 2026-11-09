
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset_loader import VisDroneCrowdDataset
from patch_refiner import PatchRefiner
from refinement_data import VisDronePatchDataset


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATASET_ROOT = "dataset/VisDrone2019-DET-train"

BATCH_SIZE = 8
EPOCHS = 8
PATCHES_PER_IMAGE = 8
LEARNING_RATE = 1e-4
COUNT_LOSS_WEIGHT = 0.05

BEST_CHECKPOINT = "patch_refiner_validated.pth"
LAST_CHECKPOINT = "patch_refiner_last.pth"

NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    density_criterion,
    count_criterion,
    device,
):
    model.train()

    total_loss = 0.0
    total_density_loss = 0.0
    total_count_loss = 0.0
    total_samples = 0

    for images, densities, counts in loader:
        images = images.to(device, non_blocking=True)
        densities = densities.to(device, non_blocking=True)
        counts = counts.to(device, non_blocking=True).view(-1)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(images)

        predicted_counts = predictions.sum(dim=(1, 2, 3))

        density_loss = density_criterion(predictions, densities)
        count_loss = count_criterion(predicted_counts, counts)

        loss = density_loss + COUNT_LOSS_WEIGHT * count_loss

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_density_loss += density_loss.item() * batch_size
        total_count_loss += count_loss.item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "density_loss": total_density_loss / total_samples,
        "count_loss": total_count_loss / total_samples,
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    density_criterion,
    count_criterion,
    device,
):
    model.eval()

    total_density_loss = 0.0
    total_count_loss = 0.0
    total_absolute_error = 0.0
    total_squared_error = 0.0
    total_samples = 0

    for images, densities, counts in loader:
        images = images.to(device, non_blocking=True)
        densities = densities.to(device, non_blocking=True)
        counts = counts.to(device, non_blocking=True).view(-1)

        predictions = model(images)

        predicted_counts = predictions.sum(dim=(1, 2, 3))

        density_loss = density_criterion(predictions, densities)
        count_loss = count_criterion(predicted_counts, counts)

        absolute_errors = torch.abs(predicted_counts - counts)
        squared_errors = (predicted_counts - counts) ** 2

        batch_size = images.size(0)

        total_density_loss += density_loss.item() * batch_size
        total_count_loss += count_loss.item() * batch_size
        total_absolute_error += absolute_errors.sum().item()
        total_squared_error += squared_errors.sum().item()
        total_samples += batch_size

    validation_mae = total_absolute_error / total_samples
    validation_rmse = (total_squared_error / total_samples) ** 0.5

    return {
        "density_loss": total_density_loss / total_samples,
        "count_loss": total_count_loss / total_samples,
        "mae": validation_mae,
        "rmse": validation_rmse,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TRAINING PATCH REFINER WITH VALIDATION")
    print("=" * 60)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Load the original dataset
    # --------------------------------------------------------

    base_dataset = VisDroneCrowdDataset(
        root=DATASET_ROOT,
        image_size=(320, 320),
    )

    print(f"Loaded dataset: {len(base_dataset)} images")

    # Reproduce the same 80/20 split used by the coarse model
    train_size = int(0.8 * len(base_dataset))
    validation_size = len(base_dataset) - train_size

    train_subset, validation_subset = random_split(
        base_dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    print(f"Training images: {len(train_subset)}")
    print(f"Validation images: {len(validation_subset)}")

    # --------------------------------------------------------
    # Create patch datasets
    # --------------------------------------------------------

    train_patch_dataset = VisDronePatchDataset(
        base_dataset=base_dataset,
        image_indices=train_subset.indices,
        patches_per_image=PATCHES_PER_IMAGE,
        crop_size=160,
        sigma=1.0,
        seed=SEED,
    )

    validation_patch_dataset = VisDronePatchDataset(
        base_dataset=base_dataset,
        image_indices=validation_subset.indices,
        patches_per_image=PATCHES_PER_IMAGE,
        crop_size=160,
        sigma=1.0,
        seed=SEED + 100000,
    )

    # Keep validation patches identical across epochs
    validation_patch_dataset.set_epoch(0)

    train_loader = DataLoader(
        train_patch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    validation_loader = DataLoader(
        validation_patch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Training patches per epoch: {len(train_patch_dataset)}")
    print(f"Validation patches: {len(validation_patch_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")

    # --------------------------------------------------------
    # Initialize model
    # --------------------------------------------------------

    model = PatchRefiner().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    density_criterion = nn.MSELoss()
    count_criterion = nn.L1Loss()

    best_validation_mae = float("inf")

    print("\nStarting training...")
    print("Checkpoint selection: lowest validation patch-count MAE")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(EPOCHS):
        print("\n" + "=" * 60)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print("=" * 60)

        # Change training patch samples each epoch
        train_patch_dataset.set_epoch(epoch)

        start_time = time.time()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            density_criterion,
            count_criterion,
            device,
        )

        validation_metrics = validate(
            model,
            validation_loader,
            density_criterion,
            count_criterion,
            device,
        )

        elapsed = time.time() - start_time

        print("\nTRAINING")
        print(f"Loss:          {train_metrics['loss']:.6f}")
        print(f"Density Loss:  {train_metrics['density_loss']:.6f}")
        print(f"Count Loss:    {train_metrics['count_loss']:.6f}")

        print("\nVALIDATION")
        print(f"Density Loss:  {validation_metrics['density_loss']:.6f}")
        print(f"Count Loss:    {validation_metrics['count_loss']:.6f}")
        print(f"Patch Count MAE:  {validation_metrics['mae']:.6f}")
        print(f"Patch Count RMSE: {validation_metrics['rmse']:.6f}")

        print(f"\nEpoch time: {elapsed:.2f} seconds")

        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f"Peak GPU memory: {peak_memory:.2f} GB")

        # ----------------------------------------------------
        # Save best model using VALIDATION MAE
        # ----------------------------------------------------

        if validation_metrics["mae"] < best_validation_mae:
            best_validation_mae = validation_metrics["mae"]

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_validation_mae": best_validation_mae,
                    "validation_rmse": validation_metrics["rmse"],
                    "validation_density_loss": validation_metrics["density_loss"],
                    "validation_count_loss": validation_metrics["count_loss"],
                    "dropout_rate": 0.0,
                    "architecture": "patch_refiner",
                    "seed": SEED,
                    "patches_per_image": PATCHES_PER_IMAGE,
                },
                BEST_CHECKPOINT,
            )

            print(f"\nBest validation checkpoint saved: {BEST_CHECKPOINT}")

        # Save the latest model separately
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_mae": validation_metrics["mae"],
                "architecture": "patch_refiner",
            },
            LAST_CHECKPOINT,
        )

        print("-" * 60)

    print("\n" + "=" * 60)
    print("PATCH REFINER TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best validation patch-count MAE: {best_validation_mae:.6f}")
    print(f"Best checkpoint: {BEST_CHECKPOINT}")
    print(f"Latest checkpoint: {LAST_CHECKPOINT}")


if __name__ == "__main__":
    main()
