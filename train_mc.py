
import os
import time
import random

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split

from dataset_loader import VisDroneCrowdDataset
from model import CrowdCounter


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = (320, 320)
DENSITY_SIZE = (80, 80)

BATCH_SIZE = 2
EPOCHS = 10

LEARNING_RATE = 1e-4
COUNT_LOSS_WEIGHT = 0.05

DROPOUT_RATE = 0.25

CHECKPOINT_PATH = "mc_multiscale_crowdcounter.pth"

NUM_WORKERS = 0  # Recommended starting point on Windows

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_num_threads(2)


# ============================================================
# GPU INFORMATION
# ============================================================

print("=" * 60)
print("MC DROPOUT MULTI-SCALE CROWD COUNTING")
print("=" * 60)

print("Using device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )


# ============================================================
# DATASET
# ============================================================

dataset = VisDroneCrowdDataset(
    root="dataset/VisDrone2019-DET-train",
    image_size=IMAGE_SIZE
)

total_size = len(dataset)

train_size = int(0.8 * total_size)
val_size = total_size - train_size

generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

print("\nDataset information")
print("Total images:", total_size)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Batch size:", BATCH_SIZE)


# ============================================================
# MODEL
# ============================================================

model = CrowdCounter(
    dropout_rate=DROPOUT_RATE
).to(DEVICE)

# Verify output dimensions
dummy_input = torch.randn(
    1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]
).to(DEVICE)

model.eval()

with torch.no_grad():
    dummy_output = model(dummy_input)

print("\nModel output shape:", tuple(dummy_output.shape))

assert dummy_output.shape == (
    1, 1, DENSITY_SIZE[0], DENSITY_SIZE[1]
), "Incorrect model output shape"

del dummy_input, dummy_output

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_density_loss = 0.0
    total_count_loss = 0.0

    for images, targets, counts in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        counts = counts.to(
            DEVICE,
            non_blocking=True
        ).float().view(-1)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(images)

        density_loss = F.mse_loss(
            predictions,
            targets
        )

        predicted_counts = predictions.sum(
            dim=(1, 2, 3)
        )

        count_loss = F.l1_loss(
            predicted_counts,
            counts
        )

        loss = (
            density_loss
            + COUNT_LOSS_WEIGHT * count_loss
        )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_density_loss += density_loss.item() * batch_size
        total_count_loss += count_loss.item() * batch_size

    n = len(train_dataset)

    return (
        total_loss / n,
        total_density_loss / n,
        total_count_loss / n
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    absolute_errors = []
    squared_errors = []

    for images, targets, counts in val_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        counts = counts.to(
            DEVICE,
            non_blocking=True
        ).float().view(-1)

        predictions = model(images)

        predicted_counts = predictions.sum(
            dim=(1, 2, 3)
        )

        errors = predicted_counts - counts

        absolute_errors.extend(
            errors.abs().cpu().tolist()
        )

        squared_errors.extend(
            errors.square().cpu().tolist()
        )

    mae = float(np.mean(absolute_errors))
    rmse = float(np.sqrt(np.mean(squared_errors)))

    return mae, rmse


# ============================================================
# TRAINING LOOP
# ============================================================

best_mae = float("inf")

print("\nStarting GPU training...\n")

for epoch in range(EPOCHS):

    start_time = time.time()

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_loss, density_loss, count_loss = train_one_epoch()

    val_mae, val_rmse = validate()

    elapsed = time.time() - start_time

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

    print(f"Training Loss: {train_loss:.4f}")
    print(f"Density Loss:  {density_loss:.4f}")
    print(f"Count Loss:    {count_loss:.4f}")

    print(f"Validation MAE:  {val_mae:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")

    print(f"Time: {elapsed:.2f} seconds")

    if DEVICE.type == "cuda":

        peak_memory = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )

        print(f"Peak GPU Memory: {peak_memory:.2f} GB")

    # Save the best model
    if val_mae < best_mae:

        best_mae = val_mae

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mae": best_mae,
                "val_rmse": val_rmse,
                "dropout_rate": DROPOUT_RATE,
                "architecture": "multiscale_mc_dropout"
            },
            CHECKPOINT_PATH
        )

        print("Best checkpoint saved.")

    print("-" * 60)


print("\nTraining completed.")
print("Best validation MAE:", round(best_mae, 4))
print("Checkpoint:", CHECKPOINT_PATH)
