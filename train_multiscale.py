
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from dataset_loader import VisDroneCrowdDataset
from model import CrowdCounter


SEED = 42
IMAGE_SIZE = (320, 320)

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4
COUNT_LOSS_WEIGHT = 0.05
NUM_WORKERS = 0

SMOKE_TEST = False
SMOKE_TRAIN_SAMPLES = 128
SMOKE_VAL_SAMPLES = 32

DATASET_ROOT = "dataset/VisDrone2019-DET-train"
CHECKPOINT_PATH = Path("multiscale_crowdcounter.pth")

torch.set_num_threads(2)
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print("Training device:", device)


# ------------------------------------------------------------
# Dataset and reproducible split
# ------------------------------------------------------------

dataset = VisDroneCrowdDataset(
    root=DATASET_ROOT,
    image_size=IMAGE_SIZE,
)

generator = torch.Generator().manual_seed(SEED)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(
    dataset,
    [train_size, val_size],
    generator=generator,
)

if SMOKE_TEST:
    train_set = Subset(
        train_set,
        range(min(SMOKE_TRAIN_SAMPLES, len(train_set))),
    )
    val_set = Subset(
        val_set,
        range(min(SMOKE_VAL_SAMPLES, len(val_set))),
    )

print("Training images:", len(train_set))
print("Validation images:", len(val_set))

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ------------------------------------------------------------
# Model and optimizer
# ------------------------------------------------------------

model = CrowdCounter().to(device)

# Verify the output dimensions before starting training.
with torch.no_grad():
    dummy = torch.zeros(1, 3, *IMAGE_SIZE)
    output = model(dummy)

assert output.shape == (1, 1, 80, 80), output.shape
print("Model output shape:", tuple(output.shape))

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_val_mae = float("inf")


# ------------------------------------------------------------
# Training and validation
# ------------------------------------------------------------

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    total_density_loss = 0.0
    total_count_loss = 0.0
    train_samples = 0

    for images, targets, counts in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        counts = counts.to(device)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(images)

        if predictions.shape != targets.shape:
            raise RuntimeError(
                f"Prediction shape {predictions.shape} "
                f"does not match target shape {targets.shape}"
            )

        density_loss = F.mse_loss(predictions, targets)

        predicted_counts = predictions.sum(dim=(1, 2, 3))
        count_loss = F.l1_loss(predicted_counts, counts)

        loss = (
            density_loss
            + COUNT_LOSS_WEIGHT * count_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        n = images.size(0)
        total_loss += loss.item() * n
        total_density_loss += density_loss.item() * n
        total_count_loss += count_loss.item() * n
        train_samples += n

    train_loss = total_loss / train_samples
    train_density_loss = total_density_loss / train_samples
    train_count_loss = total_count_loss / train_samples

    # ------------------ Validation ------------------

    model.eval()

    abs_error = 0.0
    squared_error = 0.0
    predicted_sum = 0.0
    target_sum = 0.0
    val_samples = 0

    buckets = {
        "Low (0-10)": {"abs": 0.0, "sq": 0.0, "n": 0},
        "Medium (11-30)": {"abs": 0.0, "sq": 0.0, "n": 0},
        "High (>30)": {"abs": 0.0, "sq": 0.0, "n": 0},
    }

    with torch.no_grad():
        for images, targets, counts in val_loader:
            images = images.to(device)
            counts = counts.to(device)

            predictions = model(images)
            predicted_counts = predictions.sum(dim=(1, 2, 3))
            errors = predicted_counts - counts

            abs_error += errors.abs().sum().item()
            squared_error += errors.square().sum().item()
            predicted_sum += predicted_counts.sum().item()
            target_sum += counts.sum().item()
            val_samples += images.size(0)

            for gt, error in zip(counts, errors):
                gt_value = gt.item()
                error_value = error.item()

                if gt_value <= 10:
                    bucket = buckets["Low (0-10)"]
                elif gt_value <= 30:
                    bucket = buckets["Medium (11-30)"]
                else:
                    bucket = buckets["High (>30)"]

                bucket["abs"] += abs(error_value)
                bucket["sq"] += error_value ** 2
                bucket["n"] += 1

    val_mae = abs_error / val_samples
    val_rmse = (squared_error / val_samples) ** 0.5

    mean_prediction = predicted_sum / val_samples
    mean_target = target_sum / val_samples

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"  Combined train loss: {train_loss:.6f}")
    print(f"  Density loss:        {train_density_loss:.6f}")
    print(f"  Count loss:          {train_count_loss:.6f}")
    print(f"  Validation MAE:      {val_mae:.3f}")
    print(f"  Validation RMSE:     {val_rmse:.3f}")
    print(f"  Mean predicted count:{mean_prediction:8.3f}")
    print(f"  Mean target count:   {mean_target:8.3f}")

    print("  MAE by crowd-density range:")
    for name, stats in buckets.items():
        if stats["n"] == 0:
            print(f"    {name}: no samples")
            continue

        bucket_mae = stats["abs"] / stats["n"]
        bucket_rmse = (stats["sq"] / stats["n"]) ** 0.5

        print(
            f"    {name}: MAE={bucket_mae:.3f}, "
            f"RMSE={bucket_rmse:.3f}, "
            f"samples={stats['n']}"
        )

    if val_mae < best_val_mae:
        best_val_mae = val_mae

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "image_size": IMAGE_SIZE,
                "count_loss_weight": COUNT_LOSS_WEIGHT,
                "architecture": "multiscale_cnn",
            },
            CHECKPOINT_PATH,
        )

        print(f"  Saved best checkpoint: {CHECKPOINT_PATH}")

print("\nMulti-scale training finished.")
print(f"Best validation MAE: {best_val_mae:.3f}")
print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")
