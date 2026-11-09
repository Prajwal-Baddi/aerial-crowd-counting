
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from dataset_loader import VisDroneCrowdDataset
from model import CrowdCounter


# ============================================================
# CONFIGURATION
# ============================================================

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
CHECKPOINT_PATH = Path("baseline_crowdcounter.pth")

torch.set_num_threads(2)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print("Training device:", device)


# ============================================================
# DATASET
# ============================================================

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


# ============================================================
# MODEL
# ============================================================

model = CrowdCounter().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_val_mae = float("inf")


# ============================================================
# TRAINING
# ============================================================

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

        # Predictions and targets should both be [B, 1, 80, 80].
        if predictions.shape != targets.shape:
            raise RuntimeError(
                f"Prediction shape {predictions.shape} "
                f"does not match target shape {targets.shape}"
            )

        density_loss = F.mse_loss(
            predictions,
            targets,
        )

        predicted_counts = predictions.sum(dim=(1, 2, 3))

        count_loss = F.l1_loss(
            predicted_counts,
            counts,
        )

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

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_density_loss += density_loss.item() * batch_size
        total_count_loss += count_loss.item() * batch_size
        train_samples += batch_size

    train_loss = total_loss / train_samples
    train_density_loss = total_density_loss / train_samples
    train_count_loss = total_count_loss / train_samples


   
    # ========================================================
    # VALIDATION WITH CROWD-DENSITY BREAKDOWN
    # ========================================================

    model.eval()

    absolute_error = 0.0
    squared_error = 0.0
    val_samples = 0

    # Buckets are based on ground-truth pedestrian counts.
    buckets = {
        "Low (0-10)": {"abs": 0.0, "sq": 0.0, "n": 0},
        "Medium (11-30)": {"abs": 0.0, "sq": 0.0, "n": 0},
        "High (>30)": {"abs": 0.0, "sq": 0.0, "n": 0},
    }

    predicted_count_sum = 0.0
    target_count_sum = 0.0

    with torch.no_grad():
        for images, targets, counts in val_loader:
            images = images.to(device)
            counts = counts.to(device)

            predictions = model(images)
            predicted_counts = predictions.sum(dim=(1, 2, 3))

            errors = predicted_counts - counts

            absolute_error += errors.abs().sum().item()
            squared_error += errors.square().sum().item()
            predicted_count_sum += predicted_counts.sum().item()
            target_count_sum += counts.sum().item()
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

    val_mae = absolute_error / val_samples
    val_rmse = (squared_error / val_samples) ** 0.5

    mean_predicted_count = predicted_count_sum / val_samples
    mean_target_count = target_count_sum / val_samples

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"  Combined train loss: {train_loss:.6f}")
    print(f"  Density loss:        {train_density_loss:.6f}")
    print(f"  Count loss:          {train_count_loss:.6f}")
    print(f"  Validation MAE:      {val_mae:.3f}")
    print(f"  Validation RMSE:     {val_rmse:.3f}")
    print(f"  Mean predicted count:{mean_predicted_count:8.3f}")
    print(f"  Mean target count:   {mean_target_count:8.3f}")

    print("  MAE by crowd-density range:")
    for name, stats in buckets.items():
        if stats["n"] > 0:
            bucket_mae = stats["abs"] / stats["n"]
            bucket_rmse = (stats["sq"] / stats["n"]) ** 0.5

            print(
                f"    {name}: MAE={bucket_mae:.3f}, "
                f"RMSE={bucket_rmse:.3f}, "
                f"samples={stats['n']}"
            )
        else:
            print(f"    {name}: no samples")

    # Save the best model for this experiment.
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
            },
            CHECKPOINT_PATH,
        )

        print(f"  Saved best checkpoint: {CHECKPOINT_PATH}")

