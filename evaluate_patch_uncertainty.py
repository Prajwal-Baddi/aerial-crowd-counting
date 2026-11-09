
import csv
import math
import time

import numpy as np
import torch

from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, random_split

from dataset_loader import VisDroneCrowdDataset
from model import CrowdCounter
from uncertainty import enable_mc_dropout


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

CHECKPOINT_PATH = "mc_multiscale_crowdcounter.pth"

NUM_MC_SAMPLES = 20

PATCH_SIZE = 10
DENSITY_SIZE = 80

PATCHES_PER_SIDE = DENSITY_SIZE // PATCH_SIZE
PATCHES_PER_IMAGE = PATCHES_PER_SIDE ** 2

TOP_PERCENTAGE = 0.20

RANDOM_TRIALS = 100

OUTPUT_CSV = "patch_uncertainty_evaluation.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)


# ============================================================
# DATASET
# ============================================================

dataset = VisDroneCrowdDataset(
    root="dataset/VisDrone2019-DET-train",
    image_size=(320, 320)
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(SEED)

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model = CrowdCounter(
    dropout_rate=checkpoint.get("dropout_rate", 0.25)
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("=" * 65)
print("PATCH-LEVEL UNCERTAINTY EVALUATION")
print("=" * 65)

print("Device:", DEVICE)
print("Validation images:", len(val_dataset))
print("MC samples:", NUM_MC_SAMPLES)
print("Density map:", f"{DENSITY_SIZE} x {DENSITY_SIZE}")
print("Patch size:", f"{PATCH_SIZE} x {PATCH_SIZE}")
print("Patches per image:", PATCHES_PER_IMAGE)
print("Top uncertain patch percentage:", TOP_PERCENTAGE * 100)


# ============================================================
# PATCH-LEVEL EVALUATION
# ============================================================

all_results = []

image_error_coverages = []
image_random_coverages = []

image_selected_errors = []
image_random_errors = []

start_time = time.time()

for image_index, (image, target, gt_count) in enumerate(val_loader):

    image = image.to(
        DEVICE,
        non_blocking=True
    )

    target = target.to(
        DEVICE,
        non_blocking=True
    )

    # --------------------------------------------------------
    # Generate stochastic predictions
    # --------------------------------------------------------

    enable_mc_dropout(model)

    repeated_images = image.repeat(
        NUM_MC_SAMPLES,
        1,
        1,
        1
    )

    with torch.no_grad():

        predictions = model(repeated_images)

    # predictions:
    # [MC samples, 1, 80, 80]

    # --------------------------------------------------------
    # Divide each density map into 10 x 10 patches
    # --------------------------------------------------------

    prediction_patches = predictions.reshape(
        NUM_MC_SAMPLES,
        1,
        PATCHES_PER_SIDE,
        PATCH_SIZE,
        PATCHES_PER_SIDE,
        PATCH_SIZE
    ).sum(dim=(3, 5))

    # Shape:
    # [MC samples, 1, 8, 8]

    target_patches = target.reshape(
        1,
        PATCHES_PER_SIDE,
        PATCH_SIZE,
        PATCHES_PER_SIDE,
        PATCH_SIZE
    ).sum(dim=(2, 4))

    # Shape:
    # [1, 8, 8]

    mean_patch_counts = prediction_patches.mean(
        dim=0
    ).squeeze(0)

    patch_uncertainties = prediction_patches.var(
        dim=0,
        unbiased=False
    ).squeeze(0)

    # Absolute error for each patch
    patch_errors = (
        mean_patch_counts - target_patches.squeeze(0)
    ).abs()

    # Convert to NumPy
    errors = patch_errors.cpu().numpy().flatten()

    uncertainties = (
        patch_uncertainties.cpu().numpy().flatten()
    )

    predicted_counts = (
        mean_patch_counts.cpu().numpy().flatten()
    )

    ground_truth_counts = (
        target_patches.squeeze(0).cpu().numpy().flatten()
    )

    # --------------------------------------------------------
    # Store patch-level results
    # --------------------------------------------------------

    for patch_index in range(PATCHES_PER_IMAGE):

        row = patch_index // PATCHES_PER_SIDE
        column = patch_index % PATCHES_PER_SIDE

        all_results.append({
            "image_index": image_index,
            "patch_index": patch_index,
            "patch_row": row,
            "patch_column": column,
            "ground_truth_count": float(
                ground_truth_counts[patch_index]
            ),
            "predicted_count": float(
                predicted_counts[patch_index]
            ),
            "absolute_error": float(
                errors[patch_index]
            ),
            "count_variance": float(
                uncertainties[patch_index]
            )
        })

    # --------------------------------------------------------
    # Select top 20% most uncertain patches per image
    # --------------------------------------------------------

    number_selected = math.ceil(
        PATCHES_PER_IMAGE * TOP_PERCENTAGE
    )

    selected_indices = np.argsort(
        uncertainties
    )[-number_selected:]

    total_image_error = errors.sum()

    selected_error = errors[selected_indices].sum()

    if total_image_error > 0:

        uncertainty_coverage = (
            selected_error / total_image_error
        )

        image_error_coverages.append(
            uncertainty_coverage
        )

    else:

        uncertainty_coverage = np.nan

    # --------------------------------------------------------
    # Random patch selection baseline
    # --------------------------------------------------------

    random_coverages = []
    random_selected_errors = []

    for _ in range(RANDOM_TRIALS):

        random_indices = np.random.choice(
            PATCHES_PER_IMAGE,
            size=number_selected,
            replace=False
        )

        random_error = errors[random_indices].sum()

        random_selected_errors.append(
            random_error / number_selected
        )

        if total_image_error > 0:

            random_coverages.append(
                random_error / total_image_error
            )

    if random_coverages:

        image_random_coverages.append(
            np.mean(random_coverages)
        )

    image_selected_errors.append(
        selected_error / number_selected
    )

    image_random_errors.append(
        np.mean(random_selected_errors)
    )

    model.eval()

    if (image_index + 1) % 100 == 0:

        elapsed = time.time() - start_time

        print(
            f"Processed {image_index + 1}/"
            f"{len(val_dataset)} images "
            f"| Elapsed: {elapsed:.1f}s"
        )


# ============================================================
# AGGREGATE CORRELATION
# ============================================================

patch_errors = np.array([
    row["absolute_error"]
    for row in all_results
])

patch_uncertainties = np.array([
    row["count_variance"]
    for row in all_results
])

pearson = pearsonr(
    patch_uncertainties,
    patch_errors
)

spearman = spearmanr(
    patch_uncertainties,
    patch_errors
)


# ============================================================
# TOP 20% VS RANDOM
# ============================================================

mean_uncertainty_coverage = np.mean(
    image_error_coverages
)

mean_random_coverage = np.mean(
    image_random_coverages
)

mean_selected_error = np.mean(
    image_selected_errors
)

mean_random_error = np.mean(
    image_random_errors
)

coverage_improvement = (
    mean_uncertainty_coverage - mean_random_coverage
)

if mean_random_coverage > 0:

    coverage_ratio = (
        mean_uncertainty_coverage / mean_random_coverage
    )

else:

    coverage_ratio = float("nan")


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = list(all_results[0].keys())

with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_results)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 65)
print("PATCH-LEVEL RESULTS")
print("=" * 65)

print("Total images:", len(val_dataset))
print("Total patches:", len(all_results))

print("\nPatch uncertainty vs patch absolute error")

print(
    f"Pearson correlation:  "
    f"{pearson.statistic:.4f}"
)

print(
    f"Spearman correlation: "
    f"{spearman.statistic:.4f}"
)

print("\nTop 20% uncertain patches vs random selection")

print(
    f"Average error coverage - uncertainty: "
    f"{mean_uncertainty_coverage * 100:.2f}%"
)

print(
    f"Average error coverage - random: "
    f"{mean_random_coverage * 100:.2f}%"
)

print(
    f"Coverage improvement: "
    f"{coverage_improvement * 100:.2f} percentage points"
)

print(
    f"Coverage ratio: "
    f"{coverage_ratio:.3f}x"
)

print(
    f"Mean selected patch error: "
    f"{mean_selected_error:.4f}"
)

print(
    f"Mean random selected patch error: "
    f"{mean_random_error:.4f}"
)

print("\nEvaluation time:", round(
    time.time() - start_time, 2
), "seconds")

print("\nSaved patch results to:", OUTPUT_CSV)

print("=" * 65)
