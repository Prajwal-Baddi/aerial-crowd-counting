
import csv
import math
import time
import random

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

# Different percentages of patches selected for refinement
REFINEMENT_BUDGETS = [0.10, 0.20, 0.30, 0.40]

RANDOM_TRIALS = 100

PATCH_OUTPUT_CSV = "patch_uncertainty_evaluation.csv"
BUDGET_OUTPUT_CSV = "refinement_budget_results.csv"

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


# ============================================================
# DATASET
# ============================================================

dataset = VisDroneCrowdDataset(
    root="dataset/VisDrone2019-DET-train",
    image_size=(320, 320)
)

total_size = len(dataset)

train_size = int(0.8 * total_size)
val_size = total_size - train_size

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


# ============================================================
# EXPERIMENT INFORMATION
# ============================================================

print("=" * 70)
print("PATCH-LEVEL UNCERTAINTY AND REFINEMENT BUDGET EVALUATION")
print("=" * 70)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("Validation images:", len(val_dataset))
print("MC samples:", NUM_MC_SAMPLES)

print("Density map:", f"{DENSITY_SIZE} x {DENSITY_SIZE}")
print("Patch size:", f"{PATCH_SIZE} x {PATCH_SIZE}")

print("Patches per image:", PATCHES_PER_IMAGE)

print("Refinement budgets:", REFINEMENT_BUDGETS)

print("Random trials per budget:", RANDOM_TRIALS)


# ============================================================
# STORAGE
# ============================================================

all_patch_results = []
all_budget_results = []

start_time = time.time()


# ============================================================
# VALIDATION LOOP
# ============================================================

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
    # Generate MC Dropout predictions
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

    # Shape:
    # [NUM_MC_SAMPLES, 1, 80, 80]

    # --------------------------------------------------------
    # Divide predictions into patches
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
    # [NUM_MC_SAMPLES, 1, 8, 8]

    target_patches = target.reshape(
        1,
        PATCHES_PER_SIDE,
        PATCH_SIZE,
        PATCHES_PER_SIDE,
        PATCH_SIZE
    ).sum(dim=(2, 4))

    # Shape:
    # [1, 8, 8]

    # --------------------------------------------------------
    # Calculate mean prediction and uncertainty
    # --------------------------------------------------------

    mean_patch_counts = prediction_patches.mean(
        dim=0
    ).squeeze(0)

    patch_uncertainties = prediction_patches.var(
        dim=0,
        unbiased=False
    ).squeeze(0)

    # --------------------------------------------------------
    # Calculate patch-level error
    # --------------------------------------------------------

    patch_errors = (
        mean_patch_counts - target_patches.squeeze(0)
    ).abs()

    errors = (
        patch_errors.detach().cpu().numpy().flatten()
    )

    uncertainties = (
        patch_uncertainties.detach().cpu().numpy().flatten()
    )

    predicted_counts = (
        mean_patch_counts.detach().cpu().numpy().flatten()
    )

    ground_truth_counts = (
        target_patches.squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )

    # --------------------------------------------------------
    # Store individual patch results
    # --------------------------------------------------------

    for patch_index in range(PATCHES_PER_IMAGE):

        patch_row = patch_index // PATCHES_PER_SIDE
        patch_column = patch_index % PATCHES_PER_SIDE

        all_patch_results.append({

            "image_index": image_index,

            "patch_index": patch_index,

            "patch_row": patch_row,

            "patch_column": patch_column,

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
    # Evaluate different refinement budgets
    # --------------------------------------------------------

    total_image_error = errors.sum()

    # Sort patches by uncertainty, highest first
    uncertainty_order = np.argsort(uncertainties)[::-1]

    for budget in REFINEMENT_BUDGETS:

        number_selected = math.ceil(
            PATCHES_PER_IMAGE * budget
        )

        # ----------------------------------------------------
        # Uncertainty-guided selection
        # ----------------------------------------------------

        selected_indices = uncertainty_order[
            :number_selected
        ]

        selected_error = errors[selected_indices].sum()

        if total_image_error > 0:

            uncertainty_coverage = (
                selected_error / total_image_error
            )

        else:

            uncertainty_coverage = np.nan

        # ----------------------------------------------------
        # Random selection baseline
        # ----------------------------------------------------

        random_coverages = []
        random_selected_errors = []

        for _ in range(RANDOM_TRIALS):

            random_indices = np.random.choice(
                PATCHES_PER_IMAGE,
                size=number_selected,
                replace=False
            )

            random_error = errors[random_indices].sum()

            if total_image_error > 0:

                random_coverages.append(
                    random_error / total_image_error
                )

            random_selected_errors.append(
                random_error / number_selected
            )

        if random_coverages:

            random_coverage = np.mean(
                random_coverages
            )

        else:

            random_coverage = np.nan

        random_mean_error = np.mean(
            random_selected_errors
        )

        uncertainty_mean_error = (
            selected_error / number_selected
        )

        # ----------------------------------------------------
        # Store budget results
        # ----------------------------------------------------

        all_budget_results.append({

            "image_index": image_index,

            "budget_percentage": budget * 100,

            "patches_selected": number_selected,

            "uncertainty_error_coverage": float(
                uncertainty_coverage
            ),

            "random_error_coverage": float(
                random_coverage
            ),

            "uncertainty_mean_patch_error": float(
                uncertainty_mean_error
            ),

            "random_mean_patch_error": float(
                random_mean_error
            )
        })

    # --------------------------------------------------------
    # Reset model to evaluation mode
    # --------------------------------------------------------

    model.eval()

    if (image_index + 1) % 100 == 0:

        elapsed = time.time() - start_time

        print(
            f"Processed {image_index + 1}/"
            f"{len(val_dataset)} images "
            f"| Elapsed: {elapsed:.1f}s"
        )


# ============================================================
# PATCH-LEVEL CORRELATION
# ============================================================

patch_errors = np.array([
    row["absolute_error"]
    for row in all_patch_results
])

patch_uncertainties = np.array([
    row["count_variance"]
    for row in all_patch_results
])

pearson_result = pearsonr(
    patch_uncertainties,
    patch_errors
)

spearman_result = spearmanr(
    patch_uncertainties,
    patch_errors
)


# ============================================================
# AGGREGATE BUDGET RESULTS
# ============================================================

budget_summary = []

for budget in REFINEMENT_BUDGETS:

    budget_rows = [
        row
        for row in all_budget_results
        if row["budget_percentage"] == budget * 100
    ]

    uncertainty_coverages = np.array([
        row["uncertainty_error_coverage"]
        for row in budget_rows
    ])

    random_coverages = np.array([
        row["random_error_coverage"]
        for row in budget_rows
    ])

    uncertainty_errors = np.array([
        row["uncertainty_mean_patch_error"]
        for row in budget_rows
    ])

    random_errors = np.array([
        row["random_mean_patch_error"]
        for row in budget_rows
    ])

    # Ignore images with zero total patch error
    valid_uncertainty = uncertainty_coverages[
        np.isfinite(uncertainty_coverages)
    ]

    valid_random = random_coverages[
        np.isfinite(random_coverages)
    ]

    mean_uncertainty_coverage = (
        float(valid_uncertainty.mean())
        if len(valid_uncertainty) > 0
        else float("nan")
    )

    mean_random_coverage = (
        float(valid_random.mean())
        if len(valid_random) > 0
        else float("nan")
    )

    coverage_improvement = (
        mean_uncertainty_coverage - mean_random_coverage
    )

    coverage_ratio = (
        mean_uncertainty_coverage / mean_random_coverage
        if mean_random_coverage > 0
        else float("nan")
    )

    summary = {

        "budget_percentage": budget * 100,

        "patches_selected": math.ceil(
            PATCHES_PER_IMAGE * budget
        ),

        "uncertainty_coverage": mean_uncertainty_coverage,

        "random_coverage": mean_random_coverage,

        "coverage_improvement": coverage_improvement,

        "coverage_ratio": coverage_ratio,

        "uncertainty_mean_patch_error": float(
            uncertainty_errors.mean()
        ),

        "random_mean_patch_error": float(
            random_errors.mean()
        )
    }

    budget_summary.append(summary)


# ============================================================
# SAVE PATCH-LEVEL CSV
# ============================================================

with open(
    PATCH_OUTPUT_CSV,
    "w",
    newline=""
) as file:

    fieldnames = list(all_patch_results[0].keys())

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_patch_results)


# ============================================================
# SAVE BUDGET-LEVEL CSV
# ============================================================

with open(
    BUDGET_OUTPUT_CSV,
    "w",
    newline=""
) as file:

    fieldnames = list(all_budget_results[0].keys())

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_budget_results)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PATCH-LEVEL UNCERTAINTY RESULTS")
print("=" * 70)

print("Total images:", len(val_dataset))

print("Total patches:", len(all_patch_results))

print(
    f"Pearson correlation: "
    f"{pearson_result.statistic:.4f}"
)

print(
    f"Spearman correlation: "
    f"{spearman_result.statistic:.4f}"
)


print("\n" + "=" * 70)
print("REFINEMENT BUDGET COMPARISON")
print("=" * 70)

print(
    f"{'Budget':<12}"
    f"{'Uncertainty':<18}"
    f"{'Random':<15}"
    f"{'Improvement':<15}"
    f"{'Ratio':<10}"
)

print("-" * 70)

for result in budget_summary:

    print(
        f"{result['budget_percentage']:<12.0f}"
        f"{result['uncertainty_coverage'] * 100:<18.2f}"
        f"{result['random_coverage'] * 100:<15.2f}"
        f"{result['coverage_improvement'] * 100:<15.2f}"
        f"{result['coverage_ratio']:<10.3f}"
    )


print("\nMean selected patch errors:")

for result in budget_summary:

    print(
        f"{result['budget_percentage']:.0f}% budget | "
        f"Uncertainty: "
        f"{result['uncertainty_mean_patch_error']:.4f} | "
        f"Random: "
        f"{result['random_mean_patch_error']:.4f}"
    )


print("\nTotal evaluation time:", round(
    time.time() - start_time, 2
), "seconds")

print("\nFiles generated:")

print("-", PATCH_OUTPUT_CSV)

print("-", BUDGET_OUTPUT_CSV)

print("=" * 70)
