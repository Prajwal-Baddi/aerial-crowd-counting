import os
import csv
import time
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import random_split
from tqdm import tqdm

from dataset_loader import VisDroneCrowdDataset
from model import CrowdCounter
from patch_refiner import PatchRefiner


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
DATASET_ROOT = "dataset/VisDrone2019-DET-train"

COARSE_CHECKPOINT = "mc_multiscale_crowdcounter.pth"
REFINER_CHECKPOINT = "patch_refiner_validated.pth"
OUTPUT_CSV = "adaptive_refinement_results.csv"

IMAGE_SIZE = 320
GRID_SIZE = 8
PATCH_SIZE = 10
PATCHES_TOTAL = GRID_SIZE * GRID_SIZE

MC_SAMPLES = 20
BUDGETS = [0.10, 0.20, 0.30, 0.40]

# Use 20 for a quick test. Set to None for all validation images.
MAX_VALIDATION_IMAGES = None

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
# DEVICE AND IMAGE PREPROCESSING
# ============================================================

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_image_tensor(image, device):
    """Resize an RGB image to the coarse model's 320x320 input."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    tensor = transform(image.convert("RGB"))
    return tensor.unsqueeze(0).to(device)


def load_patch_tensor(image, row, col, device):
    """Extract an original-resolution 8x8 grid crop and resize to 160x160."""
    width, height = image.size

    x0 = round(col * width / GRID_SIZE)
    x1 = round((col + 1) * width / GRID_SIZE)
    y0 = round(row * height / GRID_SIZE)
    y1 = round((row + 1) * height / GRID_SIZE)

    crop = image.crop((x0, y0, x1, y1))
    crop = crop.resize((160, 160), Image.Resampling.BILINEAR)

    tensor = transforms.ToTensor()(crop.convert("RGB"))
    return tensor.unsqueeze(0).to(device)


# ============================================================
# MC DROPOUT UNCERTAINTY
# ============================================================

@torch.no_grad()
def get_patch_uncertainties(model, image_tensor):
    """
    Estimate uncertainty for each of 64 patches by calculating
    the variance of each patch's predicted count across MC passes.

    BatchNorm remains in eval mode; only dropout layers are enabled.
    """
    model.eval()

    for module in model.modules():
        if isinstance(
            module,
            (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d),
        ):
            module.train()

    patch_count_samples = []

    for _ in range(MC_SAMPLES):
        prediction = model(image_tensor)

        expected_spatial = (GRID_SIZE * PATCH_SIZE, GRID_SIZE * PATCH_SIZE)
        if tuple(prediction.shape[-2:]) != expected_spatial:
            model.eval()
            raise ValueError(
                f"Expected coarse output {expected_spatial}, "
                f"got {tuple(prediction.shape[-2:])}"
            )

        patch_counts = []
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                y0 = row * PATCH_SIZE
                y1 = (row + 1) * PATCH_SIZE
                x0 = col * PATCH_SIZE
                x1 = (col + 1) * PATCH_SIZE

                patch_counts.append(
                    prediction[0, 0, y0:y1, x0:x1].sum()
                )

        patch_count_samples.append(torch.stack(patch_counts))

    samples = torch.stack(patch_count_samples, dim=0)
    uncertainties = samples.var(dim=0, unbiased=False)

    model.eval()
    return uncertainties.cpu().numpy()


# ============================================================
# PATCH SELECTION
# ============================================================

def get_budget_patch_count(budget):
    return int(np.ceil(PATCHES_TOTAL * budget))


def select_uncertain_patches(uncertainties, number_of_patches):
    indices = np.argsort(uncertainties)[::-1]
    return indices[:number_of_patches].tolist()


def select_random_patches(number_of_patches, image_index):
    """Reproducible random patch selection for each image."""
    rng = np.random.default_rng(SEED + image_index)
    return rng.permutation(PATCHES_TOTAL)[:number_of_patches].tolist()


# ============================================================
# APPLY PATCH REFINEMENT
# ============================================================

@torch.no_grad()
def refine_selected_patches(
    coarse_density,
    original_image,
    selected_indices,
    refiner,
    device,
):
    """Replace selected 10x10 coarse regions with refiner predictions."""
    refined_density = coarse_density.clone()

    for patch_index in selected_indices:
        row = patch_index // GRID_SIZE
        col = patch_index % GRID_SIZE

        patch_tensor = load_patch_tensor(
            original_image, row, col, device
        )
        refined_patch = refiner(patch_tensor)

        expected_shape = (1, 1, PATCH_SIZE, PATCH_SIZE)
        if tuple(refined_patch.shape) != expected_shape:
            raise ValueError(
                f"Expected refiner output {expected_shape}, "
                f"got {tuple(refined_patch.shape)}"
            )

        y0 = row * PATCH_SIZE
        y1 = (row + 1) * PATCH_SIZE
        x0 = col * PATCH_SIZE
        x1 = (col + 1) * PATCH_SIZE

        refined_density[0, 0, y0:y1, x0:x1] = refined_patch[0, 0]

    return refined_density


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(errors, timings):
    errors = np.asarray(errors, dtype=np.float64)
    timings = np.asarray(timings, dtype=np.float64)

    if len(errors) == 0:
        raise ValueError("No evaluation errors were collected.")

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "Mean_Inference_Time_ms": float(np.mean(timings) * 1000.0),
    }


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=device)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("UNCERTAINTY-GUIDED ADAPTIVE REFINEMENT EVALUATION")
    print("=" * 65)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Check checkpoint files.
    for checkpoint_path in (
        COARSE_CHECKPOINT,
        REFINER_CHECKPOINT,
    ):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

    # Load dataset and reproduce the exact 80/20 split.
    dataset = VisDroneCrowdDataset(
        root=DATASET_ROOT,
        image_size=(320, 320),
    )

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    _, validation_subset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    validation_indices = list(validation_subset.indices)

    if MAX_VALIDATION_IMAGES is not None:
        validation_indices = validation_indices[:MAX_VALIDATION_IMAGES]

    if not validation_indices:
        raise ValueError("The validation image list is empty.")

    print(f"\nTotal dataset images: {len(dataset)}")
    print(f"Validation images selected: {len(validation_indices)}")

    # Load coarse MC Dropout model.
    coarse_checkpoint = load_checkpoint(COARSE_CHECKPOINT, device)
    dropout_rate = coarse_checkpoint.get("dropout_rate", 0.25)

    coarse_model = CrowdCounter(
        dropout_rate=dropout_rate
    ).to(device)

    coarse_model.load_state_dict(
        coarse_checkpoint["model_state_dict"]
    )
    coarse_model.eval()

    print(
        "\nCoarse checkpoint epoch:",
        coarse_checkpoint.get("epoch", "unknown"),
    )
    print("Coarse dropout rate:", dropout_rate)

    # Load validated patch refiner.
    refiner_checkpoint = load_checkpoint(REFINER_CHECKPOINT, device)

    refiner = PatchRefiner().to(device)
    refiner.load_state_dict(
        refiner_checkpoint["model_state_dict"]
    )
    refiner.eval()

    print(
        "Refiner checkpoint epoch:",
        refiner_checkpoint.get("epoch", "unknown"),
    )
    print(
        "Best validation patch MAE:",
        refiner_checkpoint.get("best_validation_mae", "unknown"),
    )

    # Results are appended image-major:
    # image 1: budgets 10%, 20%, 30%, 40%; then image 2, etc.
    results = {
        "baseline": {"errors": [], "timings": []},
        "random": {"errors": [], "timings": []},
        "uncertainty": {"errors": [], "timings": []},
    }

    # Evaluate images one at a time to limit GPU memory use.
    for image_index in tqdm(
        validation_indices,
        desc="Evaluating validation images",
    ):
        image_path = dataset.image_paths[image_index]

        with Image.open(image_path) as image_file:
            original_image = image_file.convert("RGB")

        # Ground-truth count from the dataset loader.
        _, _, ground_truth_tensor = dataset[image_index]
        ground_truth_count = float(ground_truth_tensor.item())

        image_tensor = load_image_tensor(original_image, device)

        # 1. Deterministic coarse prediction.
        coarse_model.eval()

        synchronize(device)
        start = time.perf_counter()

        with torch.no_grad():
            coarse_density = coarse_model(image_tensor)

        synchronize(device)
        coarse_time = time.perf_counter() - start

        if tuple(coarse_density.shape) != (1, 1, 80, 80):
            raise ValueError(
                "Expected coarse density shape (1, 1, 80, 80), "
                f"got {tuple(coarse_density.shape)}"
            )

        coarse_count = float(coarse_density.sum().item())

        results["baseline"]["errors"].append(
            coarse_count - ground_truth_count
        )
        results["baseline"]["timings"].append(coarse_time)

        # 2. MC Dropout uncertainty.
        synchronize(device)
        start = time.perf_counter()

        uncertainties = get_patch_uncertainties(
            coarse_model,
            image_tensor,
        )

        synchronize(device)
        mc_time = time.perf_counter() - start

        # 3. Evaluate every refinement budget.
        for budget in BUDGETS:
            number_of_patches = get_budget_patch_count(budget)

            # Random refinement.
            random_indices = select_random_patches(
                number_of_patches,
                image_index,
            )

            synchronize(device)
            start = time.perf_counter()

            random_density = refine_selected_patches(
                coarse_density,
                original_image,
                random_indices,
                refiner,
                device,
            )

            synchronize(device)
            random_refinement_time = time.perf_counter() - start

            random_count = float(random_density.sum().item())

            results["random"]["errors"].append(
                random_count - ground_truth_count
            )
            results["random"]["timings"].append(
                coarse_time + random_refinement_time
            )

            # Uncertainty-guided refinement.
            uncertain_indices = select_uncertain_patches(
                uncertainties,
                number_of_patches,
            )

            synchronize(device)
            start = time.perf_counter()

            uncertain_density = refine_selected_patches(
                coarse_density,
                original_image,
                uncertain_indices,
                refiner,
                device,
            )

            synchronize(device)
            uncertainty_refinement_time = time.perf_counter() - start

            uncertain_count = float(uncertain_density.sum().item())

            results["uncertainty"]["errors"].append(
                uncertain_count - ground_truth_count
            )
            results["uncertainty"]["timings"].append(
                coarse_time + mc_time + uncertainty_refinement_time
            )

        del image_tensor
        del coarse_density
        del original_image

    # Aggregate metrics.
    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    rows = []

    baseline_metrics = calculate_metrics(
        results["baseline"]["errors"],
        results["baseline"]["timings"],
    )

    rows.append({
        "Method": "Baseline",
        "Budget": 0,
        "Patches_Refined": 0,
        **baseline_metrics,
    })

    print("\nBASELINE")
    print(f"MAE:  {baseline_metrics['MAE']:.4f}")
    print(f"RMSE: {baseline_metrics['RMSE']:.4f}")
    print(
        "Mean inference time: "
        f"{baseline_metrics['Mean_Inference_Time_ms']:.2f} ms"
    )

    number_of_images = len(validation_indices)

    # Correct image-major indexing:
    # result index = image_position * number_of_budgets + budget_index.
    for budget_index, budget in enumerate(BUDGETS):
        number_of_patches = get_budget_patch_count(budget)

        result_indices = [
            image_position * len(BUDGETS) + budget_index
            for image_position in range(number_of_images)
        ]

        for method in ("random", "uncertainty"):
            errors = [
                results[method]["errors"][i]
                for i in result_indices
            ]
            timings = [
                results[method]["timings"][i]
                for i in result_indices
            ]

            metrics = calculate_metrics(errors, timings)

            method_name = (
                "Random refinement"
                if method == "random"
                else "Uncertainty-guided refinement"
            )

            rows.append({
                "Method": method_name,
                "Budget": budget,
                "Patches_Refined": number_of_patches,
                **metrics,
            })

            print(f"\n{method_name.upper()} - {budget * 100:.0f}%")
            print(
                f"Patches refined: {number_of_patches}/{PATCHES_TOTAL}"
            )
            print(f"MAE:  {metrics['MAE']:.4f}")
            print(f"RMSE: {metrics['RMSE']:.4f}")
            print(
                "Mean inference time: "
                f"{metrics['Mean_Inference_Time_ms']:.2f} ms"
            )

    # Save CSV.
    fieldnames = [
        "Method",
        "Budget",
        "Patches_Refined",
        "MAE",
        "RMSE",
        "Mean_Inference_Time_ms",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Images evaluated: {number_of_images}")


if __name__ == "__main__":
    main()
