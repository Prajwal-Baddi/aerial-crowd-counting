import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dataset_loader import VisDroneCrowdDataset
from model import CrowdCounter
from residual_model import ResidualPatchRefiner

SEED = 42
DATASET_ROOT = "dataset/VisDrone2019-DET-train"
SPLIT_FILE = "residual_phase2_split.json"
COARSE_CHECKPOINT = "residual_phase2_coarse.pth"
REFINER_CHECKPOINT = "residual_patch_refiner_best.pth"
OUTPUT_CSV = "residual_refinement_test_results.csv"

GRID = 8
PATCH = 10
MC_SAMPLES = 20
BUDGETS = [0.10, 0.20, 0.30, 0.40]
MAX_TEST_IMAGES = None  # Set to 20 for a smoke test; None for full final test.


def load_file(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def image_tensor(image, device):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])
    return transform(image.convert("RGB")).unsqueeze(0).to(device)


def crop_tensor(image, row, col, device):
    width, height = image.size
    x0, x1 = round(col * width / GRID), round((col + 1) * width / GRID)
    y0, y1 = round(row * height / GRID), round((row + 1) * height / GRID)
    crop = image.crop((x0, y0, x1, y1))
    crop = crop.resize((160, 160), Image.Resampling.BILINEAR)
    return transforms.ToTensor()(crop.convert("RGB")).unsqueeze(0).to(device)


@torch.no_grad()
def patch_uncertainty(model, x):
    model.eval()
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()

    samples = []
    for _ in range(MC_SAMPLES):
        density = model(x)
        counts = []
        for row in range(GRID):
            for col in range(GRID):
                counts.append(
                    density[0, 0, row*PATCH:(row+1)*PATCH,
                            col*PATCH:(col+1)*PATCH].sum()
                )
        samples.append(torch.stack(counts))
    values = torch.stack(samples, dim=0)
    uncertainty = values.var(dim=0, unbiased=False).cpu().numpy()
    model.eval()
    return uncertainty


@torch.no_grad()
def apply_residual(coarse_map, image, indices, refiner, device):
    refined_map = coarse_map.clone()
    for idx in indices:
        row, col = divmod(int(idx), GRID)
        y0, y1 = row * PATCH, (row + 1) * PATCH
        x0, x1 = col * PATCH, (col + 1) * PATCH
        coarse_patch = coarse_map[:, :, y0:y1, x0:x1]
        crop = crop_tensor(image, row, col, device)
        residual = refiner(crop, coarse_patch)
        refined_patch = torch.clamp(coarse_patch + residual, min=0.0)
        refined_map[:, :, y0:y1, x0:x1] = refined_patch
    return refined_map


def summarize(errors, times):
    errors = np.asarray(errors, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "Mean_Inference_Time_ms": float(np.mean(times) * 1000.0),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads(Path(SPLIT_FILE).read_text(encoding="utf-8"))
    dataset = VisDroneCrowdDataset(root=DATASET_ROOT, image_size=(320, 320))
    test_indices = list(split["final_test"])
    if MAX_TEST_IMAGES is not None:
        test_indices = test_indices[:MAX_TEST_IMAGES]
    if not test_indices:
        raise RuntimeError("The final test split is empty.")

    coarse_ckpt = load_file(COARSE_CHECKPOINT, device)
    coarse = CrowdCounter(
        dropout_rate=coarse_ckpt.get("dropout_rate", 0.25)
    ).to(device)
    coarse.load_state_dict(coarse_ckpt["model_state_dict"])
    coarse.eval()

    refiner_ckpt = load_file(REFINER_CHECKPOINT, device)
    refiner = ResidualPatchRefiner().to(device)
    refiner.load_state_dict(refiner_ckpt["model_state_dict"])
    refiner.eval()

    results = {
        "baseline": {"errors": [], "times": []},
        "random": {b: {"errors": [], "times": []} for b in BUDGETS},
        "uncertainty": {b: {"errors": [], "times": []} for b in BUDGETS},
    }

    for image_index in tqdm(test_indices, desc="Final test"):
        image_path = dataset.image_paths[image_index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        _, _, gt_count_tensor = dataset[image_index]
        gt_count = float(gt_count_tensor.item())
        x = image_tensor(image, device)

        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            coarse_map = coarse(x)
        sync(device)
        coarse_time = time.perf_counter() - t0
        baseline_count = float(coarse_map.sum().item())
        results["baseline"]["errors"].append(baseline_count - gt_count)
        results["baseline"]["times"].append(coarse_time)

        sync(device)
        t0 = time.perf_counter()
        uncertainties = patch_uncertainty(coarse, x)
        sync(device)
        mc_time = time.perf_counter() - t0

        for budget in BUDGETS:
            k = int(np.ceil(64 * budget))
            rng = np.random.default_rng(SEED + image_index)
            random_ids = rng.permutation(64)[:k].tolist()
            uncertain_ids = np.argsort(uncertainties)[::-1][:k].tolist()

            for method, ids in (("random", random_ids), ("uncertainty", uncertain_ids)):
                sync(device)
                t0 = time.perf_counter()
                final_map = apply_residual(coarse_map, image, ids, refiner, device)
                sync(device)
                refine_time = time.perf_counter() - t0

                final_count = float(final_map.sum().item())
                elapsed = coarse_time + refine_time
                if method == "uncertainty":
                    elapsed += mc_time
                results[method][budget]["errors"].append(final_count - gt_count)
                results[method][budget]["times"].append(elapsed)

        del x, coarse_map

    rows = []
    base = summarize(results["baseline"]["errors"], results["baseline"]["times"])
    rows.append({"Method": "Baseline", "Budget": 0, "Patches_Refined": 0, **base})

    for budget in BUDGETS:
        k = int(np.ceil(64 * budget))
        for method in ("random", "uncertainty"):
            metrics = summarize(
                results[method][budget]["errors"],
                results[method][budget]["times"],
            )
            rows.append({
                "Method": method,
                "Budget": budget,
                "Patches_Refined": k,
                **metrics,
            })
            print(
                f"{method:12s} {budget:.0%} | "
                f"MAE {metrics['MAE']:.4f} | RMSE {metrics['RMSE']:.4f} | "
                f"{metrics['Mean_Inference_Time_ms']:.2f} ms"
            )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Method", "Budget", "Patches_Refined", "MAE", "RMSE",
            "Mean_Inference_Time_ms",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Final test images: {len(test_indices)}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
