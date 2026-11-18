"""
AirCount standalone evaluation script.

Run from D:\\aerial-crowd-counting:
    python evaluate_aircount.py

Optional:
    python evaluate_aircount.py --manifest error_aware/processed/test.jsonl
    python evaluate_aircount.py --max-samples 100

Manifest JSONL records must contain:
  - an image path under one of: image, image_path, img, img_path, crop_path, file
  - either a density-map path under one of:
    density, density_path, gt_density, gt_density_path, density_map,
    density_map_path, label, label_path
    OR a numeric count under: count, gt_count, pedestrian_count,
    num_people, people, target

If density maps are available, the script calculates cell-level precision,
recall, F1, specificity and IoU using occupied-cell thresholds. These are
density-cell proxy metrics, not bounding-box pedestrian detection metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "error_aware" / "processed" / "test.jsonl"
DEFAULT_CHECKPOINT = (
    SCRIPT_DIR / "error_aware" / "checkpoints_stage1_holdout80" / "coarse_best.pth"
)

IMAGE_KEYS = ("image", "image_path", "img", "img_path", "crop_path", "file")
DENSITY_KEYS = (
    "density", "density_path", "gt_density", "gt_density_path",
    "density_map", "density_map_path", "label", "label_path",
)
COUNT_KEYS = ("count", "gt_count", "pedestrian_count", "num_people", "people", "target")


def first_present(record, keys):
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def resolve_path(value, manifest_path):
    """Resolve manifest paths, including VisDrone's images/<filename> paths."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        manifest_path.parent / path,
        SCRIPT_DIR / path,
        SCRIPT_DIR / "dataset" / "VisDrone2019-DET-train" / path,
        SCRIPT_DIR / "dataset" / "VisDrone2019-DET-val" / path,
        SCRIPT_DIR / "dataset" / "VisDrone2019-DET-test-dev" / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Last-resort lookup by filename inside the dataset directory.
    dataset_dir = SCRIPT_DIR / "dataset"
    if dataset_dir.exists():
        matches = list(dataset_dir.rglob(path.name))
        if len(matches) == 1:
            return matches[0].resolve()

    return (SCRIPT_DIR / path).resolve()


def read_manifest(path):
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def load_density(path):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        key = next((k for k in ("density", "arr_0", "map") if k in archive.files), archive.files[0])
        array = archive[key]
    else:
        array = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
    array = np.squeeze(np.asarray(array, dtype=np.float32))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D density map in {path}; got {array.shape}")
    return np.maximum(array, 0)


def load_ground_truth(record, manifest_path):
    density_value = first_present(record, DENSITY_KEYS)
    if density_value is not None:
        density_path = resolve_path(density_value, manifest_path)
        if not density_path.exists():
            raise FileNotFoundError(f"Ground-truth density map not found: {density_path}")
        density = load_density(density_path)
        return float(density.sum()), density

    count_value = first_present(record, COUNT_KEYS)
    if count_value is None:
        # This project manifest stores the point annotations in "points".
        points = record.get("points")
        if isinstance(points, list):
            count_value = len(points)

    if count_value is None:
        raise ValueError(
            f"Record has neither a density-map path nor a count. Keys: {list(record)}"
        )
    return float(count_value), None


def load_manifest_crop(image_path, record):
    """
    Reconstruct the 512x512 crop described by the VisDrone manifest.
    The manifest stores the original image path plus crop_x/crop_y;
    it does not store a separate crop image.
    """
    image = Image.open(image_path).convert("RGB")

    crop_size = int(record.get("crop_size", 512))
    crop_x = int(record.get("crop_x", 0))
    crop_y = int(record.get("crop_y", 0))

    original_width = int(record.get("original_width", image.width))
    original_height = int(record.get("original_height", image.height))
    padded_width = int(record.get("padded_width", original_width))
    padded_height = int(record.get("padded_height", original_height))

    # Validate manifest metadata against the source image.
    if image.size != (original_width, original_height):
        raise ValueError(
            f"Manifest/source image size mismatch for {image_path}: "
            f"manifest={(original_width, original_height)}, actual={image.size}"
        )

    # Reproduce right/bottom edge padding used to form the crop grid.
    if padded_width < original_width or padded_height < original_height:
        raise ValueError(f"Invalid padded dimensions in manifest record: {record}")

    canvas = Image.new("RGB", (padded_width, padded_height))
    canvas.paste(image, (0, 0))

    if padded_width > original_width:
        right_edge = image.crop(
            (original_width - 1, 0, original_width, original_height)
        ).resize((padded_width - original_width, original_height))
        canvas.paste(right_edge, (original_width, 0))

    if padded_height > original_height:
        bottom_edge = canvas.crop(
            (0, original_height - 1, padded_width, original_height)
        ).resize((padded_width, padded_height - original_height))
        canvas.paste(bottom_edge, (0, original_height))

    crop = canvas.crop(
        (crop_x, crop_y, crop_x + crop_size, crop_y + crop_size)
    )
    if crop.size != (512, 512):
        raise ValueError(f"Expected 512x512 crop, got {crop.size} for {image_path}")

    array = np.asarray(crop, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
    return tensor


def load_model(checkpoint_path, device):
    from error_aware.model_gated import CoarseCounter

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported checkpoint format.")

    state = checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(checkpoint.get(key), dict) and checkpoint[key]:
            state = checkpoint[key]
            break

    state = {str(k): v for k, v in state.items() if torch.is_tensor(v)}
    candidates = [state]
    for prefix in ("module.", "model.", "coarse.", "module.coarse.", "model.coarse."):
        stripped = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if stripped:
            candidates.append(stripped)
    for candidate in list(candidates):
        nested = {k[len("coarse."):]: v for k, v in candidate.items() if k.startswith("coarse.")}
        if nested:
            candidates.append(nested)

    model = CoarseCounter(pretrained=False, freeze_early=True).to(device)
    expected = model.state_dict()
    diagnostics = []

    for candidate in candidates:
        matching = {
            k: v for k, v in candidate.items()
            if k in expected and tuple(v.shape) == tuple(expected[k].shape)
        }
        if len(matching) == len(expected):
            model.load_state_dict(matching, strict=True)
            model.eval()
            print(f"[AirCount] Loaded {len(matching)} model tensors.")
            return model
        diagnostics.append(f"{len(matching)}/{len(expected)} matching tensors")

    raise RuntimeError("Checkpoint/model mismatch: " + "; ".join(diagnostics))


def count_metrics(gt, pred):
    error = pred - gt
    absolute = np.abs(error)
    mse = float(np.mean(error ** 2))
    denominator = np.abs(gt) + np.abs(pred)
    valid_smape = denominator > 1e-8
    nonzero_gt = np.abs(gt) > 1e-8
    ss_total = float(np.sum((gt - np.mean(gt)) ** 2))

    within = {}
    for tolerance in (5, 10, 20):
        ok = absolute <= np.maximum(1.0, (tolerance / 100.0) * np.abs(gt))
        within[f"within_{tolerance}_percent_accuracy"] = float(ok.mean() * 100)

    return {
        "samples": int(len(gt)),
        "MAE_people": float(absolute.mean()),
        "MSE_people_squared": mse,
        "RMSE_people": float(np.sqrt(mse)),
        "mean_bias_pred_minus_gt": float(error.mean()),
        "MAPE_percent_nonzero_gt_only": (
            float((absolute[nonzero_gt] / np.abs(gt[nonzero_gt])).mean() * 100)
            if nonzero_gt.any() else None
        ),
        "sMAPE_percent": (
            float((2 * absolute[valid_smape] / denominator[valid_smape]).mean() * 100)
            if valid_smape.any() else 0.0
        ),
        "R2": (
            float(1 - np.sum(error ** 2) / ss_total)
            if ss_total > 1e-12 else None
        ),
        **within,
    }


def binary_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "cell_accuracy_percent": accuracy * 100,
        "cell_precision_percent": precision * 100,
        "cell_recall_percent": recall * 100,
        "cell_f1_percent": f1 * 100,
        "cell_specificity_percent": specificity * 100,
        "cell_IoU_percent": iou * 100,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def cell_classification_metrics(gt_maps, pred_maps, gt_threshold, pred_threshold):
    tp = fp = fn = tn = 0
    for gt_map, pred_map in zip(gt_maps, pred_maps):
        if gt_map.shape != pred_map.shape:
            gt_img = Image.fromarray(gt_map.astype(np.float32), mode="F")
            gt_img = gt_img.resize(
                (pred_map.shape[1], pred_map.shape[0]), Image.Resampling.BILINEAR
            )
            gt_map = np.asarray(gt_img, dtype=np.float32)
        actual = gt_map > gt_threshold
        predicted = pred_map > pred_threshold
        tp += int(np.logical_and(actual, predicted).sum())
        fp += int(np.logical_and(~actual, predicted).sum())
        fn += int(np.logical_and(actual, ~predicted).sum())
        tn += int(np.logical_and(~actual, ~predicted).sum())
    result = binary_metrics(tp, fp, fn, tn)
    result["definition"] = "Occupied density cells; proxy metric, not person-box detection."
    result["ground_truth_cell_threshold"] = gt_threshold
    result["predicted_cell_threshold"] = pred_threshold
    return result


def save_plots(out_dir, gt, pred, cell_result):
    error = pred - gt
    abs_error = np.abs(error)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(gt, pred, alpha=0.65)
    upper = max(float(gt.max()), float(pred.max()), 1.0)
    ax.plot([0, upper], [0, upper], "--")
    ax.set(title="Predicted vs Ground-Truth Counts", xlabel="Ground-truth count", ylabel="Predicted count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "01_predicted_vs_ground_truth.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(error, bins=min(30, max(8, int(np.sqrt(len(error))))))
    ax.axvline(0, linestyle="--")
    ax.set(title="Prediction Error Distribution", xlabel="Predicted - ground truth", ylabel="Images")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "02_error_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(abs_error)), abs_error)
    ax.set(title="Absolute Error per Image", xlabel="Test image index", ylabel="Absolute count error")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "03_absolute_error_per_image.png", dpi=160)
    plt.close(fig)

    means = (gt + pred) / 2
    bias = float(error.mean())
    std = float(error.std(ddof=1)) if len(error) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(means, error, alpha=0.65)
    ax.axhline(bias, linestyle="--", label=f"Bias = {bias:.2f}")
    ax.axhline(bias + 1.96 * std, linestyle=":", label="Upper 95% agreement")
    ax.axhline(bias - 1.96 * std, linestyle=":", label="Lower 95% agreement")
    ax.set(title="Bland-Altman Plot", xlabel="Mean of ground truth and prediction", ylabel="Prediction error")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "04_bland_altman.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gt, bins=20, alpha=0.6, label="Ground truth")
    ax.hist(pred, bins=20, alpha=0.6, label="Predicted")
    ax.set(title="Count Distributions", xlabel="Pedestrian count", ylabel="Images")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "05_count_distributions.png", dpi=160)
    plt.close(fig)

    if cell_result is not None:
        matrix = np.array([
            [cell_result["TN"], cell_result["FP"]],
            [cell_result["FN"], cell_result["TP"]],
        ])
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix)
        ax.set_xticks([0, 1], labels=["Predicted empty", "Predicted occupied"])
        ax.set_yticks([0, 1], labels=["Actual empty", "Actual occupied"])
        ax.set_title("Density-Cell Confusion Matrix")
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{value:,}", ha="center", va="center")
        fig.colorbar(image, ax=ax, label="Cell count")
        fig.tight_layout()
        fig.savefig(out_dir / "06_cell_confusion_matrix.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AirCount CoarseCounter.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "evaluation_results")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates the full manifest.")
    parser.add_argument("--gt-cell-threshold", type=float, default=1e-6)
    parser.add_argument("--pred-cell-threshold", type=float, default=1e-6)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    print(f"[AirCount] Device: {device}")
    print(f"[AirCount] Manifest: {manifest_path}")
    print("[AirCount] Image lookup: VisDrone dataset paths + manifest crop coordinates")
    print(f"[AirCount] Checkpoint: {checkpoint_path}")

    records = read_manifest(manifest_path)
    if args.max_samples > 0:
        records = records[:args.max_samples]

    model = load_model(checkpoint_path, device)

    gt_counts, pred_counts, times = [], [], []
    gt_maps, pred_maps = [], []
    rows = []

    for index, record in enumerate(records, 1):
        image_value = first_present(record, IMAGE_KEYS)
        if image_value is None:
            raise ValueError(f"Record {index} has no image path. Keys: {list(record)}")
        image_path = resolve_path(image_value, manifest_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        gt_count, gt_density = load_ground_truth(record, manifest_path)

        # Manifest rows refer to the original VisDrone image and identify
        # the crop with crop_x/crop_y. Do not resize the whole original image.
        tensor = load_manifest_crop(image_path, record).to(device)

        start = time.perf_counter()
        with torch.inference_mode():
            output = model(tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000

        pred_tensor = output["density"] if isinstance(output, dict) else output
        pred_density = np.maximum(
            pred_tensor.float().squeeze().detach().cpu().numpy(), 0
        )
        if pred_density.ndim != 2:
            raise ValueError(f"Expected 2D output density, got {pred_density.shape}")

        pred_count = float(pred_density.sum())
        gt_counts.append(gt_count)
        pred_counts.append(pred_count)
        times.append(elapsed_ms)

        if gt_density is not None:
            gt_maps.append(gt_density)
            pred_maps.append(pred_density)

        rows.append({
            "image": str(image_path),
            "ground_truth_count": gt_count,
            "predicted_count": pred_count,
            "absolute_error": abs(pred_count - gt_count),
            "signed_error": pred_count - gt_count,
            "inference_time_ms": elapsed_ms,
        })

        if index == 1 or index % 25 == 0 or index == len(records):
            print(
                f"[{index}/{len(records)}] GT={gt_count:.2f} "
                f"Pred={pred_count:.2f} "
                f"AbsErr={abs(pred_count - gt_count):.2f} "
                f"Time={elapsed_ms:.1f}ms"
            )

    gt = np.asarray(gt_counts, dtype=np.float64)
    pred = np.asarray(pred_counts, dtype=np.float64)
    metrics = count_metrics(gt, pred)
    metrics["mean_inference_time_ms"] = float(np.mean(times))
    metrics["median_inference_time_ms"] = float(np.median(times))

    cell_result = None
    if len(gt_maps) == len(records) and len(records):
        cell_result = cell_classification_metrics(
            gt_maps, pred_maps, args.gt_cell_threshold, args.pred_cell_threshold
        )
    elif gt_maps:
        print("[Warning] Mixed records with/without density maps: skipping cell metrics.")
    else:
        print("[Info] Count-only manifest: cell precision/recall/F1 cannot be calculated.")

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    save_plots(output_dir, gt, pred, cell_result)

    report = {
        "task": "Aerial crowd counting - Stage-1 CoarseCounter",
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "count_regression_metrics": metrics,
        "density_cell_classification_metrics": cell_result,
        "interpretation": [
            "MAE and RMSE are the primary metrics for crowd counting.",
            "Count tolerance accuracy is the fraction of images whose absolute count error "
            "is within 5%, 10%, or 20% of ground truth, with a minimum tolerance of one person.",
            "Cell precision/recall/F1 classify density-map cells as occupied or empty. "
            "They are not individual-person detection metrics.",
            "MAPE excludes samples with zero ground-truth count.",
        ],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 58)
    print("COUNT REGRESSION METRICS")
    print("=" * 58)
    for key, value in metrics.items():
        print(f"{key:38s}: {value:.4f}" if isinstance(value, float) else f"{key:38s}: {value}")

    if cell_result is not None:
        print("\n" + "=" * 58)
        print("DENSITY-CELL CLASSIFICATION METRICS")
        print("=" * 58)
        for key, value in cell_result.items():
            if key != "definition":
                print(f"{key:38s}: {value:.4f}" if isinstance(value, float) else f"{key:38s}: {value}")
        print("Note:", cell_result["definition"])

    print(f"\n[AirCount] Complete. Results: {output_dir}")
    print("Generated: metrics.json, predictions.csv, and diagnostic PNG graphs.")


if __name__ == "__main__":
    main()
