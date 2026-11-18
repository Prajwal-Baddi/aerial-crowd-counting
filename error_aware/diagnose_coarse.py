
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ErrorAwareCrowdDataset
from model import CoarseCounter


def load_model(checkpoint_path, device):
    model = CoarseCounter(
        pretrained=False,
        freeze_early=True
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()
    return model


def calculate_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    if len(targets) == 0:
        return None

    errors = predictions - targets

    return {
        "samples": int(len(targets)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_prediction": float(np.mean(predictions)),
        "mean_ground_truth": float(np.mean(targets)),
        "mean_bias": float(np.mean(errors))
    }


def save_visualization(sample, output_path):
    image = sample["image"]
    gt_density = sample["gt_density"]
    pred_density = sample["pred_density"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Input crop")

    gt_plot = axes[1].imshow(
        gt_density,
        cmap="jet",
        interpolation="nearest"
    )
    axes[1].set_title(
        f"Ground truth\nCount: {sample['target']:.2f}"
    )
    fig.colorbar(gt_plot, ax=axes[1], fraction=0.046)

    pred_plot = axes[2].imshow(
        pred_density,
        cmap="jet",
        interpolation="nearest"
    )
    axes[2].set_title(
        f"Prediction\nCount: {sample['prediction']:.2f}"
    )
    fig.colorbar(pred_plot, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Absolute count error: {sample['absolute_error']:.3f}\n"
        f"Original image: {sample['image_name']}",
        fontsize=11
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="error_aware/processed/test.jsonl"
    )
    parser.add_argument(
        "--checkpoint",
        default="error_aware/checkpoints/coarse_best.pth"
    )
    parser.add_argument(
        "--output-dir",
        default="error_aware/diagnostics"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(args.output_dir)
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    dataset = ErrorAwareCrowdDataset(
        manifest_path=args.manifest,
        dataset_root="dataset/VisDrone2019-DET-train",
        crop_size=512,
        density_stride=8,
        gaussian_sigma=3.0,
        return_metadata=True
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    model = load_model(args.checkpoint, device)

    records = []
    visual_samples = []

    print("=" * 65)
    print("COARSE COUNTER DIAGNOSTICS")
    print("=" * 65)
    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Test crops:", len(dataset))

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            images, densities, counts, metadata = batch

            images = images.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                outputs = model(images)

            pred_density = outputs["density"].float().cpu()
            gt_density = densities.float().cpu()

            pred_counts = pred_density.sum(dim=(1, 2, 3))
            true_counts = counts.float()

            for i in range(len(pred_counts)):
                prediction = float(pred_counts[i].item())
                target = float(true_counts[i].item())
                error = prediction - target

                image_name = metadata["image"][i]

                record = {
                    "crop_index": len(records),
                    "image_name": str(image_name),
                    "target": target,
                    "prediction": prediction,
                    "error": error,
                    "absolute_error": abs(error),
                    "is_empty": target == 0,
                    "is_near_zero": prediction < 0.1
                }
                records.append(record)

                # Keep candidate samples for visual inspection.
                if target > 0:
                    visual_samples.append({
                        **record,
                        "image": (
                            images[i].detach().cpu()
                            .permute(1, 2, 0).numpy()
                        ),
                        "gt_density": gt_density[i, 0].numpy(),
                        "pred_density": pred_density[i, 0].numpy()
                    })
                elif prediction >= 1.0:
                    visual_samples.append({
                        **record,
                        "image": (
                            images[i].detach().cpu()
                            .permute(1, 2, 0).numpy()
                        ),
                        "gt_density": gt_density[i, 0].numpy(),
                        "pred_density": pred_density[i, 0].numpy()
                    })

            if (batch_idx + 1) % 500 == 0:
                print(f"Processed {batch_idx + 1}/{len(loader)}")

    # Separate empty and nonempty crops.
    empty = [r for r in records if r["is_empty"]]
    nonempty = [r for r in records if not r["is_empty"]]

    predictions = [r["prediction"] for r in records]
    targets = [r["target"] for r in records]

    nonempty_zero = [
        r for r in nonempty if r["prediction"] < 0.1
    ]

    # Crowd-count ranges, based on ground-truth crop counts.
    count_ranges = [
        ("1", 1, 2),
        ("2-5", 2, 6),
        ("6-10", 6, 11),
        ("11-20", 11, 21),
        ("21+", 21, float("inf"))
    ]

    range_results = {}

    for label, lower, upper in count_ranges:
        subset = [
            r for r in nonempty
            if lower <= r["target"] < upper
        ]

        range_results[label] = calculate_metrics(
            [r["prediction"] for r in subset],
            [r["target"] for r in subset]
        )

    # Aggregate crops into original images.
    image_predictions = defaultdict(float)
    image_targets = defaultdict(float)

    for record in records:
        name = record["image_name"]
        image_predictions[name] += record["prediction"]
        image_targets[name] += record["target"]

    image_names = sorted(image_targets)

    image_metrics = calculate_metrics(
        [image_predictions[name] for name in image_names],
        [image_targets[name] for name in image_names]
    )

    results = {
        "overall_crop_metrics": calculate_metrics(
            predictions, targets
        ),
        "empty_crop_metrics": calculate_metrics(
            [r["prediction"] for r in empty],
            [r["target"] for r in empty]
        ),
        "nonempty_crop_metrics": calculate_metrics(
            [r["prediction"] for r in nonempty],
            [r["target"] for r in nonempty]
        ),
        "nonempty_near_zero_diagnostics": {
            "near_zero_threshold": 0.1,
            "nonempty_crops": len(nonempty),
            "near_zero_predictions": len(nonempty_zero),
            "near_zero_percentage": (
                100.0 * len(nonempty_zero) / len(nonempty)
                if nonempty else 0.0
            )
        },
        "count_range_metrics": range_results,
        "original_image_metrics": image_metrics
    }

    # Write JSON report.
    with open(output_dir / "diagnostic_report.json", "w") as f:
        json.dump(results, f, indent=4)

    # Write per-crop predictions.
    with open(
        output_dir / "crop_predictions.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "crop_index",
                "image_name",
                "target",
                "prediction",
                "error",
                "absolute_error",
                "is_empty",
                "is_near_zero"
            ]
        )
        writer.writeheader()
        writer.writerows(records)

    # Select useful visual examples:
    # 1. Worst nonempty crop.
    # 2. Best nonempty crop.
    # 3. Largest false-positive empty crop.
    nonempty_sorted = sorted(
        [s for s in visual_samples if s["target"] > 0],
        key=lambda s: s["absolute_error"]
    )

    empty_sorted = sorted(
        [s for s in visual_samples if s["target"] == 0],
        key=lambda s: s["prediction"],
        reverse=True
    )

    selected = []

    if nonempty_sorted:
        selected.append(("best_nonempty", nonempty_sorted[0]))
        selected.append(("worst_nonempty", nonempty_sorted[-1]))

    if empty_sorted:
        selected.append(("largest_empty_false_positive", empty_sorted[0]))

    for name, sample in selected:
        save_visualization(
            sample,
            visualization_dir / f"{name}.png"
        )

    # Print report.
    print("\n" + "=" * 65)
    print("DIAGNOSTIC RESULTS")
    print("=" * 65)

    print("\nOverall crop metrics:")
    print(json.dumps(results["overall_crop_metrics"], indent=2))

    print("\nEmpty crop metrics:")
    print(json.dumps(results["empty_crop_metrics"], indent=2))

    print("\nNonempty crop metrics:")
    print(json.dumps(results["nonempty_crop_metrics"], indent=2))

    print("\nNear-zero predictions on NONEMPTY crops:")
    print(json.dumps(
        results["nonempty_near_zero_diagnostics"],
        indent=2
    ))

    print("\nMetrics by ground-truth count range:")
    for label, metrics in range_results.items():
        print(f"\n{label}: {metrics}")

    print("\nOriginal-image metrics:")
    print(json.dumps(image_metrics, indent=2))

    print("\nSaved report:", output_dir / "diagnostic_report.json")
    print("Saved CSV:", output_dir / "crop_predictions.csv")
    print("Saved visualizations:", visualization_dir)
    print("=" * 65)


if __name__ == "__main__":
    main()
