
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ErrorAwareCrowdDataset
from model import CoarseCounter


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    return model


def calculate_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    errors = predictions - targets

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_prediction": float(np.mean(predictions)),
        "mean_ground_truth": float(np.mean(targets)),
        "count": int(len(targets))
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=str,
        default="error_aware/processed/test.jsonl"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="error_aware/checkpoints/coarse_best.pth"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="error_aware/test_results/coarse_test_results.json"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("STAGE 1: COARSE COUNTER TEST EVALUATION")
    print("=" * 65)
    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Test manifest:", args.manifest)

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

    model = CoarseCounter(
        pretrained=False,
        freeze_early=True
    ).to(device)

    model = load_checkpoint(
        model,
        args.checkpoint,
        device
    )

    model.eval()

    crop_predictions = []
    crop_targets = []

    image_predictions = defaultdict(float)
    image_targets = defaultdict(float)

    empty_predictions = []
    nonempty_predictions = []
    empty_targets = []
    nonempty_targets = []

    prediction_min = float("inf")
    prediction_max = float("-inf")
    zero_predictions = 0

    print("\nEvaluating test crops...")

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):

            images, densities, counts, metadata = batch

            images = images.to(
                device,
                non_blocking=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                outputs = model(images)

            predicted_density = outputs["density"].float()

            predicted_counts = (
                predicted_density.sum(dim=(1, 2, 3))
            )

            true_counts = counts.float()

            for i in range(len(predicted_counts)):

                pred = float(predicted_counts[i].item())
                target = float(true_counts[i].item())

                crop_predictions.append(pred)
                crop_targets.append(target)

                if pred < 1e-8:
                    zero_predictions += 1

                prediction_min = min(prediction_min, pred)
                prediction_max = max(prediction_max, pred)

                if target == 0:
                    empty_predictions.append(pred)
                    empty_targets.append(target)
                else:
                    nonempty_predictions.append(pred)
                    nonempty_targets.append(target)

                image_name = metadata["image"][i]

                image_predictions[image_name] += pred
                image_targets[image_name] += target

            if (batch_idx + 1) % 500 == 0:
                print(
                    f"Processed {batch_idx + 1}/{len(loader)} "
                    f"batches"
                )

    crop_metrics = calculate_metrics(
        crop_predictions,
        crop_targets
    )

    image_names = sorted(image_targets.keys())

    original_image_metrics = calculate_metrics(
        [image_predictions[name] for name in image_names],
        [image_targets[name] for name in image_names]
    )

    empty_metrics = calculate_metrics(
        empty_predictions,
        empty_targets
    ) if empty_targets else None

    nonempty_metrics = calculate_metrics(
        nonempty_predictions,
        nonempty_targets
    ) if nonempty_targets else None

    results = {
        "checkpoint": args.checkpoint,
        "test_crops": crop_metrics,
        "original_images": original_image_metrics,
        "empty_crops": empty_metrics,
        "nonempty_crops": nonempty_metrics,
        "prediction_diagnostics": {
            "minimum_crop_prediction": prediction_min,
            "maximum_crop_prediction": prediction_max,
            "near_zero_crop_predictions": zero_predictions,
            "near_zero_prediction_percentage": (
                100.0 * zero_predictions / len(crop_predictions)
            )
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 65)
    print("TEST RESULTS")
    print("=" * 65)

    print("\nCrop-level metrics")
    print(f"MAE:  {crop_metrics['mae']:.4f}")
    print(f"RMSE: {crop_metrics['rmse']:.4f}")

    print("\nOriginal-image-level metrics")
    print(f"Images: {original_image_metrics['count']}")
    print(f"MAE:    {original_image_metrics['mae']:.4f}")
    print(f"RMSE:   {original_image_metrics['rmse']:.4f}")

    print("\nEmpty crops")
    if empty_metrics:
        print(f"Count: {empty_metrics['count']}")
        print(f"MAE:   {empty_metrics['mae']:.4f}")

    print("\nNonempty crops")
    if nonempty_metrics:
        print(f"Count: {nonempty_metrics['count']}")
        print(f"MAE:   {nonempty_metrics['mae']:.4f}")

    print("\nPrediction diagnostics")
    print(f"Minimum prediction: {prediction_min:.6f}")
    print(f"Maximum prediction: {prediction_max:.6f}")
    print(
        f"Near-zero predictions: {zero_predictions} "
        f"({100.0 * zero_predictions / len(crop_predictions):.2f}%)"
    )

    print("\nResults saved to:", output_path)
    print("=" * 65)


if __name__ == "__main__":
    main()
