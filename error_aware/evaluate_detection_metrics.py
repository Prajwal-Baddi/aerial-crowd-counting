"""Point-level and empty/non-empty evaluation for aerial crowd counting.

Run from the project root:
  python error_aware/evaluate_detection_metrics.py --split validation --mode coarse
  python error_aware/evaluate_detection_metrics.py --split test --mode coarse --empty-threshold 0.5

For Stage-2:
  python error_aware/evaluate_detection_metrics.py --split validation --mode stage2 \
    --checkpoint error_aware/checkpoints_stage2_multiplicative_stability/stage2_best.pth
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from data import ErrorAwareCrowdDataset
from model_multiplicative import ESRNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="error_aware/processed")
    p.add_argument("--dataset-root", default="dataset/VisDrone2019-DET-train")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--mode", choices=["coarse", "stage2"], default="coarse")
    p.add_argument("--checkpoint", default="error_aware/checkpoints_stage1_holdout80/coarse_best.pth")
    p.add_argument("--stage2-checkpoint",
                   default="error_aware/checkpoints_stage2_multiplicative_stability/stage2_best.pth")
    p.add_argument("--budget", type=float, default=0.25)
    p.add_argument("--empty-threshold", type=float, default=None,
                   help="Predicted count threshold for non-empty classification. If omitted, report a validation sweep.")
    p.add_argument("--empty-thresholds", default="0.1,0.25,0.5,0.75,1.0")
    p.add_argument("--match-radii", default="8,16,24",
                   help="Point matching radii in original crop pixels.")
    p.add_argument("--nms-distance", type=float, default=16.0,
                   help="Minimum separation between extracted peaks, in original crop pixels.")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--output-dir", default="error_aware/detection_evaluation")
    return p.parse_args()


def load_weights(model, path, mode, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    if mode == "coarse":
        model.coarse.load_state_dict(state, strict=True)
        model.coarse.eval()
    else:
        model.load_state_dict(state, strict=True)
        model.eval()
    return ckpt


def extract_points(density, predicted_count, nms_distance=16.0):
    """Extract up to round(predicted_count) local-maxima points from a 64x64 map.

    Coordinates are returned in 512x512 crop pixels. Density-map cells represent
    8x8 pixels, so the coordinate quantization is approximately 8 pixels.
    """
    arr = np.asarray(density, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected one 2-D density map, got {arr.shape}")
    count = max(0, int(np.rint(float(predicted_count))))
    if count == 0:
        return np.empty((0, 2), dtype=np.float32)

    local_max = arr == maximum_filter(arr, size=3, mode="constant")
    ys, xs = np.where(local_max & (arr > 0))
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    scores = arr[ys, xs]
    order = np.argsort(scores)[::-1]
    chosen = []
    min_dist_cells = max(0.0, float(nms_distance) / 8.0)
    min_dist_sq = min_dist_cells ** 2

    for idx in order:
        x, y = float(xs[idx]), float(ys[idx])
        if all((x-px)**2 + (y-py)**2 >= min_dist_sq for px, py in chosen):
            chosen.append((x, y))
            if len(chosen) >= count:
                break

    # Cell centers mapped back to the crop coordinate system.
    return np.asarray([((x + 0.5) * 8.0, (y + 0.5) * 8.0) for x, y in chosen],
                      dtype=np.float32).reshape(-1, 2)


def match_points(pred, gt, radius):
    """Maximum-cardinality one-to-one matching under a Euclidean distance gate."""
    if len(pred) == 0 or len(gt) == 0:
        return 0
    delta = pred[:, None, :] - gt[None, :, :]
    dist = np.sqrt((delta ** 2).sum(axis=2))
    # Hungarian assignment minimizes total distance; invalid edges are expensive.
    cost = dist.copy()
    cost[cost > radius] = radius + 1e6
    rows, cols = linear_sum_assignment(cost)
    return int(np.sum(dist[rows, cols] <= radius))


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def occupancy_metrics(gt_counts, pred_counts, threshold):
    gt_positive = np.asarray(gt_counts) > 0
    pred_positive = np.asarray(pred_counts) >= threshold
    tp = int(np.sum(gt_positive & pred_positive))
    tn = int(np.sum(~gt_positive & ~pred_positive))
    fp = int(np.sum(~gt_positive & pred_positive))
    fn = int(np.sum(gt_positive & ~pred_positive))
    precision, recall, f1 = prf(tp, fp, fn)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "threshold": float(threshold), "accuracy": float(accuracy),
        "precision_nonempty": float(precision), "recall_nonempty": float(recall),
        "f1_nonempty": float(f1), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    if not 0.0 <= args.budget <= 1.0:
        raise ValueError("--budget must be between 0 and 1")
    radii = [float(x) for x in args.match_radii.split(",") if x.strip()]
    thresholds = [float(x) for x in args.empty_thresholds.split(",") if x.strip()]

    manifest = Path(args.data_dir) / f"{args.split}.jsonl"
    ds = ErrorAwareCrowdDataset(
        manifest_path=manifest,
        dataset_root=args.dataset_root,
        crop_size=512,
        density_stride=8,
        gaussian_sigma=3.0,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    model = ESRNet(pretrained=False, freeze_early=True).to(device)
    ckpt_path = args.checkpoint if args.mode == "coarse" else args.stage2_checkpoint
    ckpt = load_weights(model, ckpt_path, args.mode, device)
    model.eval()
    model.coarse.eval()

    gt_counts, pred_counts = [], []
    pred_points_all, gt_points_all = [], []
    per_crop = []
    print("=" * 68)
    print("POINT-LEVEL AND OCCUPANCY EVALUATION")
    print("=" * 68)
    print(f"Device: {device} | AMP: {amp}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Split: {args.split} | mode: {args.mode} | crops: {len(ds)}")
    print(f"Checkpoint: {ckpt_path} | checkpoint epoch: {ckpt.get('epoch', 'Stage-1')}")
    print("Point extraction: local maxima + NMS; predicted point count = rounded density integral")
    print(f"Matching radii: {radii} pixels")

    for i, (image, target_density, count) in enumerate(loader):
        if args.max_batches is not None and i >= args.max_batches:
            break
        image = image.to(device, non_blocking=True)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                if args.mode == "coarse":
                    output = model.coarse(image)
                    pred_density = output["density"]
                else:
                    output = model.encode(image)
                    mask = None
                    from model_multiplicative import select_topk
                    mask = select_topk(output["error_scores"], args.budget)
                    pred_density = model.apply_refine(
                        output["density"].float(), output["stride4_features"], mask
                    )
            pred_density = pred_density.float()
            pred_count = float(pred_density.sum().item())
        gt_count = float(count.item())
        record = ds.records[i]
        gt_points = np.asarray(record["points"], dtype=np.float32).reshape(-1, 2)
        pred_points = extract_points(
            pred_density[0, 0].detach().cpu().numpy(),
            pred_count,
            nms_distance=args.nms_distance,
        )
        gt_counts.append(gt_count)
        pred_counts.append(pred_count)
        pred_points_all.append(pred_points)
        gt_points_all.append(gt_points)
        per_crop.append({
            "crop_index": i,
            "image": record["image"],
            "crop_x": record["crop_x"],
            "crop_y": record["crop_y"],
            "gt_count": gt_count,
            "pred_count": pred_count,
            "count_error": pred_count - gt_count,
            "gt_points": len(gt_points),
            "pred_points": len(pred_points),
        })
        if (i + 1) % 250 == 0 or i + 1 == len(loader):
            print(f"Processed {i+1}/{min(len(loader), args.max_batches or len(loader))}")

    gt_counts = np.asarray(gt_counts, dtype=np.float64)
    pred_counts = np.asarray(pred_counts, dtype=np.float64)
    count_mae = float(np.mean(np.abs(pred_counts - gt_counts)))
    count_rmse = float(np.sqrt(np.mean((pred_counts - gt_counts) ** 2)))

    point_results = {}
    for radius in radii:
        total_tp = total_fp = total_fn = 0
        for pred_pts, gt_pts in zip(pred_points_all, gt_points_all):
            tp = match_points(pred_pts, gt_pts, radius)
            total_tp += tp
            total_fp += len(pred_pts) - tp
            total_fn += len(gt_pts) - tp
        p, r, f = prf(total_tp, total_fp, total_fn)
        point_results[str(radius)] = {
            "matching_radius_pixels": radius,
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": p, "recall": r, "f1": f,
        }

    occ_sweep = [occupancy_metrics(gt_counts, pred_counts, t) for t in thresholds]
    best_occ = max(occ_sweep, key=lambda x: x["f1_nonempty"])
    chosen_occ = occupancy_metrics(gt_counts, pred_counts, args.empty_threshold) \
        if args.empty_threshold is not None else best_occ

    result = {
        "split": args.split, "mode": args.mode, "checkpoint": ckpt_path,
        "checkpoint_epoch": ckpt.get("epoch", None),
        "samples": len(gt_counts),
        "count_metrics": {"mae": count_mae, "rmse": count_rmse},
        "point_localization": point_results,
        "occupancy_best_from_sweep": best_occ,
        "occupancy_selected_threshold": chosen_occ,
        "occupancy_threshold_sweep": occ_sweep,
        "protocol": {
            "point_extraction": "3x3 local maxima on stride-8 density map; keep up to rounded predicted count",
            "coordinate_mapping": "stride-8 cell centers mapped to 512x512 crop coordinates",
            "nms_distance_pixels": args.nms_distance,
            "matching": "one-to-one Hungarian assignment with Euclidean distance gate",
            "note": "Choose extraction and occupancy thresholds on validation; freeze them before test evaluation.",
        },
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_{args.mode}_metrics"
    json_path = outdir / f"{stem}.json"
    csv_path = outdir / f"{stem}_per_crop.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_crop[0].keys()))
        writer.writeheader()
        writer.writerows(per_crop)

    # Save paper-ready diagnostic plots from the actual evaluated samples.
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))

        # Predicted-vs-ground-truth count scatter.
        ax = axes[0, 0]
        ax.scatter(gt_counts, pred_counts, s=9, alpha=0.35)
        upper = max(float(gt_counts.max(initial=0)), float(pred_counts.max(initial=0)), 1.0)
        ax.plot([0, upper], [0, upper], linestyle="--")
        ax.set(xlabel="Ground-truth crop count", ylabel="Predicted crop count",
               title="Count prediction scatter")
        ax.grid(alpha=0.2)

        # Count residual distribution.
        ax = axes[0, 1]
        residuals = pred_counts - gt_counts
        ax.hist(residuals, bins=40)
        ax.axvline(0, linestyle="--")
        ax.set(xlabel="Predicted − ground-truth count", ylabel="Number of crops",
               title="Count residual distribution")
        ax.grid(alpha=0.2)

        # Point-level precision, recall, F1 across matching radii.
        ax = axes[1, 0]
        radius_labels = list(point_results.keys())
        ax.plot(radius_labels, [point_results[k]["precision"] for k in radius_labels],
                marker="o", label="Precision")
        ax.plot(radius_labels, [point_results[k]["recall"] for k in radius_labels],
                marker="o", label="Recall")
        ax.plot(radius_labels, [point_results[k]["f1"] for k in radius_labels],
                marker="o", label="F1")
        ax.set(xlabel="Matching radius (pixels)", ylabel="Score",
               title="Point-level localization metrics", ylim=(0, 1.02))
        ax.legend()
        ax.grid(alpha=0.2)

        # Empty/non-empty confusion matrix at the selected threshold.
        ax = axes[1, 1]
        cm = np.array([[chosen_occ["tn"], chosen_occ["fp"]],
                       [chosen_occ["fn"], chosen_occ["tp"]]])
        im = ax.imshow(cm)
        ax.set_xticks([0, 1], ["Predicted empty", "Predicted non-empty"])
        ax.set_yticks([0, 1], ["Actual empty", "Actual non-empty"])
        ax.set_title(f"Occupancy confusion matrix (threshold={chosen_occ['threshold']:.2f})")
        for (row, col), value in np.ndenumerate(cm):
            ax.text(col, row, str(value), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        plot_path = outdir / f"{stem}_plots.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Plots: {plot_path}")
    except ImportError as exc:
        print(f"Plot generation skipped; install matplotlib to enable plots ({exc}).")

    print("\nCOUNT REGRESSION")
    print(f"MAE: {count_mae:.4f} | RMSE: {count_rmse:.4f}")
    print("\nPOINT-LEVEL LOCALIZATION")
    for radius, m in point_results.items():
        print(f"{radius}px: P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f} (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
    print("\nEMPTY/NON-EMPTY CLASSIFICATION")
    print("Threshold sweep:")
    for m in occ_sweep:
        print(f"threshold={m['threshold']:.2f} accuracy={m['accuracy']:.4f} "
              f"precision={m['precision_nonempty']:.4f} "
              f"recall={m['recall_nonempty']:.4f} F1={m['f1_nonempty']:.4f}")
    print(f"Best validation-sweep threshold: {best_occ['threshold']:.3f}")
    if args.empty_threshold is None:
        print("NOTE: threshold above is selected on this split. Do not report test-split sweep as final.")
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
