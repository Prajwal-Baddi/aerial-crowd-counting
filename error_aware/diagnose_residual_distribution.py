"""
Residual distribution diagnostic for Error-Aware Selective Refinement.

Run from the project root:
    python error_aware/diagnose_residual_distribution.py

Smoke test:
    python error_aware/diagnose_residual_distribution.py --max-test-batches 20

This script does not train or modify model weights.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data import get_loader
from model import ESRNet, cell_sum, select_topk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="error_aware/processed")
    parser.add_argument("--checkpoint", default="error_aware/checkpoints/stage2_best.pth")
    parser.add_argument("--history", default="error_aware/checkpoints/stage2_training_history.json")
    parser.add_argument("--output-dir", default="error_aware/diagnostics")
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.10, 0.25, 0.40])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--refine-width", type=int, default=96)
    return parser.parse_args()


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state, strict=True)
    print(
        f"Loaded checkpoint: {path} | epoch={checkpoint.get('epoch', 'unknown')} "
        f"| best_val_mae={checkpoint.get('best_val_mae', 'unknown')}"
    )
    return checkpoint


def new_accumulator():
    return {
        "cells": 0,
        "raw_sum": 0.0, "raw_abs_sum": 0.0, "raw_sq_sum": 0.0,
        "raw_min": float("inf"), "raw_max": float("-inf"),
        "raw_negative": 0, "raw_positive": 0, "raw_near_zero": 0,
        "effective_sum": 0.0, "effective_abs_sum": 0.0, "effective_sq_sum": 0.0,
        "effective_min": float("inf"), "effective_max": float("-inf"),
        "clamp_changed_pixels": 0, "refined_pixels": 0,
        "under_cells": 0, "over_cells": 0,
        "under_raw_sum": 0.0, "under_effective_sum": 0.0,
        "over_raw_sum": 0.0, "over_effective_sum": 0.0,
        "under_before": 0.0, "under_after": 0.0,
        "over_before": 0.0, "over_after": 0.0,
        "error_abs_sum": 0.0, "correction_abs_sum": 0.0,
    }


def add_batch(acc, raw_map, effective_map, before, after, mask, clamp_changed):
    selected = mask.bool()
    raw = raw_map[selected].float()
    effective = effective_map[selected].float()
    b = before[selected].float()
    a = after[selected].float()

    if raw.numel() == 0:
        return

    acc["cells"] += raw.numel()
    acc["raw_sum"] += raw.sum().item()
    acc["raw_abs_sum"] += raw.abs().sum().item()
    acc["raw_sq_sum"] += raw.square().sum().item()
    acc["raw_min"] = min(acc["raw_min"], raw.min().item())
    acc["raw_max"] = max(acc["raw_max"], raw.max().item())
    acc["raw_negative"] += int((raw < -1e-8).sum().item())
    acc["raw_positive"] += int((raw > 1e-8).sum().item())
    acc["raw_near_zero"] += int((raw.abs() <= 1e-8).sum().item())

    acc["effective_sum"] += effective.sum().item()
    acc["effective_abs_sum"] += effective.abs().sum().item()
    acc["effective_sq_sum"] += effective.square().sum().item()
    acc["effective_min"] = min(acc["effective_min"], effective.min().item())
    acc["effective_max"] = max(acc["effective_max"], effective.max().item())

    acc["clamp_changed_pixels"] += int(clamp_changed)
    acc["refined_pixels"] += raw.numel() * 16

    # before = coarse cell count - target cell count.
    under = b < -1e-6
    over = b > 1e-6
    if under.any():
        acc["under_cells"] += int(under.sum().item())
        acc["under_raw_sum"] += raw[under].sum().item()
        acc["under_effective_sum"] += effective[under].sum().item()
        acc["under_before"] += b[under].abs().sum().item()
        acc["under_after"] += a[under].abs().sum().item()
    if over.any():
        acc["over_cells"] += int(over.sum().item())
        acc["over_raw_sum"] += raw[over].sum().item()
        acc["over_effective_sum"] += effective[over].sum().item()
        acc["over_before"] += b[over].abs().sum().item()
        acc["over_after"] += a[over].abs().sum().item()

    acc["error_abs_sum"] += b.abs().sum().item()
    acc["correction_abs_sum"] += effective.abs().sum().item()


def finalize(acc):
    n = max(acc["cells"], 1)
    un = max(acc["under_cells"], 1)
    on = max(acc["over_cells"], 1)
    raw_mean = acc["raw_sum"] / n
    eff_mean = acc["effective_sum"] / n
    return {
        "selected_cells": acc["cells"],
        "raw_residual_mean": raw_mean,
        "raw_residual_mean_abs": acc["raw_abs_sum"] / n,
        "raw_residual_std": max(acc["raw_sq_sum"] / n - raw_mean**2, 0.0) ** 0.5,
        "raw_residual_min": acc["raw_min"] if acc["cells"] else None,
        "raw_residual_max": acc["raw_max"] if acc["cells"] else None,
        "raw_negative_fraction": acc["raw_negative"] / n,
        "raw_positive_fraction": acc["raw_positive"] / n,
        "raw_near_zero_fraction": acc["raw_near_zero"] / n,
        "effective_residual_mean": eff_mean,
        "effective_residual_mean_abs": acc["effective_abs_sum"] / n,
        "effective_residual_std": max(acc["effective_sq_sum"] / n - eff_mean**2, 0.0) ** 0.5,
        "effective_residual_min": acc["effective_min"] if acc["cells"] else None,
        "effective_residual_max": acc["effective_max"] if acc["cells"] else None,
        "underestimated_cells": acc["under_cells"],
        "under_raw_residual_mean": acc["under_raw_sum"] / un,
        "under_effective_residual_mean": acc["under_effective_sum"] / un,
        "under_cell_mae_before": acc["under_before"] / un,
        "under_cell_mae_after": acc["under_after"] / un,
        "overestimated_cells": acc["over_cells"],
        "over_raw_residual_mean": acc["over_raw_sum"] / on,
        "over_effective_residual_mean": acc["over_effective_sum"] / on,
        "over_cell_mae_before": acc["over_before"] / on,
        "over_cell_mae_after": acc["over_after"] / on,
        "mean_abs_count_error_before": acc["error_abs_sum"] / n,
        "mean_abs_effective_correction": acc["correction_abs_sum"] / n,
        "correction_to_error_ratio": acc["correction_abs_sum"] / max(acc["error_abs_sum"], 1e-12),
        "clamp_changed_pixel_fraction": acc["clamp_changed_pixels"] / max(acc["refined_pixels"], 1),
    }


def read_history(path):
    p = Path(path)
    if not p.exists():
        return {"available": False, "message": f"Training history not found: {p}"}
    with p.open("r", encoding="utf-8") as f:
        history = json.load(f)
    rows = []
    for epoch in history:
        train = epoch.get("train", {})
        val = epoch.get("validation", {})
        rows.append({
            "epoch": epoch.get("epoch"),
            "train_total": train.get("total"),
            "train_error": train.get("error"),
            "train_rank": train.get("rank"),
            "train_refine": train.get("refine"),
            "train_selected_mse": train.get("selected_mse"),
            "train_selected_count_l1": train.get("selected_count_l1"),
            "train_global": train.get("global"),
            "validation_mae": val.get("mae"),
            "validation_rmse": val.get("rmse"),
            "learning_rate": epoch.get("learning_rate"),
            "teacher_probability": epoch.get("teacher_probability"),
        })
    return {"available": True, "epochs": rows}


@torch.inference_mode()
def main():
    args = parse_args()
    for budget in args.budgets:
        if not 0.0 < budget < 1.0:
            raise ValueError(f"Budget must be between 0 and 1, got {budget}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = ESRNet(
        pretrained=False, freeze_early=True, refine_width=args.refine_width
    ).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    model.eval()
    model.coarse.eval()

    loader = get_loader(
        data_dir=args.data_dir,
        split="test",
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    total = len(loader)
    limit = min(args.max_test_batches, total) if args.max_test_batches else total
    print(f"Test crops: {total}; evaluating: {limit}")

    accumulators = {
        f"error_aware_{int(round(b * 100)):02d}%": new_accumulator()
        for b in args.budgets
    }

    for batch_index, (images, target_density, _counts) in enumerate(loader, start=1):
        if args.max_test_batches is not None and batch_index > args.max_test_batches:
            break

        images = images.to(device, non_blocking=True)
        target_density = target_density.to(device, non_blocking=True).float()

        # Encoder output features are half precision on CUDA.
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            outputs = model.encode(images)

        coarse = outputs["density"].float()
        features = outputs["stride4_features"]
        scores = outputs["error_scores"].float()

        coarse_cells = cell_sum(coarse)
        target_cells = cell_sum(target_density)
        before_error = coarse_cells - target_cells

        for budget in args.budgets:
            mask = select_topk(scores, budget)
            selected = mask[:, 0].bool().nonzero(as_tuple=False)
            if selected.numel() == 0:
                continue

            density_s4 = F.interpolate(
                coarse / 4.0, scale_factor=2, mode="bilinear", align_corners=False
            )
            feat_windows, den_windows, batch_idx, cell_y, cell_x = (
                model._extract_selected_windows(features, density_s4, selected)
            )

            # Run the direct RefineNet call with autocast because feature windows
            # are FP16 while the model parameters are FP32.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                raw_s4 = model.refiner(feat_windows, den_windows)

            raw_s8 = F.avg_pool2d(raw_s4, 2, 2) * 4.0
            raw_central = raw_s8[:, :, 4:8, 4:8].float()
            raw_cell_counts = raw_central.sum(dim=(1, 2, 3))

            raw_map = torch.zeros_like(before_error)
            for i in range(selected.shape[0]):
                b = int(batch_idx[i])
                y = int(cell_y[i])
                x = int(cell_x[i])
                raw_map[b, 0, y, x] = raw_cell_counts[i]

            # IMPORTANT FIX: apply_refine invokes the refiner internally.
            # It must also run under autocast to match the FP16 encoder features.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                refined = model.apply_refine(coarse, features, mask)

            refined = refined.float()
            after_cells = cell_sum(refined)
            effective_map = after_cells - coarse_cells

            clamp_changed = 0
            for i in range(selected.shape[0]):
                b = int(batch_idx[i])
                y0 = int(cell_y[i]) * 4
                x0 = int(cell_x[i]) * 4
                coarse_patch = coarse[b:b + 1, :, y0:y0 + 4, x0:x0 + 4]
                raw_patch = raw_central[i:i + 1]
                clamp_changed += int((coarse_patch + raw_patch < 0).sum().item())

            key = f"error_aware_{int(round(budget * 100)):02d}%"
            add_batch(
                accumulators[key], raw_map, effective_map,
                before_error, after_cells, mask, clamp_changed
            )

        if batch_index == 1 or batch_index % 250 == 0:
            print(f"Processed {batch_index}/{limit} crops")

    results = {name: finalize(acc) for name, acc in accumulators.items()}
    history = read_history(args.history)
    payload = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "test_split": "full" if args.max_test_batches is None else "partial",
        "budgets": args.budgets,
        "residual_statistics": results,
        "training_history": history,
        "notes": [
            "Raw residual is summed over the central 4x4 stride-8 density region after RefineNet output pooling.",
            "Effective residual is measured from final density after apply_refine and its nonnegative clamp.",
            "Correction-to-error ratio is the sum of absolute effective corrections divided by the sum of absolute coarse cell errors over selected cells.",
        ],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "stage2_residual_distribution.json"
    csv_path = out_dir / "stage2_residual_distribution.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    columns = list(next(iter(results.values())).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["budget", *columns])
        writer.writeheader()
        for name, metrics in results.items():
            writer.writerow({"budget": name.rsplit("_", 1)[-1], **metrics})

    print("\nRESIDUAL DISTRIBUTION")
    print("-" * 118)
    print(
        f"{'Budget':<9} {'Raw mean':>11} {'Raw |mean|':>12} {'Raw std':>11} "
        f"{'Raw min':>11} {'Raw max':>11} {'Eff. |mean|':>12} "
        f"{'Corr/error':>12} {'Clamp %':>10}"
    )
    print("-" * 118)
    for name, m in results.items():
        budget = name.rsplit("_", 1)[-1]
        print(
            f"{budget:<9} {m['raw_residual_mean']:>11.6f} "
            f"{m['raw_residual_mean_abs']:>12.6f} "
            f"{m['raw_residual_std']:>11.6f} "
            f"{m['raw_residual_min']:>11.6f} "
            f"{m['raw_residual_max']:>11.6f} "
            f"{m['effective_residual_mean_abs']:>12.6f} "
            f"{m['correction_to_error_ratio']:>12.6f} "
            f"{m['clamp_changed_pixel_fraction']:>9.2%}"
        )

    print("\nUNDERCOUNTED VS OVERCOUNTED SELECTED CELLS")
    for name, m in results.items():
        print(f"\n{name}")
        print(
            f"  Underestimated: {m['underestimated_cells']} cells | "
            f"raw correction {m['under_raw_residual_mean']:+.6f} | "
            f"effective correction {m['under_effective_residual_mean']:+.6f} | "
            f"cell MAE {m['under_cell_mae_before']:.6f} -> {m['under_cell_mae_after']:.6f}"
        )
        print(
            f"  Overestimated:  {m['overestimated_cells']} cells | "
            f"raw correction {m['over_raw_residual_mean']:+.6f} | "
            f"effective correction {m['over_effective_residual_mean']:+.6f} | "
            f"cell MAE {m['over_cell_mae_before']:.6f} -> {m['over_cell_mae_after']:.6f}"
        )

    print("\nTRAINING LOSS HISTORY")
    if history["available"]:
        for row in history["epochs"]:
            print(
                f"Epoch {row['epoch']}: refine={row['train_refine']} "
                f"(density_mse={row['train_selected_mse']}, "
                f"selected_count_l1={row['train_selected_count_l1']}), "
                f"global={row['train_global']}, val_MAE={row['validation_mae']}"
            )
    else:
        print(history["message"])

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    if args.max_test_batches is not None:
        print("WARNING: Partial test run; do not report these as final statistics.")


if __name__ == "__main__":
    main()
