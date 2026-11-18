"""
Refinement correction diagnostic.

Run from the project root:
    python error_aware/diagnose_refiner.py

Optional smoke test:
    python error_aware/diagnose_refiner.py --max-test-batches 20

Measures whether RefineNet corrections improve selected cell counts,
whether correction signs match the required direction, and whether
corrections are too small or too large.

No model weights are modified.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from data import get_loader
from model import ESRNet, cell_sum, select_topk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose the direction and effectiveness of RefineNet corrections."
    )
    parser.add_argument("--data-dir", default="error_aware/processed")
    parser.add_argument(
        "--checkpoint", default="error_aware/checkpoints/stage2_best.pth"
    )
    parser.add_argument("--output-dir", default="error_aware/diagnostics")
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.10, 0.25, 0.40])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--refine-width", type=int, default=96)
    return parser.parse_args()


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get(
        "model_state_dict", checkpoint.get("model", checkpoint)
    )
    model.load_state_dict(state, strict=True)
    print(
        f"Loaded checkpoint: {path} | epoch={checkpoint.get('epoch', 'unknown')}"
    )


def new_store():
    return {
        "selected_cells": 0,
        "selected_abs_before": 0.0,
        "selected_abs_after": 0.0,
        "selected_signed_before": 0.0,
        "selected_signed_after": 0.0,
        "selected_residual_sum": 0.0,
        "selected_abs_residual_sum": 0.0,
        "selected_under_count": 0,
        "selected_over_count": 0,
        "selected_correct_direction": 0,
        "selected_wrong_direction": 0,
        "selected_improved_cells": 0,
        "selected_worsened_cells": 0,
        "selected_unchanged_cells": 0,
        "all_cells": 0,
        "all_abs_before": 0.0,
        "all_abs_after": 0.0,
        "selected_crop_abs_before": 0.0,
        "selected_crop_abs_after": 0.0,
        "crop_count": 0,
    }


def add_cell_metrics(store, before, after, residual, mask, target):
    # All tensors are [B,1,16,16] and are evaluated in float32.
    before_error = before - target
    after_error = after - target
    selected = mask.bool()

    selected_count = int(selected.sum().item())
    if selected_count:
        b = before_error[selected]
        a = after_error[selected]
        r = residual[selected]

        store["selected_cells"] += selected_count
        store["selected_abs_before"] += b.abs().sum().item()
        store["selected_abs_after"] += a.abs().sum().item()
        store["selected_signed_before"] += b.sum().item()
        store["selected_signed_after"] += a.sum().item()
        store["selected_residual_sum"] += r.sum().item()
        store["selected_abs_residual_sum"] += r.abs().sum().item()

        under = b < -1e-6
        over = b > 1e-6
        correct = ((under & (r > 0)) | (over & (r < 0)))
        wrong = ((under & (r < 0)) | (over & (r > 0)))

        store["selected_under_count"] += int(under.sum().item())
        store["selected_over_count"] += int(over.sum().item())
        store["selected_correct_direction"] += int(correct.sum().item())
        store["selected_wrong_direction"] += int(wrong.sum().item())

        improvement = a.abs() - b.abs()
        store["selected_improved_cells"] += int((improvement < -1e-6).sum().item())
        store["selected_worsened_cells"] += int((improvement > 1e-6).sum().item())
        store["selected_unchanged_cells"] += int(
            (improvement.abs() <= 1e-6).sum().item()
        )

    store["all_cells"] += before_error.numel()
    store["all_abs_before"] += before_error.abs().sum().item()
    store["all_abs_after"] += after_error.abs().sum().item()


def finish_store(store):
    n = max(store["selected_cells"], 1)
    all_n = max(store["all_cells"], 1)
    directional_n = max(
        store["selected_under_count"] + store["selected_over_count"], 1
    )
    crop_n = max(store["crop_count"], 1)
    return {
        "selected_cells": store["selected_cells"],
        "selected_cell_mae_before": store["selected_abs_before"] / n,
        "selected_cell_mae_after": store["selected_abs_after"] / n,
        "selected_cell_mae_change": (
            store["selected_abs_after"] - store["selected_abs_before"]
        ) / n,
        "selected_signed_error_before": store["selected_signed_before"] / n,
        "selected_signed_error_after": store["selected_signed_after"] / n,
        "mean_residual_sum": store["selected_residual_sum"] / n,
        "mean_absolute_residual_sum": store["selected_abs_residual_sum"] / n,
        "underestimated_selected_cells": store["selected_under_count"],
        "overestimated_selected_cells": store["selected_over_count"],
        "correct_direction_rate": (
            store["selected_correct_direction"] / directional_n
        ),
        "wrong_direction_rate": (
            store["selected_wrong_direction"] / directional_n
        ),
        "improved_selected_cells": store["selected_improved_cells"],
        "worsened_selected_cells": store["selected_worsened_cells"],
        "unchanged_selected_cells": store["selected_unchanged_cells"],
        "selected_cells_improved_rate": (
            store["selected_improved_cells"] / n
        ),
        "selected_cells_worsened_rate": (
            store["selected_worsened_cells"] / n
        ),
        "all_cell_mae_before": store["all_abs_before"] / all_n,
        "all_cell_mae_after": store["all_abs_after"] / all_n,
        "all_cell_mae_change": (
            store["all_abs_after"] - store["all_abs_before"]
        ) / all_n,
        "crops": store["crop_count"],
    }


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
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    model.coarse.eval()

    loader = get_loader(
        data_dir=args.data_dir,
        split="test",
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    stores = {}
    for budget in args.budgets:
        label = f"{int(round(budget * 100)):02d}%"
        for method in ("error_aware", "random", "oracle"):
            stores[f"{method}_{label}"] = new_store()

    total = len(loader)
    limit = min(args.max_test_batches, total) if args.max_test_batches else total
    print(f"Test crops: {total}; evaluating: {limit}")

    for batch_index, (images, target_density, _counts) in enumerate(loader, start=1):
        if args.max_test_batches is not None and batch_index > args.max_test_batches:
            break

        images = images.to(device, non_blocking=True)
        target_density = target_density.to(device, non_blocking=True).float()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            outputs = model.encode(images)

        coarse = outputs["density"]
        scores = outputs["error_scores"].float()
        features = outputs["stride4_features"]

        # Cell counts and actual coarse count errors.
        coarse_cells = cell_sum(coarse.float())
        target_cells = cell_sum(target_density)
        true_cell_errors = (coarse_cells - target_cells).abs()

        for budget in args.budgets:
            label = f"{int(round(budget * 100)):02d}%"

            masks = {
                "error_aware": select_topk(scores, budget),
                "oracle": select_topk(true_cell_errors, budget),
            }

            generator = torch.Generator(device=device)
            generator.manual_seed(42 + batch_index * 1009 + int(budget * 10000))
            random_scores = torch.rand(
                scores.shape, device=device, generator=generator
            )
            masks["random"] = select_topk(random_scores, budget)

            for method, mask in masks.items():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    refined = model.apply_refine(coarse, features, mask)

                refined_cells = cell_sum(refined.float())
                # Effective change after clamp_min(0), not raw network output.
                effective_residual = refined_cells - coarse_cells

                key = f"{method}_{label}"
                add_cell_metrics(
                    stores[key],
                    coarse_cells,
                    refined_cells,
                    effective_residual,
                    mask,
                    target_cells,
                )
                stores[key]["crop_count"] += images.size(0)

        if batch_index == 1 or batch_index % 250 == 0:
            print(f"Processed {batch_index}/{limit} crops")

    results = {name: finish_store(store) for name, store in stores.items()}
    payload = {
        "checkpoint": args.checkpoint,
        "test_split": "full" if args.max_test_batches is None else "partial",
        "budgets": args.budgets,
        "results": results,
        "interpretation": {
            "selected_cell_mae_change": "Negative means selected cell count errors decreased.",
            "all_cell_mae_change": "Negative means mean absolute cell-count error decreased across the entire crop.",
            "correct_direction_rate": "Among selected cells with nonzero coarse error, fraction whose effective correction points toward the target count.",
            "note": "Residual is measured after applying the final nonnegative-density clamp.",
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage2_refiner_diagnostics.json"
    csv_path = output_dir / "stage2_refiner_diagnostics.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    columns = list(next(iter(results.values())).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "budget", *columns])
        writer.writeheader()
        for name, metrics in results.items():
            method, budget = name.rsplit("_", 1)
            writer.writerow({"method": method, "budget": budget, **metrics})

    print("\nREFINER CORRECTION DIAGNOSTICS")
    print("-" * 115)
    print(
        f"{'Method':<22} {'Sel MAE before':>14} {'Sel MAE after':>14} "
        f"{'Change':>10} {'Correct dir.':>13} {'Improved':>10} {'Worsened':>10}"
    )
    print("-" * 115)
    for name, m in results.items():
        print(
            f"{name:<22} "
            f"{m['selected_cell_mae_before']:>14.4f} "
            f"{m['selected_cell_mae_after']:>14.4f} "
            f"{m['selected_cell_mae_change']:>10.4f} "
            f"{m['correct_direction_rate']:>12.1%} "
            f"{m['selected_cells_improved_rate']:>9.1%} "
            f"{m['selected_cells_worsened_rate']:>9.1%}"
        )

    print("\nALL-CELL MAE CHANGE (negative is better)")
    for name, m in results.items():
        print(f"{name:<22} {m['all_cell_mae_change']:+.6f}")

    print("\nMEAN EFFECTIVE RESIDUAL SUM PER SELECTED CELL")
    for name, m in results.items():
        print(
            f"{name:<22} {m['mean_residual_sum']:+.6f} "
            f"(absolute magnitude {m['mean_absolute_residual_sum']:.6f})"
        )

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    if args.max_test_batches is not None:
        print("WARNING: Partial test run; do not report these as final metrics.")


if __name__ == "__main__":
    main()
