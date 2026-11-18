"""
Oracle-selection diagnostic for Error-Aware Selective Refinement.

Run from the project root:
    python error_aware/diagnose_stage2.py

Optional smoke test:
    python error_aware/diagnose_stage2.py --max-test-batches 20

This compares:
  1. coarse_only: no refinement
  2. error_aware: select cells using ErrorHead predictions
  3. random: select the same budget randomly
  4. oracle: select cells using ground-truth coarse cell errors

Oracle is an evaluation-only upper-bound selector. It uses test labels and
must never be used for real inference or training.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from data import get_loader
from model import ESRNet, cell_sum, select_topk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose ErrorHead ranking and RefineNet using oracle selection."
    )
    parser.add_argument("--data-dir", default="error_aware/processed")
    parser.add_argument(
        "--checkpoint", default="error_aware/checkpoints/stage2_best.pth"
    )
    parser.add_argument("--output-dir", default="error_aware/diagnostics")
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.10, 0.25, 0.40])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--refine-width", type=int, default=96)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get(
        "model_state_dict", checkpoint.get("model", checkpoint)
    )
    model.load_state_dict(state, strict=True)
    print(
        f"Loaded checkpoint: {path} | "
        f"epoch={checkpoint.get('epoch', 'unknown')} | "
        f"best_val_mae={checkpoint.get('best_val_mae', 'unknown')}"
    )


def rankdata_average_ties(values):
    """Return 1-based average ranks, handling ties without scipy."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        # Ranks occupied are start+1 through end; average them.
        ranks_sorted[start:end] = ((start + 1) + end) / 2.0
        start = end

    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def spearman_for_crop(predicted_scores, true_errors):
    x = np.asarray(predicted_scores, dtype=np.float64).reshape(-1)
    y = np.asarray(true_errors, dtype=np.float64).reshape(-1)

    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None

    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if denominator == 0:
        return None
    return float(np.sum(rx * ry) / denominator)


def new_metric_store():
    return {
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "n": 0,
        "empty_abs_sum": 0.0,
        "empty_n": 0,
        "nonempty_abs_sum": 0.0,
        "nonempty_n": 0,
        "seconds": 0.0,
        "coverage_sum": 0.0,
        "coverage_n": 0,
    }


def add_count_metrics(store, prediction, target, elapsed=0.0):
    prediction = prediction.float().reshape(-1)
    target = target.float().reshape(-1)
    error = prediction - target
    absolute = error.abs()

    store["abs_sum"] += absolute.sum().item()
    store["sq_sum"] += error.square().sum().item()
    store["n"] += error.numel()
    store["seconds"] += elapsed

    empty = target == 0
    nonempty = target > 0
    if empty.any():
        store["empty_abs_sum"] += absolute[empty].sum().item()
        store["empty_n"] += int(empty.sum().item())
    if nonempty.any():
        store["nonempty_abs_sum"] += absolute[nonempty].sum().item()
        store["nonempty_n"] += int(nonempty.sum().item())


def finish_count_metrics(store):
    n = max(store["n"], 1)
    return {
        "mae": store["abs_sum"] / n,
        "rmse": (store["sq_sum"] / n) ** 0.5,
        "empty_mae": (
            store["empty_abs_sum"] / store["empty_n"]
            if store["empty_n"] else None
        ),
        "nonempty_mae": (
            store["nonempty_abs_sum"] / store["nonempty_n"]
            if store["nonempty_n"] else None
        ),
        "samples": store["n"],
        "empty_samples": store["empty_n"],
        "nonempty_samples": store["nonempty_n"],
        "mean_inference_ms_per_crop": store["seconds"] / n * 1000.0,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    for budget in args.budgets:
        if not 0.0 < budget < 1.0:
            raise ValueError(f"Budget must be between 0 and 1, got {budget}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

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

    stores = {"coarse_only": new_metric_store()}
    for budget in args.budgets:
        label = f"{int(round(budget * 100)):02d}%"
        for method in ("error_aware", "random", "oracle"):
            stores[f"{method}_{label}"] = new_metric_store()

    spearman_values = []
    coverage = {
        f"{method}_{int(round(budget * 100)):02d}%": []
        for budget in args.budgets
        for method in ("error_aware", "random", "oracle")
    }

    total = len(loader)
    limit = min(args.max_test_batches, total) if args.max_test_batches else total
    print(f"Test crops: {total}; evaluating: {limit}")
    print(f"Budgets: {[f'{b:.0%}' for b in args.budgets]}")

    for batch_index, (images, target_density, _counts) in enumerate(loader, start=1):
        if args.max_test_batches is not None and batch_index > args.max_test_batches:
            break

        images = images.to(device, non_blocking=True)
        target_density = target_density.to(device, non_blocking=True).float()

        sync(device)
        start = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            outputs = model.encode(images)
        sync(device)
        encode_seconds = time.perf_counter() - start

        coarse_density = outputs["density"]
        error_scores = outputs["error_scores"].float()
        stride4_features = outputs["stride4_features"]

        # Per-cell target: absolute difference between coarse and GT cell counts.
        true_cell_errors = (
            cell_sum(coarse_density.detach().float())
            - cell_sum(target_density)
        ).abs()

        rho = spearman_for_crop(
            error_scores[0, 0].detach().cpu().numpy(),
            true_cell_errors[0, 0].detach().cpu().numpy(),
        )
        if rho is not None:
            spearman_values.append(rho)

        true_count = target_density.sum(dim=(1, 2, 3))
        coarse_count = coarse_density.float().sum(dim=(1, 2, 3))
        add_count_metrics(
            stores["coarse_only"], coarse_count, true_count, encode_seconds
        )

        total_true_cell_error = true_cell_errors.sum().item()

        for budget in args.budgets:
            label = f"{int(round(budget * 100)):02d}%"

            predicted_mask = select_topk(error_scores, budget)
            oracle_mask = select_topk(true_cell_errors, budget)

            generator = torch.Generator(device=device)
            generator.manual_seed(
                args.seed + batch_index * 1009 + int(round(budget * 10000))
            )
            random_scores = torch.rand(
                error_scores.shape, device=device, generator=generator
            )
            random_mask = select_topk(random_scores, budget)

            masks = {
                "error_aware": predicted_mask,
                "random": random_mask,
                "oracle": oracle_mask,
            }

            for method, mask in masks.items():
                # Fraction of the crop's actual coarse cell error selected.
                selected_error = (true_cell_errors * mask.float()).sum().item()
                selected_coverage = (
                    selected_error / total_true_cell_error
                    if total_true_cell_error > 0 else 0.0
                )
                coverage[f"{method}_{label}"].append(selected_coverage)

                sync(device)
                start = time.perf_counter()
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    refined_density = model.apply_refine(
                        coarse_density, stride4_features, mask
                    )
                sync(device)
                refine_seconds = time.perf_counter() - start

                refined_count = refined_density.float().sum(dim=(1, 2, 3))
                add_count_metrics(
                    stores[f"{method}_{label}"],
                    refined_count,
                    true_count,
                    encode_seconds + refine_seconds,
                )

        if batch_index == 1 or batch_index % 250 == 0:
            print(f"Processed {batch_index}/{limit} crops")

    results = {
        name: finish_count_metrics(store) for name, store in stores.items()
    }

    ranking_summary = {
        "mean_per_crop_spearman": (
            float(np.mean(spearman_values)) if spearman_values else None
        ),
        "median_per_crop_spearman": (
            float(np.median(spearman_values)) if spearman_values else None
        ),
        "valid_crops": len(spearman_values),
        "evaluated_crops": limit,
    }

    coverage_summary = {}
    for name, values in coverage.items():
        coverage_summary[name] = {
            "mean_error_coverage": float(np.mean(values)) if values else None,
            "median_error_coverage": float(np.median(values)) if values else None,
            "crops": len(values),
        }

    payload = {
        "checkpoint": args.checkpoint,
        "test_split": "full" if args.max_test_batches is None else "partial",
        "seed": args.seed,
        "budgets": args.budgets,
        "ranking": ranking_summary,
        "error_coverage": coverage_summary,
        "count_metrics": results,
        "notes": [
            "Oracle selection uses ground-truth test labels and is evaluation-only.",
            "Error coverage is the fraction of the sum of true coarse cell-count errors selected.",
            "Spearman is calculated per crop over its 256 cells; crops with constant scores or targets are excluded.",
        ],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage2_oracle_diagnostics.json"
    csv_path = output_dir / "stage2_oracle_metrics.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    columns = [
        "method", "mae", "rmse", "empty_mae", "nonempty_mae",
        "samples", "empty_samples", "nonempty_samples",
        "mean_inference_ms_per_crop",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for method, metrics in results.items():
            writer.writerow({"method": method, **metrics})

    print("\nCOUNT RESULTS (crop-level)")
    print("-" * 104)
    print(
        f"{'Method':<22} {'MAE':>9} {'RMSE':>9} "
        f"{'Empty MAE':>11} {'Nonempty MAE':>14} {'ms/crop':>11}"
    )
    print("-" * 104)
    for method, metrics in results.items():
        empty = (
            f"{metrics['empty_mae']:.4f}"
            if metrics["empty_mae"] is not None else "N/A"
        )
        nonempty = (
            f"{metrics['nonempty_mae']:.4f}"
            if metrics["nonempty_mae"] is not None else "N/A"
        )
        print(
            f"{method:<22} {metrics['mae']:>9.4f} {metrics['rmse']:>9.4f} "
            f"{empty:>11} {nonempty:>14} "
            f"{metrics['mean_inference_ms_per_crop']:>11.2f}"
        )

    print("\nERROR-HEAD RANKING")
    print(
        f"Mean per-crop Spearman: {ranking_summary['mean_per_crop_spearman']}"
    )
    print(
        f"Median per-crop Spearman: {ranking_summary['median_per_crop_spearman']}"
    )
    print(
        f"Valid crops: {ranking_summary['valid_crops']}/"
        f"{ranking_summary['evaluated_crops']}"
    )

    print("\nTRUE CELL-ERROR COVERAGE")
    for name, metrics in coverage_summary.items():
        mean = metrics["mean_error_coverage"]
        print(f"{name:<22} {mean:.4f}" if mean is not None else f"{name:<22} N/A")

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    if args.max_test_batches is not None:
        print("WARNING: Partial test run; do not report these as final metrics.")


if __name__ == "__main__":
    main()
