"""
GPU-only multiplicative Stage-2 diagnostic.

CUDA is mandatory; this script never silently falls back to CPU.
Run from project root with .venv activated.

Example:
python error_aware/diagnose_stage2_multiplicative.py `
  --manifest error_aware/processed/stage2_train20_holdout.jsonl `
  --checkpoint error_aware/checkpoints_stage2_multiplicative_stability/stage2_best.pth `
  --budgets 0.10 0.25 0.40
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from data import ErrorAwareCrowdDataset
from model_multiplicative import ESRNet, cell_sum, select_topk


def parse_args():
    p = argparse.ArgumentParser(description="GPU-only multiplicative Stage-2 diagnostic")
    p.add_argument("--manifest", default="error_aware/processed/stage2_train20_holdout.jsonl")
    p.add_argument("--dataset-root", default="dataset/VisDrone2019-DET-train")
    p.add_argument("--checkpoint", default="error_aware/checkpoints_stage2_multiplicative_stability/stage2_best.pth")
    p.add_argument("--output-dir", default="error_aware/diagnostics_multiplicative")
    p.add_argument("--budgets", nargs="+", type=float, default=[0.10, 0.25, 0.40])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--refine-width", type=int, default=96)
    p.add_argument("--no-amp", action="store_true", help="Disable CUDA FP16 autocast")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rankdata_average_ties(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = ((start + 1) + end) / 2.0
        start = end
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    value = np.corrcoef(rankdata_average_ties(x), rankdata_average_ties(y))[0, 1]
    return float(value) if np.isfinite(value) else None


def new_store():
    return {k: [] for k in (
        "abs_errors", "squared_errors", "predictions", "targets",
        "empty_abs_errors", "nonempty_abs_errors",
        "empty_predictions", "nonempty_predictions",
        "empty_targets", "nonempty_targets",
    )}


def add_counts(store, predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    errors = np.abs(predictions - targets)
    store["predictions"].extend(predictions.tolist())
    store["targets"].extend(targets.tolist())
    store["abs_errors"].extend(errors.tolist())
    store["squared_errors"].extend(((predictions - targets) ** 2).tolist())
    empty = targets < 0.5
    nonempty = ~empty
    for key, mask in (("empty", empty), ("nonempty", nonempty)):
        store[f"{key}_abs_errors"].extend(errors[mask].tolist())
        store[f"{key}_predictions"].extend(predictions[mask].tolist())
        store[f"{key}_targets"].extend(targets[mask].tolist())


def summarize(store):
    mean = lambda x: float(np.mean(x)) if x else None
    rmse = lambda x: float(np.sqrt(np.mean(x))) if x else None
    pm, tm = mean(store["predictions"]), mean(store["targets"])
    return {
        "samples": len(store["abs_errors"]),
        "mae": mean(store["abs_errors"]),
        "rmse": rmse(store["squared_errors"]),
        "mean_prediction": pm,
        "mean_target": tm,
        "mean_bias_prediction_minus_target": pm - tm if pm is not None and tm is not None else None,
        "empty": {
            "samples": len(store["empty_abs_errors"]),
            "mae": mean(store["empty_abs_errors"]),
            "mean_prediction": mean(store["empty_predictions"]),
            "mean_target": mean(store["empty_targets"]),
        },
        "nonempty": {
            "samples": len(store["nonempty_abs_errors"]),
            "mae": mean(store["nonempty_abs_errors"]),
            "mean_prediction": mean(store["nonempty_predictions"]),
            "mean_target": mean(store["nonempty_targets"]),
        },
    }


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {path}")
    print(f"  epoch={ckpt.get('epoch', 'unknown')}")
    print(f"  best_val_mae={ckpt.get('best_val_mae', 'unknown')}")
    return ckpt


def main():
    args = parse_args()

    # Never run this long diagnostic on CPU by accident.
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError(
            "CUDA is unavailable in this Python environment. Activate the project "
            ".venv and verify torch.cuda.is_available(). This script refuses CPU fallback."
        )
    for b in args.budgets:
        if not 0.0 < b <= 1.0:
            raise ValueError(f"Budget must be in (0,1], got {b}")

    set_seed(args.seed)
    device = torch.device("cuda:0")
    amp = not args.no_amp
    torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print("MULTIPLICATIVE STAGE-2 GPU DIAGNOSTIC")
    print("=" * 70)
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__} | CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"AMP: {'enabled' if amp else 'disabled'}")
    print(f"Manifest: {args.manifest}")
    print(f"Checkpoint: {args.checkpoint}")

    dataset = ErrorAwareCrowdDataset(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        crop_size=512,
        density_stride=8,
        gaussian_sigma=3.0,
        return_metadata=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    model = ESRNet(pretrained=False, freeze_early=True, refine_width=args.refine_width).to(device)
    ckpt = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    methods = ["error_aware", "random", "density", "oracle"]
    stores = {"coarse_only": new_store()}
    for budget in args.budgets:
        label = f"{int(round(budget * 100)):02d}%"
        for method in methods:
            stores[f"{method}_{label}"] = new_store()

    coverage_values = {
        f"{method}_{int(round(budget * 100)):02d}%": []
        for budget in args.budgets for method in methods
    }
    spearman_values, gain_values, add_mass_values = [], [], []
    total = len(loader)
    limit = min(args.max_batches, total) if args.max_batches else total
    print(f"Manifest crops: {total}; evaluating: {limit}")
    print(f"Budgets: {[f'{b:.0%}' for b in args.budgets]}")
    print("Batch size: 1 | CUDA inference mode")

    started = time.perf_counter()
    with torch.inference_mode():
        for index, (images, target_density, _counts) in enumerate(loader):
            if args.max_batches is not None and index >= args.max_batches:
                break
            images = images.to(device, non_blocking=True)
            target_density = target_density.to(device, non_blocking=True).float()

            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                encoded = model.encode(images)

            coarse = encoded["density"].float()
            scores = encoded["error_scores"].float()
            features = encoded["stride4_features"]
            true_errors = (cell_sum(coarse) - cell_sum(target_density)).abs()

            rho = spearman(
                scores[0, 0].detach().cpu().numpy(),
                true_errors[0, 0].detach().cpu().numpy(),
            )
            if rho is not None:
                spearman_values.append(rho)

            target_count = target_density.sum((1, 2, 3))
            coarse_count = coarse.sum((1, 2, 3))
            add_counts(
                stores["coarse_only"],
                coarse_count.detach().cpu().numpy(),
                target_count.detach().cpu().numpy(),
            )

            for budget in args.budgets:
                label = f"{int(round(budget * 100)):02d}%"
                k = max(1, int(round(256 * budget)))

                masks = {
                    "error_aware": select_topk(scores, budget),
                    "density": select_topk(cell_sum(coarse), budget),
                    "oracle": select_topk(true_errors, budget),
                }
                random_mask = torch.zeros_like(masks["error_aware"])
                rand_idx = torch.randperm(256, device=device)[:k]
                random_mask.view(1, 1, 256).scatter_(2, rand_idx.view(1, 1, -1), True)
                masks["random"] = random_mask

                # All four selector masks are refined in one call per budget.
                mask_batch = torch.cat([masks[m].float() for m in methods], dim=0).bool()
                error_total = true_errors.sum()
                selected = (
                    mask_batch.float() * true_errors.expand(len(methods), -1, -1, -1)
                ).sum((1, 2, 3))
                coverage = torch.where(
                    error_total > 1e-12,
                    selected / error_total.clamp_min(1e-12),
                    torch.zeros_like(selected),
                ).detach().cpu().numpy()
                for method, value in zip(methods, coverage):
                    coverage_values[f"{method}_{label}"].append(float(value))

                coarse_batch = coarse.expand(len(methods), -1, -1, -1).contiguous()
                feature_batch = features.expand(len(methods), -1, -1, -1).contiguous()
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    refined_batch, aux = model.apply_refine(
                        coarse_batch, feature_batch, mask_batch, return_aux=True
                    )

                refined_counts = refined_batch.float().sum((1, 2, 3)).detach().cpu().numpy()
                target_np = target_count.detach().cpu().numpy()
                for method_idx, method in enumerate(methods):
                    add_counts(
                        stores[f"{method}_{label}"],
                        refined_counts[method_idx:method_idx + 1],
                        target_np,
                    )

                # aux["selected"] columns are (batch_idx, cell_y, cell_x).
                # Batch index zero corresponds to the error-aware selector.
                if aux is not None:
                    is_error_aware = aux["selected"][:, 0] == 0
                    if is_error_aware.any():
                        g = aux["gain"][is_error_aware].float()
                        a = aux["add"][is_error_aware].float()
                        gain_values.extend(g.mean((1, 2, 3)).detach().cpu().tolist())
                        add_mass_values.extend(a.sum((1, 2, 3)).detach().cpu().tolist())

            if (index + 1) % 250 == 0 or index + 1 == limit:
                elapsed = time.perf_counter() - started
                speed = (index + 1) / max(elapsed, 1e-9)
                eta = (limit - index - 1) / max(speed, 1e-9)
                print(
                    f"Processed {index + 1}/{limit} | {speed:.1f} crops/s | "
                    f"ETA {eta / 60:.1f} min"
                )

    elapsed = time.perf_counter() - started
    results = {
        "device": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "amp_enabled": amp,
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_val_mae": ckpt.get("best_val_mae"),
        "evaluated_crops": limit,
        "budgets": args.budgets,
        "metrics": {name: summarize(store) for name, store in stores.items()},
        "ranking": {
            "mean_per_crop_spearman": float(np.mean(spearman_values)) if spearman_values else None,
            "median_per_crop_spearman": float(np.median(spearman_values)) if spearman_values else None,
            "valid_crops": len(spearman_values),
        },
        "error_coverage": {
            name: float(np.mean(vals)) if vals else None
            for name, vals in coverage_values.items()
        },
        "refiner_output": {
            "selected_error_aware_cells": len(gain_values),
            "gain_mean": float(np.mean(gain_values)) if gain_values else None,
            "gain_std": float(np.std(gain_values)) if gain_values else None,
            "gain_min": float(np.min(gain_values)) if gain_values else None,
            "gain_max": float(np.max(gain_values)) if gain_values else None,
            "additive_mass_mean_per_selected_cell": float(np.mean(add_mass_values)) if add_mass_values else None,
            "additive_mass_max_per_selected_cell": float(np.max(add_mass_values)) if add_mass_values else None,
        },
        "elapsed_seconds": elapsed,
        "notes": [
            "Oracle selection uses ground-truth labels and is diagnostic only.",
            "Error coverage is the fraction of absolute coarse cell-count error inside selected cells.",
            "Spearman is computed per crop over 256 cells; constant arrays are excluded.",
            "Gain is averaged over the 4x4 density pixels in a selected cell.",
            "Additive mass is summed over the 4x4 density pixels in a selected cell.",
        ],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "stage2_multiplicative_diagnostics.json"
    csv_path = out_dir / "stage2_multiplicative_metrics.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "samples", "mae", "rmse", "empty_mae", "nonempty_mae",
            "mean_prediction", "mean_target", "mean_bias",
        ])
        for name, m in results["metrics"].items():
            writer.writerow([
                name, m["samples"], m["mae"], m["rmse"],
                m["empty"]["mae"], m["nonempty"]["mae"],
                m["mean_prediction"], m["mean_target"],
                m["mean_bias_prediction_minus_target"],
            ])

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Elapsed: {elapsed / 60:.1f} minutes")
    print(f"Mean per-crop Spearman: {results['ranking']['mean_per_crop_spearman']}")
    print(f"Median per-crop Spearman: {results['ranking']['median_per_crop_spearman']}")
    print("\nError coverage:")
    for name, value in results["error_coverage"].items():
        print(f"  {name}: {value:.4f}" if value is not None else f"  {name}: N/A")
    print("\nCount metrics:")
    for name, m in results["metrics"].items():
        print(
            f"  {name:20s} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
            f"empty={m['empty']['mae']:.4f} nonempty={m['nonempty']['mae']:.4f}"
        )
    print("\nRefiner output:")
    for key, value in results["refiner_output"].items():
        print(f"  {key}: {value}")
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
