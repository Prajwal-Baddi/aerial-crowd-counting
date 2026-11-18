"""
Diagnose whether deployable scores rank cells by actual coarse-count error magnitude.

Run from the project root:
python error_aware/diagnose_error_magnitude.py `
  --checkpoint error_aware/checkpoints_stage1_holdout80/coarse_best.pth `
  --error-head-checkpoint error_aware/checkpoints_error_head/error_head_best.pth `
  --data-dir error_aware/processed `
  --dataset-root dataset/VisDrone2019-DET-train

This is evaluation only: it does not train or modify checkpoints.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ErrorAwareCrowdDataset
from model_gated import ESRNet, cell_sum
from selection_baselines import DELTA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--error-head-checkpoint", required=True)
    p.add_argument("--data-dir", default="error_aware/processed")
    p.add_argument("--dataset-root", default="dataset/VisDrone2019-DET-train")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--budgets", default="3,6,13,26")
    p.add_argument("--quantiles", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="error_aware/analysis/error_magnitude_diagnostic.csv")
    return p.parse_args()


@torch.no_grad()
def collect(model, loader, device, amp):
    model.eval()
    all_c, all_g, all_p = [], [], []
    for images, gt_density, _ in loader:
        images = images.to(device, non_blocking=True)
        gt_density = gt_density.to(device, non_blocking=True).float()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            coarse_out = model.coarse(images)
            logits = model.error_head(
                coarse_out["context"],
                coarse_out["density"],
                coarse_out["stride4_features"],
            )
        coarse_cells = cell_sum(coarse_out["density"].float())[:, 0]
        gt_cells = cell_sum(gt_density)[:, 0]
        probs = logits.float().softmax(1)
        all_c.append(coarse_cells.cpu().numpy())
        all_g.append(gt_cells.cpu().numpy())
        all_p.append(probs.cpu().numpy())
    if not all_c:
        raise RuntimeError("Validation loader produced no batches.")
    return np.concatenate(all_c), np.concatenate(all_g), np.concatenate(all_p)


def average_rank(x):
    """1-based average ranks for ties; stable and dependency-free."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def spearman(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(average_rank(x), average_rank(y))[0, 1])


def topk_mask(scores, k, tie_noise):
    flat = np.asarray(scores, dtype=np.float64).reshape(scores.shape[0], -1)
    if not 1 <= k <= flat.shape[1]:
        raise ValueError(f"k={k} is outside [1, {flat.shape[1]}]")
    indices = np.argpartition(flat + tie_noise, flat.shape[1] - k, axis=1)[:, -k:]
    mask = np.zeros(flat.shape, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=1)
    return mask


def quantile_report(score, target, bins):
    """Global score quantiles; reports mean/median actual error in each bin."""
    s = np.asarray(score).reshape(-1)
    e = np.asarray(target).reshape(-1)
    edges = np.quantile(s, np.linspace(0, 1, bins + 1))
    # Duplicate edges can occur for discrete/constant scores. Rank-based bins
    # keep the requested number of approximately equal-sized groups.
    order = np.argsort(s, kind="mergesort")
    chunks = np.array_split(order, bins)
    rows = []
    for i, idx in enumerate(chunks):
        if len(idx):
            rows.append({
                "bin": i + 1,
                "cells": len(idx),
                "score_min": float(s[idx].min()),
                "score_max": float(s[idx].max()),
                "mean_abs_error": float(e[idx].mean()),
                "median_abs_error": float(np.median(e[idx])),
                "material_error_rate": float((e[idx] >= DELTA).mean()),
            })
    return rows


def topk_metrics(mask, error):
    selected = (mask * error).sum(axis=1)
    total = error.sum(axis=1)
    valid = total > 0
    return {
        "selected_error_share_macro": float((selected[valid] / total[valid]).mean()) if valid.any() else float("nan"),
        "selected_error_share_micro": float(selected.sum() / total.sum()) if total.sum() > 0 else float("nan"),
        "mean_selected_abs_error_per_crop": float(selected.mean()),
    }


def main():
    args = parse_args()
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    if args.quantiles < 2:
        raise ValueError("--quantiles must be at least 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device} | AMP: {amp} | material threshold: {DELTA}")

    model = ESRNet(pretrained=False, freeze_early=True).to(device)
    coarse_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.coarse.load_state_dict(coarse_ckpt.get("model_state_dict", coarse_ckpt), strict=True)
    head_ckpt = torch.load(args.error_head_checkpoint, map_location=device, weights_only=False)
    head_state = head_ckpt.get("error_head_state_dict", head_ckpt)
    model.error_head.load_state_dict(head_state, strict=True)
    model.eval()

    print("\nCheckpoint metadata (if present):")
    for label, ckpt in (("coarse", coarse_ckpt), ("ErrorHead", head_ckpt)):
        if isinstance(ckpt, dict):
            fields = {k: ckpt[k] for k in ("epoch", "best_val_ap", "best_val_loss", "val_ap", "config")
                      if k in ckpt and isinstance(ckpt[k], (str, int, float, bool, dict, list, tuple))}
            print(f"  {label}: {fields if fields else 'no standard metadata fields found'}")

    val_path = Path(args.data_dir) / "validation.jsonl"
    dataset = ErrorAwareCrowdDataset(
        val_path, dataset_root=args.dataset_root, crop_size=512,
        density_stride=8, gaussian_sigma=3.0,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    print(f"\nValidation crops: {len(dataset)}")
    coarse, gt, probs = collect(model, loader, device, amp)
    if coarse.shape != gt.shape or coarse.ndim != 3:
        raise ValueError(f"Expected matching [N,H,W] maps, got {coarse.shape}, {gt.shape}")
    if probs.shape != (coarse.shape[0], 3, *coarse.shape[1:]):
        raise ValueError(f"Unexpected ErrorHead probability shape: {probs.shape}")

    error = np.abs(coarse - gt)
    scores = {
        "density_abs_coarse": np.abs(coarse),
        "head_material_prob": probs[:, 1] + probs[:, 2],
        "head_increase_prob": probs[:, 1],
        "head_decrease_prob": probs[:, 2],
        "oracle_abs_error_non_deployable": error,
    }

    print("\nGLOBAL RANKING AGAINST ACTUAL ABSOLUTE ERROR")
    print(f"{'score':<36}{'Spearman':>12}{'mean score':>14}{'mean |err|':>14}")
    print("-" * 76)
    flat_error = error.reshape(-1)
    global_rows = []
    for name, score in scores.items():
        rho = spearman(score, error)
        row = {
            "section": "global",
            "score": name,
            "spearman": rho,
            "mean_score": float(np.mean(score)),
            "mean_abs_error": float(error.mean()),
        }
        global_rows.append(row)
        print(f"{name:<36}{rho:>12.4f}{np.mean(score):>14.5f}{error.mean():>14.5f}")

    print("\nPER-CROP RANKING CONSISTENCY")
    print(f"{'score':<36}{'median rho':>14}{'mean rho':>12}{'positive crops':>17}")
    print("-" * 79)
    crop_rows = []
    for name, score in scores.items():
        rhos = np.array([spearman(score[i], error[i]) for i in range(len(error))], dtype=float)
        finite = rhos[np.isfinite(rhos)]
        row = {
            "section": "per_crop",
            "score": name,
            "median_spearman": float(np.median(finite)) if len(finite) else float("nan"),
            "mean_spearman": float(np.mean(finite)) if len(finite) else float("nan"),
            "positive_rho_crops": int((finite > 0).sum()),
            "valid_crops": int(len(finite)),
        }
        crop_rows.append(row)
        print(f"{name:<36}{row['median_spearman']:>14.4f}{row['mean_spearman']:>12.4f}"
              f"{row['positive_rho_crops']:>9}/{row['valid_crops']:<8}")

    print("\nSCORE QUANTILES: MEAN ACTUAL ABSOLUTE ERROR")
    quantile_rows = []
    for name, score in scores.items():
        print(f"\n{name}")
        print(f"{'bin (low to high)':<20}{'cells':>10}{'mean |err|':>14}{'median |err|':>15}{'material rate':>16}")
        for row in quantile_report(score, error, args.quantiles):
            row["section"] = "quantile"
            row["score"] = name
            quantile_rows.append(row)
            print(f"{row['bin']:>4}/{args.quantiles:<15}{row['cells']:>10,}"
                  f"{row['mean_abs_error']:>14.5f}{row['median_abs_error']:>15.5f}"
                  f"{100*row['material_error_rate']:>15.2f}%")

    print("\nPER-CROP TOP-K ERROR COVERAGE")
    rng = np.random.default_rng(args.seed)
    tie_noise = rng.random((coarse.shape[0], coarse.shape[1] * coarse.shape[2])) * 1e-12
    topk_rows = []
    print(f"{'score':<36}{'k':>5}{'macro coverage':>18}{'micro coverage':>18}{'mean error captured':>22}")
    print("-" * 99)
    for name, score in scores.items():
        for k in budgets:
            mask = topk_mask(score, k, tie_noise)
            metrics = topk_metrics(mask, error.reshape(error.shape[0], -1))
            row = {"section": "topk", "score": name, "k": k, **metrics}
            topk_rows.append(row)
            print(f"{name:<36}{k:>5}{100*metrics['selected_error_share_macro']:>17.2f}%"
                  f"{100*metrics['selected_error_share_micro']:>17.2f}%"
                  f"{metrics['mean_selected_abs_error_per_crop']:>22.5f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in global_rows + crop_rows + quantile_rows + topk_rows for key in row})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(global_rows + crop_rows + quantile_rows + topk_rows)
    print(f"\nSaved diagnostic CSV: {out_path}")
    print("\nNotes:")
    print("- Oracle is a non-deployable upper bound because it uses ground truth.")
    print("- Spearman and quantile reports test error-magnitude ranking, not refinement benefit.")
    print("- Top-k coverage is within each crop; AP/Spearman above are global across cells.")
    print("- This script evaluates the loaded checkpoints only; it performs no training.")


if __name__ == "__main__":
    main()
