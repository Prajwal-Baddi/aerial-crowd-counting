"""Evaluate per-crop rank fusion between ErrorHead and coarse-density selection.

Example:
python error_aware/evaluate_score_fusion.py `
  --checkpoint error_aware/checkpoints_stage1_holdout80/coarse_best.pth `
  --error-head-checkpoint error_aware/checkpoints_error_head/error_head_best.pth `
  --data-dir error_aware/processed `
  --dataset-root dataset/VisDrone2019-DET-train `
  --lambdas 0.25,0.50,0.75

Scores are converted to within-crop percentile ranks before fusion. This avoids
mixing incompatible raw scales. The script evaluates the FULL validation set.
It does not train or modify any checkpoint.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ErrorAwareCrowdDataset
from head_metrics import head_report, print_head_report
from model_gated import ESRNet, cell_sum
from selection_baselines import DELTA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Stage-1 coarse checkpoint")
    p.add_argument("--error-head-checkpoint", required=True, help="Trained ErrorHead checkpoint")
    p.add_argument("--data-dir", default="error_aware/processed")
    p.add_argument("--dataset-root", default="dataset/VisDrone2019-DET-train")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--lambdas", default="0.25,0.50,0.75")
    p.add_argument("--budgets", default="3,6,13,26",
                   help="Number of selected cells per crop")
    p.add_argument("--output", default="error_aware/analysis/score_fusion_results.csv")
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

        coarse_density = coarse_out["density"].float()
        coarse_cells = cell_sum(coarse_density)[:, 0]
        gt_cells = cell_sum(gt_density)[:, 0]
        probabilities = logits.float().softmax(1)

        all_c.append(coarse_cells.cpu().numpy())
        all_g.append(gt_cells.cpu().numpy())
        all_p.append(probabilities.cpu().numpy())

    if not all_c:
        raise RuntimeError("Validation loader produced no batches.")

    return np.concatenate(all_c), np.concatenate(all_g), np.concatenate(all_p)


def percentile_rank_per_crop(scores):
    """Return [0,1] percentile ranks independently for each crop."""
    scores = np.asarray(scores, dtype=np.float64)
    n, h, w = scores.shape
    flat = scores.reshape(n, -1)
    ranks = np.empty_like(flat, dtype=np.float64)

    # Stable sorting makes ties deterministic.
    for i in range(n):
        order = np.argsort(flat[i], kind="mergesort")
        sorted_values = flat[i, order]
        start = 0
        while start < flat.shape[1]:
            end = start + 1
            while end < flat.shape[1] and sorted_values[end] == sorted_values[start]:
                end += 1
            # Average rank for ties; normalized to [0,1].
            rank_value = ((start + end - 1) / 2.0) / max(flat.shape[1] - 1, 1)
            ranks[i, order[start:end]] = rank_value
            start = end

    return ranks.reshape(n, h, w)


def topk_metrics(score, coarse, gt, k, delta):
    """Metrics averaged over crops, using exactly k selected cells per crop."""
    n, h, w = score.shape
    flat_score = score.reshape(n, -1)
    c = coarse.reshape(n, -1)
    g = gt.reshape(n, -1)
    err = c - g
    material = np.abs(err) >= delta
    under = err <= -delta
    over = err >= delta
    empty = g < 0.05

    selected = np.zeros_like(material, dtype=bool)
    indices = np.argpartition(flat_score, -k, axis=1)[:, -k:]
    rows = np.arange(n)[:, None]
    selected[rows, indices] = True

    selected_material = (selected & material).sum(axis=1)
    precision = (selected_material / k).mean()

    total_abs_error = np.abs(err).sum(axis=1)
    selected_abs_error = (np.abs(err) * selected).sum(axis=1)
    valid = total_abs_error > 0
    abs_error_share = (
        selected_abs_error[valid] / total_abs_error[valid]
    ).mean() if valid.any() else float("nan")

    empty_fp = (selected & empty).sum(axis=1) / k
    under_coverage = (
        (selected & under).sum(axis=1) / under.sum(axis=1).clip(min=1)
    )
    over_coverage = (
        (selected & over).sum(axis=1) / over.sum(axis=1).clip(min=1)
    )
    material_coverage = (
        (selected & material).sum(axis=1) / material.sum(axis=1).clip(min=1)
    )

    return {
        "precision": float(precision),
        "abs_error_share": float(abs_error_share),
        "empty_selected_share": float(empty_fp.mean()),
        "material_coverage": float(material_coverage.mean()),
        "under_coverage": float(under_coverage.mean()),
        "over_coverage": float(over_coverage.mean()),
    }


def main():
    args = parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")]
    budgets = [int(x) for x in args.budgets.split(",")]
    if any(not 0.0 <= x <= 1.0 for x in lambdas):
        raise ValueError("Every lambda must be between 0 and 1.")
    if any(x < 1 or x > 256 for x in budgets):
        raise ValueError("Budgets must be between 1 and 256 cells per crop.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device} | AMP: {amp} | material threshold: {DELTA}")

    model = ESRNet(pretrained=False, freeze_early=True).to(device)

    coarse_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.coarse.load_state_dict(
        coarse_ckpt.get("model_state_dict", coarse_ckpt), strict=True
    )

    head_ckpt = torch.load(
        args.error_head_checkpoint, map_location=device, weights_only=False
    )
    head_state = head_ckpt.get("error_head_state_dict", head_ckpt)
    model.error_head.load_state_dict(head_state, strict=True)

    for parameter in model.coarse.parameters():
        parameter.requires_grad_(False)
    model.coarse.eval()
    model.error_head.eval()

    validation_path = Path(args.data_dir) / "validation.jsonl"
    dataset = ErrorAwareCrowdDataset(
        validation_path,
        dataset_root=args.dataset_root,
        crop_size=512,
        density_stride=8,
        gaussian_sigma=3.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Validation crops: {len(dataset)}")
    coarse, gt, probs = collect(model, loader, device, amp)

    # Material error probability from the two directional classes.
    head_raw = probs[:, 1] + probs[:, 2]
    density_raw = np.abs(coarse)

    # Per-crop percentile ranks: 0 = lowest, 1 = highest.
    head_rank = percentile_rank_per_crop(head_raw)
    density_rank = percentile_rank_per_crop(density_raw)

    score_sets = {
        "head_only": head_rank,
        "density_only": density_rank,
    }
    for lam in lambdas:
        score_sets[f"fusion_lambda_{lam:.2f}"] = (
            lam * head_rank + (1.0 - lam) * density_rank
        )

    rows = []
    print("\nRANK-FUSION RESULTS")
    print("Scores are ranked independently within each crop.")
    print(
        f"{'strategy':<22} {'k':>3} {'precision':>11} "
        f"{'abs-error%':>11} {'empty%':>9} {'mat-cov%':>10} "
        f"{'under-cov%':>11} {'over-cov%':>10}"
    )
    print("-" * 94)

    for strategy, score in score_sets.items():
        for k in budgets:
            metrics = topk_metrics(score, coarse, gt, k, DELTA)
            row = {"strategy": strategy, "k": k, **metrics}
            rows.append(row)
            print(
                f"{strategy:<22} {k:>3d} "
                f"{metrics['precision'] * 100:>10.2f}% "
                f"{metrics['abs_error_share'] * 100:>10.2f}% "
                f"{metrics['empty_selected_share'] * 100:>8.2f}% "
                f"{metrics['material_coverage'] * 100:>9.2f}% "
                f"{metrics['under_coverage'] * 100:>10.2f}% "
                f"{metrics['over_coverage'] * 100:>9.2f}%"
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV: {out_path}")
    


if __name__ == "__main__":
    main()
