"""Evaluate hybrid direction-aware selection with a fixed per-crop budget.

Under-count candidates are ranked by ErrorHead P(increase).
Over-count candidates are ranked by coarse density.
The total number of unique selected cells is exactly k per crop. If the two
directional selections overlap, remaining slots are filled by the strongest
remaining directional percentile scores.

Example (PowerShell):
python error_aware/evaluate_hybrid_selection.py `
  --checkpoint error_aware/checkpoints_stage1_holdout80/coarse_best.pth `
  --error-head-checkpoint error_aware/checkpoints_error_head/error_head_best.pth `
  --data-dir error_aware/processed `
  --dataset-root dataset/VisDrone2019-DET-train `
  --allocations 0.25,0.50,0.75
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
    p.add_argument("--allocations", default="0.25,0.50,0.75",
                   help="Fraction of total budget allocated to under-count selection")
    p.add_argument("--budgets", default="3,6,13,26")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="error_aware/analysis/hybrid_selection_results.csv")
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
        all_c.append(cell_sum(coarse_out["density"].float())[:, 0].cpu().numpy())
        all_g.append(cell_sum(gt_density)[:, 0].cpu().numpy())
        all_p.append(logits.float().softmax(1).cpu().numpy())
    if not all_c:
        raise RuntimeError("Validation loader produced no batches.")
    return np.concatenate(all_c), np.concatenate(all_g), np.concatenate(all_p)


def percentile_rank_per_crop(scores):
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    flat = scores.reshape(n, -1)
    ranks = np.empty_like(flat)
    for i in range(n):
        order = np.argsort(flat[i], kind="mergesort")
        vals = flat[i, order]
        start = 0
        while start < len(vals):
            end = start + 1
            while end < len(vals) and vals[end] == vals[start]:
                end += 1
            ranks[i, order[start:end]] = (
                (start + end - 1) / 2.0
            ) / max(len(vals) - 1, 1)
            start = end
    return ranks.reshape(scores.shape)


def topk_indices(score, k, tie_noise):
    flat = np.asarray(score, dtype=np.float64).reshape(score.shape[0], -1)
    if k <= 0:
        return np.empty((flat.shape[0], 0), dtype=np.int64)
    adjusted = flat + tie_noise
    return np.argpartition(adjusted, flat.shape[1] - k, axis=1)[:, -k:]


def hybrid_mask(under_score, over_score, k, under_fraction, tie_noise):
    """Build exactly-k unique selections from directional sub-budgets."""
    n, cells = under_score.reshape(under_score.shape[0], -1).shape
    ku = int(round(k * under_fraction))
    ku = min(max(ku, 0), k)
    ko = k - ku

    u = under_score.reshape(n, cells)
    o = over_score.reshape(n, cells)
    ui = topk_indices(u, ku, tie_noise)
    oi = topk_indices(o, ko, tie_noise)

    mask = np.zeros((n, cells), dtype=bool)
    rows = np.arange(n)[:, None]
    if ku:
        mask[rows, ui] = True
    if ko:
        mask[rows, oi] = True

    # Directional selections can overlap. Fill any vacant slots using the
    # strongest remaining score from either direction, preserving exact k.
    combined = np.maximum(u, o)
    adjusted = combined + tie_noise
    for i in range(n):
        missing = k - int(mask[i].sum())
        if missing:
            candidates = np.flatnonzero(~mask[i])
            order = candidates[np.argpartition(adjusted[i, candidates], -missing)[-missing:]]
            mask[i, order] = True

    if not np.all(mask.sum(axis=1) == k):
        raise RuntimeError("Hybrid selector failed to select exactly k cells.")
    return mask, ku, ko


def compute_metrics(mask, coarse, gt, delta):
    c = coarse.reshape(coarse.shape[0], -1)
    g = gt.reshape(gt.shape[0], -1)
    err = c - g
    abs_err = np.abs(err)
    under = err <= -delta
    over = err >= delta
    material = under | over
    empty = g < 0.05

    selected_per_crop = mask.sum(axis=1)
    hits = (mask & material).sum(axis=1)
    precision_per_crop = hits / selected_per_crop
    precision_macro = float(precision_per_crop.mean())
    precision_micro = float(hits.sum() / selected_per_crop.sum())

    def recall(target):
        selected = (mask & target).sum(axis=1)
        total = target.sum(axis=1)
        valid = total > 0
        macro = float((selected[valid] / total[valid]).mean()) if valid.any() else float("nan")
        micro = float(selected.sum() / total.sum()) if total.sum() else float("nan")
        return macro, micro

    def error_coverage():
        selected_error = (mask * abs_err).sum(axis=1)
        total_error = abs_err.sum(axis=1)
        valid = total_error > 0
        macro = float((selected_error[valid] / total_error[valid]).mean()) if valid.any() else float("nan")
        micro = float(selected_error.sum() / total_error.sum()) if total_error.sum() else float("nan")
        return macro, micro

    abs_macro, abs_micro = error_coverage()
    under_macro, under_micro = recall(under)
    over_macro, over_micro = recall(over)
    mat_macro, mat_micro = recall(material)
    empty_share = float((mask & empty).sum() / mask.sum())

    return {
        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
        "abs_error_coverage_macro": abs_macro,
        "abs_error_coverage_micro": abs_micro,
        "material_recall_macro": mat_macro,
        "material_recall_micro": mat_micro,
        "under_recall_macro": under_macro,
        "under_recall_micro": under_micro,
        "over_recall_macro": over_macro,
        "over_recall_micro": over_micro,
        "empty_selected_share": empty_share,
    }


def main():
    args = parse_args()
    allocations = [float(x) for x in args.allocations.split(",") if x.strip()]
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    if any(not 0.0 <= x <= 1.0 for x in allocations):
        raise ValueError("Allocations must be fractions between 0 and 1.")
    if any(not 1 <= k <= 256 for k in budgets):
        raise ValueError("Budgets must be between 1 and 256.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device} | AMP: {amp} | DELTA: {DELTA}")

    model = ESRNet(pretrained=False, freeze_early=True).to(device)
    coarse_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.coarse.load_state_dict(coarse_ckpt.get("model_state_dict", coarse_ckpt), strict=True)
    head_ckpt = torch.load(args.error_head_checkpoint, map_location=device, weights_only=False)
    model.error_head.load_state_dict(
        head_ckpt.get("error_head_state_dict", head_ckpt), strict=True
    )
    model.eval()

    dataset = ErrorAwareCrowdDataset(
        Path(args.data_dir) / "validation.jsonl",
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
    if coarse.shape != gt.shape or coarse.ndim != 3:
        raise ValueError(f"Expected matching [N,H,W] maps, got {coarse.shape}, {gt.shape}")
    if probs.shape != (len(coarse), 3, *coarse.shape[1:]):
        raise ValueError(f"Unexpected probability shape: {probs.shape}")

    # Directional ranking signals. Both are ranked per crop before allocation.
    under_rank = percentile_rank_per_crop(probs[:, 1])
    over_rank = percentile_rank_per_crop(coarse)
    density_rank = percentile_rank_per_crop(np.abs(coarse))
    head_material_rank = percentile_rank_per_crop(probs[:, 1] + probs[:, 2])

    n = len(coarse)
    num_cells = coarse.shape[1] * coarse.shape[2]
    rng = np.random.default_rng(args.seed)
    tie_noise = rng.random((n, num_cells)) * 1e-12

    c = coarse.reshape(n, -1)
    g = gt.reshape(n, -1)
    # Baselines use the same exact-k mask implementation.
    strategies = {
        "density_material": density_rank,
        "head_material": head_material_rank,
    }

    rows = []
    print("\nHYBRID DIRECTIONAL SELECTION")
    print("Under score = ErrorHead P(increase); over score = coarse density.")
    print("Allocation is the fraction of k reserved for under-count candidates.")
    header = (
        f"{'strategy':<24}{'k':>4}{'U budget':>10}{'O budget':>10}"
        f"{'P macro':>10}{'Abs macro':>11}{'Abs micro':>11}"
        f"{'Mat Rµ':>9}{'Under Rµ':>10}{'Over Rµ':>9}"
    )
    print(header)
    print("-" * len(header))

    for k in budgets:
        # Standard exact-k baselines.
        for name, score in strategies.items():
            idx = topk_indices(score, k, tie_noise)
            mask = np.zeros((n, num_cells), dtype=bool)
            np.put_along_axis(mask, idx, True, axis=1)
            metrics = compute_metrics(mask, coarse, gt, DELTA)
            row = {"strategy": name, "k": k, "under_budget": "", "over_budget": "", **metrics}
            rows.append(row)
            print(
                f"{name:<24}{k:>4d}{'-':>10}{'-':>10}"
                f"{metrics['precision_macro']*100:>9.2f}%"
                f"{metrics['abs_error_coverage_macro']*100:>10.2f}%"
                f"{metrics['abs_error_coverage_micro']*100:>10.2f}%"
                f"{metrics['material_recall_micro']*100:>8.2f}%"
                f"{metrics['under_recall_micro']*100:>9.2f}%"
                f"{metrics['over_recall_micro']*100:>8.2f}%"
            )

        for fraction in allocations:
            mask, ku, ko = hybrid_mask(under_rank, over_rank, k, fraction, tie_noise)
            metrics = compute_metrics(mask, coarse, gt, DELTA)
            name = f"hybrid_under_{fraction:.2f}"
            row = {
                "strategy": name,
                "k": k,
                "under_budget": ku,
                "over_budget": ko,
                **metrics,
            }
            rows.append(row)
            print(
                f"{name:<24}{k:>4d}{ku:>10d}{ko:>10d}"
                f"{metrics['precision_macro']*100:>9.2f}%"
                f"{metrics['abs_error_coverage_macro']*100:>10.2f}%"
                f"{metrics['abs_error_coverage_micro']*100:>10.2f}%"
                f"{metrics['material_recall_micro']*100:>8.2f}%"
                f"{metrics['under_recall_micro']*100:>9.2f}%"
                f"{metrics['over_recall_micro']*100:>8.2f}%"
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results: {out}")
    print("\nInterpretation:")
    print("- Compare hybrid rows with density_material and head_material at the same k.")
    print("- U budget is the requested under-count allocation; O budget is the over-count allocation.")
    print("- Overlapping directional selections are deduplicated; remaining slots use max directional rank.")
    print("- Exact total selection budget is preserved for every crop.")
    print("- Macro metrics average crop-level ratios; micro metrics pool cells globally.")
    print("- This evaluates selection only; it does not apply refinement or measure final MAE/RMSE.")


if __name__ == "__main__":
    main()
