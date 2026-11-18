"""Evaluate ErrorHead/density score fusion with consistent top-k masks.

Example (PowerShell):
python error_aware/evaluate_score_fusion.py `
  --checkpoint error_aware/checkpoints_stage1_holdout80/coarse_best.pth `
  --error-head-checkpoint error_aware/checkpoints_error_head/error_head_best.pth `
  --data-dir error_aware/processed `
  --dataset-root dataset/VisDrone2019-DET-train `
  --lambdas 0.25,0.50,0.75

The script evaluates the full validation manifest. It does not train or modify
any checkpoint. Metrics are reported with both macro (per-crop average) and
micro (pooled over cells) aggregation.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ErrorAwareCrowdDataset
from model_gated import ESRNet, cell_sum
from selection_baselines import DELTA, types


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
    p.add_argument("--budgets", default="3,6,13,26")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="error_aware/analysis/score_fusion_results_v2.csv")
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
    """Average-tie percentile ranks in [0,1], independently per crop."""
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    flat = scores.reshape(n, -1)
    ranks = np.empty_like(flat, dtype=np.float64)

    for i in range(n):
        order = np.argsort(flat[i], kind="mergesort")
        sorted_values = flat[i, order]
        start = 0
        while start < flat.shape[1]:
            end = start + 1
            while end < flat.shape[1] and sorted_values[end] == sorted_values[start]:
                end += 1
            rank_value = ((start + end - 1) / 2.0) / max(flat.shape[1] - 1, 1)
            ranks[i, order[start:end]] = rank_value
            start = end
    return ranks.reshape(scores.shape)


def make_topk_mask(score, k, tie_noise):
    """Select exactly k cells per crop, with shared reproducible tie-breaking."""
    flat = np.asarray(score, dtype=np.float64).reshape(score.shape[0], -1)
    if not 1 <= k <= flat.shape[1]:
        raise ValueError(f"k={k} is outside [1, {flat.shape[1]}].")
    # The same tiny per-cell noise is reused for every strategy, so ties are
    # resolved consistently rather than depending on call order.
    adjusted = flat + tie_noise
    indices = np.argpartition(adjusted, flat.shape[1] - k, axis=1)[:, -k:]
    mask = np.zeros(flat.shape, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=1)
    return mask


def average_precision(scores, labels):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order].astype(np.float64)
    precision_at_rank = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((precision_at_rank * y).sum() / positives)


def print_ap_report(coarse, gt, probs, delta):
    c = coarse.reshape(-1)
    g = gt.reshape(-1)
    p = probs.transpose(0, 2, 3, 1).reshape(-1, 3)
    err = c - g
    inc = err <= -delta
    dec = err >= delta
    material = inc | dec

    results = {
        "material": (
            average_precision(p[:, 1] + p[:, 2], material),
            average_precision(np.abs(c), material),
        ),
        "increase / under-count": (
            average_precision(p[:, 1], inc),
            average_precision(-c, inc),
        ),
        "decrease / over-count": (
            average_precision(p[:, 2], dec),
            average_precision(c, dec),
        ),
    }

    print("\nAverage precision (global cell ranking; deployable baselines):")
    print(f"{'Target':<25}{'ErrorHead':>12}{'Simple baseline':>18}")
    for name, (head_ap, baseline_ap) in results.items():
        print(f"{name:<25}{head_ap:>12.4f}{baseline_ap:>18.4f}")
    print("Material baseline uses |coarse count|; no ground-truth error is used.")


def evaluate_mask(mask, coarse, gt):
    """Compute macro and micro metrics from one identical [N,cells] mask."""
    c = coarse.reshape(coarse.shape[0], -1)
    g = gt.reshape(gt.shape[0], -1)
    err = c - g
    abs_err = np.abs(err)
    material = abs_err >= DELTA
    error_types = types(c, g)

    selected_count = mask.sum(axis=1)
    if not np.all(selected_count == selected_count[0]):
        raise RuntimeError("Expected a fixed selection budget in every crop.")

    selected_material = (mask & material).sum(axis=1)
    selected_abs_error = (mask * abs_err).sum(axis=1)
    total_abs_error = abs_err.sum(axis=1)

    # Per-crop ratios. Crops with no positives/errors are excluded from that
    # recall/coverage macro average; their zero denominator is not informative.
    material_total = material.sum(axis=1)
    macro_precision = float((selected_material / selected_count).mean())
    micro_precision = float(selected_material.sum() / selected_count.sum())

    valid_error = total_abs_error > 0
    macro_abs_error = (
        float((selected_abs_error[valid_error] / total_abs_error[valid_error]).mean())
        if valid_error.any() else float("nan")
    )
    micro_abs_error = (
        float(selected_abs_error.sum() / total_abs_error.sum())
        if total_abs_error.sum() > 0 else float("nan")
    )

    def recall_pair(target):
        selected = (mask & target).sum(axis=1)
        totals = target.sum(axis=1)
        valid = totals > 0
        macro = float((selected[valid] / totals[valid]).mean()) if valid.any() else float("nan")
        micro = float(selected.sum() / totals.sum()) if totals.sum() else float("nan")
        return macro, micro

    mat_macro, mat_micro = recall_pair(material)
    under_macro, under_micro = recall_pair(error_types["miss"] | error_types["under"])
    over_macro, over_micro = recall_pair(error_types["over"])
    fp_macro, fp_micro = recall_pair(error_types["FP_empty"])

    empty = g < 0.05
    empty_selected = (mask & empty).sum(axis=1)
    macro_empty_share = float((empty_selected / selected_count).mean())
    micro_empty_share = float(empty_selected.sum() / selected_count.sum())

    return {
        "precision_macro": macro_precision,
        "precision_micro": micro_precision,
        "abs_error_coverage_macro": macro_abs_error,
        "abs_error_coverage_micro": micro_abs_error,
        "material_recall_macro": mat_macro,
        "material_recall_micro": mat_micro,
        "under_recall_macro": under_macro,
        "under_recall_micro": under_micro,
        "over_recall_macro": over_macro,
        "over_recall_micro": over_micro,
        "FP_empty_recall_macro": fp_macro,
        "FP_empty_recall_micro": fp_micro,
        "empty_selected_share_macro": macro_empty_share,
        "empty_selected_share_micro": micro_empty_share,
    }


def main():
    args = parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
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
    model.eval()

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
    n = coarse.shape[0]
    if coarse.shape != gt.shape or coarse.ndim != 3:
        raise ValueError(f"Expected matching [N,H,W] cell maps; got {coarse.shape}, {gt.shape}")
    if probs.shape != (n, 3, *coarse.shape[1:]):
        raise ValueError(f"Unexpected ErrorHead probability shape: {probs.shape}")

    # Use the exact same semantic scores as the original baselines:
    # density = coarse count (Softplus output, hence nonnegative).
    head_raw = probs[:, 1] + probs[:, 2]
    density_raw = coarse
    head_rank = percentile_rank_per_crop(head_raw)
    density_rank = percentile_rank_per_crop(density_raw)

    score_sets = {
        "random": np.random.default_rng(args.seed).random(coarse.shape),
        "head_only": head_rank,
        "density_only": density_rank,
    }
    for lam in lambdas:
        score_sets[f"fusion_lambda_{lam:.2f}"] = (
            lam * head_rank + (1.0 - lam) * density_rank
        )
    # Oracle is an explicitly non-deployable upper-bound reference.
    score_sets["oracle_non_deployable"] = np.abs(coarse - gt)

    # One shared tie-noise array across all strategies/budgets ensures
    # deterministic and strategy-independent tie resolution.
    rng = np.random.default_rng(args.seed)
    tie_noise = rng.random((n, coarse.shape[1] * coarse.shape[2])) * 1e-12

    rows = []
    print("\nCONSISTENT TOP-K RESULTS")
    print("Macro = average over crops; micro = pooled counts over all cells.")
    print("The same per-crop tie-breaking noise is reused for every strategy.")
    header = (
        f"{'strategy':<25}{'k':>4}{'P macro':>10}{'P micro':>10}"
        f"{'Abs macro':>12}{'Abs micro':>12}{'Mat macro':>12}{'Mat micro':>12}"
    )
    print(header)
    print("-" * len(header))

    for strategy, score in score_sets.items():
        for k in budgets:
            mask = make_topk_mask(score, k, tie_noise)
            metrics = evaluate_mask(mask, coarse, gt)
            row = {"strategy": strategy, "k": k, **metrics}
            rows.append(row)
            print(
                f"{strategy:<25}{k:>4d}"
                f"{metrics['precision_macro'] * 100:>9.2f}%"
                f"{metrics['precision_micro'] * 100:>9.2f}%"
                f"{metrics['abs_error_coverage_macro'] * 100:>11.2f}%"
                f"{metrics['abs_error_coverage_micro'] * 100:>11.2f}%"
                f"{metrics['material_recall_macro'] * 100:>11.2f}%"
                f"{metrics['material_recall_micro'] * 100:>11.2f}%"
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved complete macro/micro results: {out_path}")
    print_ap_report(coarse, gt, probs, DELTA)
    print("\nInterpretation notes:")
    print("- AP is global ranking across validation cells; top-k metrics are per-crop.")
    print("- Oracle selection uses ground truth and is not deployable.")
    print("- Macro recall excludes crops with zero positives for that error category.")
    print("- Micro recall pools positives across all validation cells.")


if __name__ == "__main__":
    main()
