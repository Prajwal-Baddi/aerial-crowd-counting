"""Evaluate direction-aware ErrorHead selection against simple density rules.

Example (PowerShell):
python error_aware/evaluate_directional_selection.py `
  --checkpoint error_aware/checkpoints_stage1_holdout80/coarse_best.pth `
  --error-head-checkpoint error_aware/checkpoints_error_head/error_head_best.pth `
  --data-dir error_aware/processed `
  --dataset-root dataset/VisDrone2019-DET-train `
  --lambdas 0.25,0.50,0.75

This is an evaluation-only script. It does not train or modify checkpoints.
It reports macro and micro metrics using identical top-k masks for each metric.
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
    p.add_argument("--lambdas", default="0.25,0.50,0.75")
    p.add_argument("--budgets", default="3,6,13,26")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="error_aware/analysis/directional_selection_results.csv")
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


def percentile_rank_per_crop(scores):
    """Average-tie percentile ranks in [0,1], independently per crop."""
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
            ranks[i, order[start:end]] = ((start + end - 1) / 2.0) / max(len(vals) - 1, 1)
            start = end
    return ranks.reshape(scores.shape)


def make_topk_mask(score, k, tie_noise):
    flat = np.asarray(score, dtype=np.float64).reshape(score.shape[0], -1)
    if not 1 <= k <= flat.shape[1]:
        raise ValueError(f"k={k} outside [1,{flat.shape[1]}]")
    adjusted = flat + tie_noise
    idx = np.argpartition(adjusted, flat.shape[1] - k, axis=1)[:, -k:]
    mask = np.zeros(flat.shape, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask


def average_precision(scores, labels):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order].astype(np.float64)
    p_at_rank = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((p_at_rank * y).sum() / positives)


def target_metrics(mask, target, abs_err):
    """Precision and recall for one target, plus total absolute-error coverage."""
    selected_per_crop = mask.sum(axis=1)
    hits = (mask & target).sum(axis=1)
    positives_per_crop = target.sum(axis=1)

    macro_precision = float((hits / selected_per_crop).mean())
    micro_precision = float(hits.sum() / selected_per_crop.sum())

    valid = positives_per_crop > 0
    macro_recall = float((hits[valid] / positives_per_crop[valid]).mean()) if valid.any() else float("nan")
    micro_recall = float(hits.sum() / positives_per_crop.sum()) if positives_per_crop.sum() else float("nan")

    total_err_per_crop = abs_err.sum(axis=1)
    captured_err_per_crop = (mask * abs_err).sum(axis=1)
    valid_err = total_err_per_crop > 0
    macro_err = (
        float((captured_err_per_crop[valid_err] / total_err_per_crop[valid_err]).mean())
        if valid_err.any() else float("nan")
    )
    micro_err = (
        float(captured_err_per_crop.sum() / total_err_per_crop.sum())
        if total_err_per_crop.sum() else float("nan")
    )
    return {
        "precision_macro": macro_precision,
        "precision_micro": micro_precision,
        "recall_macro": macro_recall,
        "recall_micro": micro_recall,
        "abs_error_coverage_macro": macro_err,
        "abs_error_coverage_micro": micro_err,
    }


def main():
    args = parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    if any(not 0 <= x <= 1 for x in lambdas):
        raise ValueError("Lambdas must be in [0,1].")
    if any(not 1 <= k <= 256 for k in budgets):
        raise ValueError("Budgets must be in [1,256].")

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
        raise ValueError(f"Expected matching [N,H,W] maps; got {coarse.shape}, {gt.shape}")
    if probs.shape != (n, 3, *coarse.shape[1:]):
        raise ValueError(f"Unexpected probability shape {probs.shape}")

    c = coarse.reshape(n, -1)
    g = gt.reshape(n, -1)
    p_inc = probs[:, 1]
    p_dec = probs[:, 2]
    p_mat = probs[:, 1] + probs[:, 2]
    err = c - g
    abs_err = np.abs(err)
    under = err <= -DELTA
    over = err >= DELTA
    material = under | over

    # Directional density baselines follow the original selection_baselines logic:
    # low predicted count for under-count candidates, high count for over-count.
    raw_scores = {
        "under": {
            "errorhead": p_inc,
            "density_baseline": -coarse,
        },
        "over": {
            "errorhead": p_dec,
            "density_baseline": coarse,
        },
        "material": {
            "errorhead": p_mat,
            "density_baseline": np.abs(coarse),
        },
    }

    score_sets = {}
    for target_name, strategies in raw_scores.items():
        score_sets[target_name] = {}
        ranked = {name: percentile_rank_per_crop(score) for name, score in strategies.items()}
        score_sets[target_name].update(ranked)
        for lam in lambdas:
            score_sets[target_name][f"fusion_{lam:.2f}"] = (
                lam * ranked["errorhead"] + (1.0 - lam) * ranked["density_baseline"]
            )

    rng = np.random.default_rng(args.seed)
    tie_noise = rng.random((n, c.shape[1])) * 1e-12

    targets = {"under": under, "over": over, "material": material}
    rows = []
    print("\nDIRECTION-AWARE TOP-K RESULTS")
    print("P = precision; R = recall; Abs = fraction of total |coarse-GT| captured.")
    print("Each strategy is evaluated against its corresponding target.")
    header = (
        f"{'target':<9}{'strategy':<20}{'k':>4}"
        f"{'P macro':>10}{'P micro':>10}{'R macro':>10}{'R micro':>10}"
        f"{'Abs macro':>12}{'Abs micro':>12}"
    )
    print(header)
    print("-" * len(header))

    for target_name, strategies in score_sets.items():
        for strategy, score in strategies.items():
            for k in budgets:
                mask = make_topk_mask(score, k, tie_noise)
                metrics = target_metrics(mask, targets[target_name], abs_err)
                row = {
                    "target": target_name,
                    "strategy": strategy,
                    "k": k,
                    **metrics,
                }
                rows.append(row)
                print(
                    f"{target_name:<9}{strategy:<20}{k:>4d}"
                    f"{metrics['precision_macro'] * 100:>9.2f}%"
                    f"{metrics['precision_micro'] * 100:>9.2f}%"
                    f"{metrics['recall_macro'] * 100:>9.2f}%"
                    f"{metrics['recall_micro'] * 100:>9.2f}%"
                    f"{metrics['abs_error_coverage_macro'] * 100:>11.2f}%"
                    f"{metrics['abs_error_coverage_micro'] * 100:>11.2f}%"
                )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results: {out}")

    print("\nGLOBAL AVERAGE PRECISION")
    print(f"{'target':<12}{'ErrorHead':>12}{'density':>12}")
    for name, target in targets.items():
        scores = raw_scores[name]
        ap_head = average_precision(scores["errorhead"], target)
        ap_density = average_precision(scores["density_baseline"], target)
        print(f"{name:<12}{ap_head:>12.4f}{ap_density:>12.4f}")

    print("\nNotes:")
    print("- Under target: coarse-GT <= -DELTA; density baseline ranks -coarse.")
    print("- Over target: coarse-GT >= DELTA; density baseline ranks +coarse.")
    print("- Material target: either direction; density baseline ranks |coarse|.")
    print("- Fusion lambda weights ErrorHead rank; (1-lambda) weights density rank.")
    print("- Absolute-error coverage is included as a secondary utility metric.")
    print("- Macro recall averages crops containing the target; micro recall pools target cells.")


if __name__ == "__main__":
    main()
