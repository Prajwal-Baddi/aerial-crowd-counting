import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import get_loader
from model_multiplicative import ESRNet, cell_sum, select_topk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect Stage-2 refinement correction behavior."
    )
    parser.add_argument("--data-dir", default="error_aware/processed")
    parser.add_argument(
        "--dataset-root",
        default="dataset/VisDrone2019-DET-train",
    )
    parser.add_argument(
        "--coarse-checkpoint",
        default="error_aware/checkpoints/coarse_best.pth",
    )
    parser.add_argument("--stage2-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--empty-threshold", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="error_aware/refinement_behavior_diagnostic.json",
    )
    return parser.parse_args()


def load_model(args, device):
    model = ESRNet(
        pretrained=False,
        freeze_early=True,
        refine_width=96,
    ).to(device)

    coarse_ckpt = torch.load(
        args.coarse_checkpoint,
        map_location=device,
        weights_only=False,
    )
    coarse_state = coarse_ckpt.get(
        "model_state_dict",
        coarse_ckpt.get("model", coarse_ckpt),
    )
    model.coarse.load_state_dict(coarse_state, strict=True)

    for parameter in model.coarse.parameters():
        parameter.requires_grad_(False)

    print(f"Loaded coarse checkpoint: {args.coarse_checkpoint}")

    if args.stage2_checkpoint:
        stage2_ckpt = torch.load(
            args.stage2_checkpoint,
            map_location=device,
            weights_only=False,
        )
        state = stage2_ckpt.get("model_state_dict", stage2_ckpt)
        model.load_state_dict(state, strict=True)
        print(f"Loaded Stage-2 checkpoint: {args.stage2_checkpoint}")

        if isinstance(stage2_ckpt, dict) and "epoch" in stage2_ckpt:
            print(f"Stage-2 epoch: {stage2_ckpt['epoch']}")
    else:
        print(
            "WARNING: Using freshly initialized Stage-2 components. "
            "No trained Stage-2 checkpoint was supplied."
        )

    model.eval()
    return model


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def error_summary(before, after):
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)

    if before.size == 0:
        return {
            "cells": 0,
            "before_mae": None,
            "after_mae": None,
            "mean_error_change": None,
            "improved_pct": None,
            "worsened_pct": None,
            "unchanged_pct": None,
        }

    if before.shape != after.shape:
        raise ValueError(
            f"Error-summary shape mismatch: before={before.shape}, "
            f"after={after.shape}"
        )

    change = after - before
    tolerance = 1e-8

    return {
        "cells": int(before.size),
        "before_mae": float(before.mean()),
        "after_mae": float(after.mean()),
        "mean_error_change": float(change.mean()),
        "improved_pct": float(np.mean(change < -tolerance) * 100.0),
        "worsened_pct": float(np.mean(change > tolerance) * 100.0),
        "unchanged_pct": float(
            np.mean(np.abs(change) <= tolerance) * 100.0
        ),
    }


def collect_branch_stats(storage, gain, add, coarse_cells, gt_cells):
    """
    The selected-cell tensors contain 4x4 density patches, not scalar counts.

    Shapes normally are:
      gain:         [N, 1, 4, 4]
      add:           [N, 1, 4, 4]
      coarse_cells:  [N, 1, 4, 4]
      gt_cells:      [N, 1] or [N]

    Aggregate each 4x4 patch to one count before comparing it with gt_cells.
    """

    gain_map = gain.detach().float()
    add_map = add.detach().float()
    coarse_map = coarse_cells.detach().float()

    if gain_map.shape != add_map.shape:
        raise RuntimeError(
            "Gain/add tensor shape mismatch: "
            f"gain={tuple(gain_map.shape)}, add={tuple(add_map.shape)}"
        )
    if gain_map.shape != coarse_map.shape:
        raise RuntimeError(
            "Gain/coarse tensor shape mismatch: "
            f"gain={tuple(gain_map.shape)}, "
            f"coarse_cells={tuple(coarse_map.shape)}"
        )
    if gain_map.ndim < 3:
        raise RuntimeError(
            "Expected per-cell density patches with at least 3 dimensions; "
            f"received shape {tuple(gain_map.shape)}"
        )

    # Sum the spatial density pixels inside each selected 4x4 cell.
    coarse_count = (
        coarse_map.sum(dim=(-2, -1)).reshape(-1).cpu().numpy()
    )
    gain_count = (
        (coarse_map * gain_map).sum(dim=(-2, -1)).reshape(-1).cpu().numpy()
    )
    add_mass = (
        add_map.sum(dim=(-2, -1)).reshape(-1).cpu().numpy()
    )
    gt_count = gt_cells.detach().float().reshape(-1).cpu().numpy()

    sizes = {
        "gain_cells": int(gain_count.size),
        "add_cells": int(add_mass.size),
        "coarse_cells": int(coarse_count.size),
        "gt_cells": int(gt_count.size),
    }
    if len(set(sizes.values())) != 1:
        raise RuntimeError(
            "Selected-cell aggregation mismatch after summing 4x4 "
            f"patches: {sizes}"
        )

    # A count-weighted effective gain is meaningful only when coarse_count > 0.
    valid_gain = coarse_count > 1e-8
    effective_gain = np.full(coarse_count.shape, np.nan, dtype=np.float64)
    effective_gain[valid_gain] = (
        gain_count[valid_gain] / coarse_count[valid_gain]
    )

    # Compare each branch as a scalar count per selected cell.
    refined_gain_count = gain_count
    refined_add_count = coarse_count + add_mass
    refined_combined_count = gain_count + add_mass

    before_error = np.abs(coarse_count - gt_count)
    gain_error = np.abs(refined_gain_count - gt_count)
    add_error = np.abs(refined_add_count - gt_count)
    combined_error = np.abs(refined_combined_count - gt_count)

    empty = gt_count < storage["empty_threshold"]
    nonempty = ~empty

    storage["gain"].extend(
        effective_gain[np.isfinite(effective_gain)].tolist()
    )
    storage["add"].extend(add_mass.tolist())

    storage["gain_below_one"] += int(
        np.sum(effective_gain[valid_gain] < 1.0)
    )
    storage["gain_above_one"] += int(
        np.sum(effective_gain[valid_gain] > 1.0)
    )
    storage["gain_equal_one"] += int(
        np.sum(np.isclose(effective_gain[valid_gain], 1.0))
    )
    storage["gain_valid_cells"] += int(np.sum(valid_gain))

    storage["add_positive"] += int(np.sum(add_mass > 1e-8))
    storage["add_zero"] += int(np.sum(add_mass <= 1e-8))
    storage["selected_cells"] += int(gt_count.size)

    storage["empty_before"].extend(before_error[empty].tolist())
    storage["empty_gain"].extend(gain_error[empty].tolist())
    storage["empty_add"].extend(add_error[empty].tolist())
    storage["empty_combined"].extend(combined_error[empty].tolist())

    storage["nonempty_before"].extend(before_error[nonempty].tolist())
    storage["nonempty_gain"].extend(gain_error[nonempty].tolist())
    storage["nonempty_add"].extend(add_error[nonempty].tolist())
    storage["nonempty_combined"].extend(
        combined_error[nonempty].tolist()
    )

    storage["all_before"].extend(before_error.tolist())
    storage["all_gain"].extend(gain_error.tolist())
    storage["all_add"].extend(add_error.tolist())
    storage["all_combined"].extend(combined_error.tolist())

    storage["gain_correction"].extend(
        (refined_gain_count - coarse_count).tolist()
    )
    storage["add_correction"].extend(add_mass.tolist())
    storage["combined_correction"].extend(
        (refined_combined_count - coarse_count).tolist()
    )


@torch.no_grad()
def run(model, loader, args, device):
    storage = {
        "empty_threshold": args.empty_threshold,
        # Effective per-cell gain (only cells with nonzero coarse count).
        "gain": [],
        # Total additive mass per selected cell.
        "add": [],
        "gain_below_one": 0,
        "gain_above_one": 0,
        "gain_equal_one": 0,
        "gain_valid_cells": 0,
        "add_positive": 0,
        "add_zero": 0,
        "selected_cells": 0,
        "empty_before": [],
        "empty_gain": [],
        "empty_add": [],
        "empty_combined": [],
        "nonempty_before": [],
        "nonempty_gain": [],
        "nonempty_add": [],
        "nonempty_combined": [],
        "all_before": [],
        "all_gain": [],
        "all_add": [],
        "all_combined": [],
        "gain_correction": [],
        "add_correction": [],
        "combined_correction": [],
    }

    samples = 0
    batches = 0

    for step, (images, gt_density, counts) in enumerate(loader, 1):
        if args.max_batches is not None and step > args.max_batches:
            break

        images = images.to(device, non_blocking=True)
        gt_density = gt_density.to(
            device,
            non_blocking=True,
        ).float()

        coarse_output = model.coarse(images)
        coarse = coarse_output["density"].float()
        features = coarse_output["stride4_features"]

        # Oracle selection: cells with the largest absolute
        # coarse-versus-ground-truth count error.
        coarse_cells_map = cell_sum(coarse)
        gt_cells_map = cell_sum(gt_density)

        oracle_error = (coarse_cells_map - gt_cells_map).abs()
        mask = select_topk(oracle_error, args.budget)

        # Apply refinement once and inspect its auxiliary outputs.
        _, aux = model.apply_refine(
            coarse,
            features,
            mask,
            return_aux=True,
        )

        if aux is not None:
            selected = aux["selected"]
            if selected.numel() > 0:
                batch_idx, cell_y, cell_x = selected.unbind(dim=1)

                # One ground-truth scalar count per selected cell.
                gt_selected = gt_cells_map[
                    batch_idx, 0, cell_y, cell_x
                ].reshape(-1)

                collect_branch_stats(
                    storage,
                    aux["gain"],
                    aux["add"],
                    aux["coarse_cells"],
                    gt_selected,
                )

        samples += int(images.shape[0])
        batches += 1

        if step == 1 or step % 25 == 0:
            print(
                f"Processed {step} batches | "
                f"selected cells so far: {storage['selected_cells']}"
            )

    selected_count = storage["selected_cells"]
    gain_denominator = storage["gain_valid_cells"]

    result = {
        "samples": samples,
        "batches": batches,
        "budget": args.budget,
        "selection": "oracle",
        "empty_cell_threshold": args.empty_threshold,
        "selected_cells": selected_count,
        "gain_distribution": distribution(storage["gain"]),
        "add_distribution": distribution(storage["add"]),
        "gain_distribution_definition": (
            "Count-weighted effective gain per selected cell; "
            "cells with zero coarse count are excluded."
        ),
        "add_distribution_definition": (
            "Sum of the 4x4 additive density patch per selected cell."
        ),
        "gain_correction_distribution": distribution(
            storage["gain_correction"]
        ),
        "add_correction_distribution": distribution(
            storage["add_correction"]
        ),
        "combined_correction_distribution": distribution(
            storage["combined_correction"]
        ),
        "gain_percentages": {
            "denominator_cells_with_nonzero_coarse_count": gain_denominator,
            "below_one": (
                storage["gain_below_one"] / gain_denominator * 100.0
                if gain_denominator else None
            ),
            "above_one": (
                storage["gain_above_one"] / gain_denominator * 100.0
                if gain_denominator else None
            ),
            "equal_one": (
                storage["gain_equal_one"] / gain_denominator * 100.0
                if gain_denominator else None
            ),
        },
        "add_percentages": {
            "denominator_selected_cells": selected_count,
            "positive": (
                storage["add_positive"] / selected_count * 100.0
                if selected_count else None
            ),
            "zero": (
                storage["add_zero"] / selected_count * 100.0
                if selected_count else None
            ),
        },
        "cell_error": {
            "all": {
                "gain_only": error_summary(
                    storage["all_before"], storage["all_gain"]
                ),
                "additive_only": error_summary(
                    storage["all_before"], storage["all_add"]
                ),
                "combined": error_summary(
                    storage["all_before"], storage["all_combined"]
                ),
            },
            "empty": {
                "gain_only": error_summary(
                    storage["empty_before"], storage["empty_gain"]
                ),
                "additive_only": error_summary(
                    storage["empty_before"], storage["empty_add"]
                ),
                "combined": error_summary(
                    storage["empty_before"], storage["empty_combined"]
                ),
            },
            "nonempty": {
                "gain_only": error_summary(
                    storage["nonempty_before"], storage["nonempty_gain"]
                ),
                "additive_only": error_summary(
                    storage["nonempty_before"], storage["nonempty_add"]
                ),
                "combined": error_summary(
                    storage["nonempty_before"],
                    storage["nonempty_combined"],
                ),
            },
        },
    }

    return result


def print_results(result):
    print("\n" + "=" * 78)
    print("REFINEMENT CORRECTION BEHAVIOR")
    print("=" * 78)

    print(f"Samples: {result['samples']}")
    print(f"Batches: {result['batches']}")
    print(f"Selected cells: {result['selected_cells']}")

    print("\nEffective per-cell gain distribution:")
    print(json.dumps(result["gain_distribution"], indent=2))

    print("\nAdditive mass per selected cell distribution:")
    print(json.dumps(result["add_distribution"], indent=2))

    print("\nGain percentages:")
    print(json.dumps(result["gain_percentages"], indent=2))

    print("\nAdditive percentages:")
    print(json.dumps(result["add_percentages"], indent=2))

    def fmt(value):
        return "n/a" if value is None else f"{value:.5f}"

    for category in ("all", "empty", "nonempty"):
        print(f"\n{category.upper()} SELECTED CELLS")
        print(
            f"{'Branch':<18}"
            f"{'Before MAE':>13}"
            f"{'After MAE':>13}"
            f"{'Improved %':>13}"
            f"{'Worsened %':>13}"
        )
        print("-" * 70)

        for branch in ("gain_only", "additive_only", "combined"):
            values = result["cell_error"][category][branch]
            print(
                f"{branch:<18}"
                f"{fmt(values['before_mae']):>13}"
                f"{fmt(values['after_mae']):>13}"
                f"{fmt(values['improved_pct']):>13}"
                f"{fmt(values['worsened_pct']):>13}"
            )

    print("=" * 78)


def main():
    args = parse_args()

    if not 0.0 < args.budget <= 1.0:
        raise ValueError("--budget must be greater than 0 and at most 1.")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be at least 1 or None.")
    if args.empty_threshold < 0:
        raise ValueError("--empty-threshold cannot be negative.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    model = load_model(args, device)

    loader = get_loader(
        data_dir=args.data_dir,
        split="validation",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        dataset_root=args.dataset_root,
    )

    result = run(model, loader, args, device)
    print_results(result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"\nSaved results to: {output}")


if __name__ == "__main__":
    main()
