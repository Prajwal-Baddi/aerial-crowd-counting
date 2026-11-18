"""
Compare Stage-1 coarse predictions with Stage-2 refined predictions on the
same validation samples, and inspect the refiner's multiplicative/additive
outputs.

Run from the project root:
    python error_aware/diagnose_stage2_predictions.py

Example:
    python error_aware/diagnose_stage2_predictions.py \
        --stage2-checkpoint error_aware/checkpoints_stage2_multiplicative_stability/stage2_best.pth \
        --max-validation-batches 50 --budget 0.25 --no-amp

This script performs inference only. It does not train or modify checkpoints.
"""

import argparse
from pathlib import Path

import torch

from data import get_loader
from model_multiplicative import (
    ESRNet,
    cell_sum,
    mask_to_pixels,
    select_topk,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Stage-1 and Stage-2 validation predictions and "
            "diagnose refiner outputs."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="error_aware/processed",
        help="Directory containing the processed dataset manifests.",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        default="error_aware/checkpoints/coarse_best.pth",
        help="Stage-1 coarse checkpoint.",
    )
    parser.add_argument(
        "--stage2-checkpoint",
        default=(
            "error_aware/checkpoints_stage2_stability/stage2_best.pth"
        ),
        help="Stage-2 multiplicative-additive checkpoint.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=50,
        help="Maximum number of validation batches; use 0 for all batches.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.25,
        help="Fraction of 32x32 image cells to refine.",
    )
    parser.add_argument(
        "--refine-width",
        type=int,
        default=None,
        help=(
            "Override refiner width. By default, read it from the "
            "Stage-2 checkpoint arguments."
        ),
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Run inference in FP32 instead of CUDA autocast.",
    )
    return parser.parse_args()


def extract_state_dict(checkpoint):
    """Support the checkpoint formats used by the training scripts."""
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
    return checkpoint


def load_checkpoints(model, stage1_path, stage2_path, device):
    stage1_path = Path(stage1_path)
    stage2_path = Path(stage2_path)

    if not stage1_path.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {stage1_path}")
    if not stage2_path.is_file():
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {stage2_path}")

    stage1_checkpoint = torch.load(
        stage1_path,
        map_location=device,
        weights_only=False,
    )
    model.coarse.load_state_dict(
        extract_state_dict(stage1_checkpoint),
        strict=True,
    )

    stage2_checkpoint = torch.load(
        stage2_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(
        extract_state_dict(stage2_checkpoint),
        strict=True,
    )

    print(f"Loaded Stage-1 checkpoint: {stage1_path}")
    print(f"Loaded Stage-2 checkpoint: {stage2_path}")
    return stage2_checkpoint


def new_bucket():
    return {
        "n": 0,
        "gt_sum": 0.0,
        "coarse_sum": 0.0,
        "refined_sum": 0.0,
        "coarse_abs_error": 0.0,
        "refined_abs_error": 0.0,
        "coarse_squared_error": 0.0,
        "refined_squared_error": 0.0,
        "coarse_density_min": float("inf"),
        "coarse_density_max": float("-inf"),
        "refined_density_min": float("inf"),
        "refined_density_max": float("-inf"),
        "absolute_density_correction_sum": 0.0,
        "selected_cell_correction_sum": 0.0,
        "selected_cells": 0,
        "selected_pixel_count": 0,
    }


def add_bucket(bucket, target, coarse, refined, mask):
    target_count = target.float().sum(dim=(1, 2, 3))
    coarse_count = coarse.float().sum(dim=(1, 2, 3))
    refined_count = refined.float().sum(dim=(1, 2, 3))

    coarse_error = coarse_count - target_count
    refined_error = refined_count - target_count

    batch_size = target.shape[0]
    bucket["n"] += batch_size
    bucket["gt_sum"] += target_count.sum().item()
    bucket["coarse_sum"] += coarse_count.sum().item()
    bucket["refined_sum"] += refined_count.sum().item()
    bucket["coarse_abs_error"] += coarse_error.abs().sum().item()
    bucket["refined_abs_error"] += refined_error.abs().sum().item()
    bucket["coarse_squared_error"] += coarse_error.square().sum().item()
    bucket["refined_squared_error"] += refined_error.square().sum().item()

    coarse_float = coarse.float()
    refined_float = refined.float()

    bucket["coarse_density_min"] = min(
        bucket["coarse_density_min"], coarse_float.min().item()
    )
    bucket["coarse_density_max"] = max(
        bucket["coarse_density_max"], coarse_float.max().item()
    )
    bucket["refined_density_min"] = min(
        bucket["refined_density_min"], refined_float.min().item()
    )
    bucket["refined_density_max"] = max(
        bucket["refined_density_max"], refined_float.max().item()
    )

    density_delta = (refined_float - coarse_float).abs()
    bucket["absolute_density_correction_sum"] += density_delta.sum().item()

    # One selected mask cell corresponds to a 4x4 region in the 64x64
    # stride-8 density map.
    selected_cell_mask = mask.float()
    coarse_cell_counts = cell_sum(coarse_float)
    refined_cell_counts = cell_sum(refined_float)
    cell_delta = (refined_cell_counts - coarse_cell_counts).abs()

    bucket["selected_cell_correction_sum"] += (
        cell_delta * selected_cell_mask
    ).sum().item()

    bucket["selected_cells"] += int(selected_cell_mask.sum().item())
    bucket["selected_pixel_count"] += int(
        mask_to_pixels(mask, scale=4).sum().item()
    )


def print_bucket(name, bucket):
    n = bucket["n"]
    print(f"\n{name} (n={n})")
    if n == 0:
        print("  No samples.")
        return

    coarse_mae = bucket["coarse_abs_error"] / n
    refined_mae = bucket["refined_abs_error"] / n
    coarse_rmse = (bucket["coarse_squared_error"] / n) ** 0.5
    refined_rmse = (bucket["refined_squared_error"] / n) ** 0.5

    print(f"  Mean GT count:                 {bucket['gt_sum'] / n:.6f}")
    print(f"  Mean coarse predicted count:   {bucket['coarse_sum'] / n:.6f}")
    print(f"  Mean refined predicted count:  {bucket['refined_sum'] / n:.6f}")
    print(f"  Coarse MAE / RMSE:             {coarse_mae:.6f} / {coarse_rmse:.6f}")
    print(f"  Refined MAE / RMSE:            {refined_mae:.6f} / {refined_rmse:.6f}")
    print(
        f"  Coarse density min / max:      "
        f"{bucket['coarse_density_min']:.8f} / "
        f"{bucket['coarse_density_max']:.8f}"
    )
    print(
        f"  Refined density min / max:     "
        f"{bucket['refined_density_min']:.8f} / "
        f"{bucket['refined_density_max']:.8f}"
    )

    mean_pixel_correction = (
        bucket["absolute_density_correction_sum"] / max(n * 64 * 64, 1)
    )
    print(f"  Mean |refined - coarse|/pixel: {mean_pixel_correction:.8f}")
    print(f"  Selected cells:                {bucket['selected_cells']}")

    if bucket["selected_cells"]:
        mean_cell_correction = (
            bucket["selected_cell_correction_sum"] / bucket["selected_cells"]
        )
        print(
            "  Mean abs count correction / selected cell: "
            f"{mean_cell_correction:.8f}"
        )


def print_tensor_stats(name, values, reference=None):
    """Print robust distribution statistics for a concatenated CPU tensor."""
    if values.numel() == 0:
        print(f"\n{name}: no selected cells were processed.")
        return

    values = values.float()
    print(f"\n{name}")
    print(f"  Number of values: {values.numel()}")
    print(f"  Mean:             {values.mean().item():.10f}")
    print(f"  Std:              {values.std(unbiased=False).item():.10f}")
    print(f"  Min:              {values.min().item():.10f}")
    print(f"  Max:              {values.max().item():.10f}")

    if reference is not None:
        reference = float(reference)
        print(
            f"  Mean absolute distance from {reference:g}: "
            f"{(values - reference).abs().mean().item():.10f}"
        )

    quantiles = torch.tensor(
        [0.0, 0.25, 0.50, 0.75, 0.95, 1.0],
        dtype=torch.float32,
    )
    q_values = torch.quantile(values, quantiles)
    print("  Quantiles:")
    for q, value in zip(quantiles.tolist(), q_values.tolist()):
        print(f"    {q * 100:5.1f}%: {value:.10f}")


@torch.no_grad()
def main():
    args = parse_args()

    if not 0.0 <= args.budget <= 1.0:
        raise ValueError("--budget must be between 0 and 1.")
    if args.max_validation_batches < 0:
        raise ValueError("--max-validation-batches must be >= 0.")
    if args.refine_width is not None and args.refine_width <= 0:
        raise ValueError("--refine-width must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    stage2_path = Path(args.stage2_checkpoint)
    if not stage2_path.is_file():
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {stage2_path}")

    stage2_checkpoint = torch.load(
        stage2_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_args = stage2_checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}

    refine_width = args.refine_width
    if refine_width is None:
        refine_width = int(checkpoint_args.get("refine_width", 96))

    model = ESRNet(
        pretrained=False,
        freeze_early=True,
        refine_width=refine_width,
    ).to(device)

    load_checkpoints(
        model,
        args.stage1_checkpoint,
        args.stage2_checkpoint,
        device,
    )

    model.eval()
    model.coarse.eval()

    loader = get_loader(
        data_dir=args.data_dir,
        split="validation",
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    all_bucket = new_bucket()
    empty_bucket = new_bucket()
    nonempty_bucket = new_bucket()

    # Store actual outputs emitted by RefineNet, not values reconstructed
    # from the final density map.
    gain_values = []
    add_values = []

    amp_enabled = device.type == "cuda" and not args.no_amp
    print(f"AMP enabled: {amp_enabled}")
    print(f"Refinement budget: {args.budget:.0%}")
    print(
        "Validation batch limit: "
        f"{args.max_validation_batches if args.max_validation_batches else 'all'}"
    )
    print(f"Refiner width: {refine_width}")

    processed_batches = 0

    for step, (images, target_density, _) in enumerate(loader, start=1):
        if args.max_validation_batches and step > args.max_validation_batches:
            break

        images = images.to(device, non_blocking=True)
        target_density = target_density.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            coarse_outputs = model.coarse(images)
            coarse_density = coarse_outputs["density"]

            error_scores = model.error_head(
                coarse_outputs["context"],
                coarse_density,
            )
            mask = select_topk(error_scores.float(), args.budget)

            refined_density, refine_aux = model.apply_refine(
                coarse_density,
                coarse_outputs["stride4_features"],
                mask,
                return_aux=True,
            )

        # Keep all metric calculations in FP32.
        coarse_density = coarse_density.float()
        refined_density = refined_density.float()
        target_density = target_density.float()

        # Collect the actual gain/add tensors predicted for selected cells.
        # Empty tensors are valid when budget is zero.
        gain_values.append(refine_aux["gain"].detach().float().cpu().flatten())
        add_values.append(refine_aux["add"].detach().float().cpu().flatten())

        add_bucket(
            all_bucket,
            target_density,
            coarse_density,
            refined_density,
            mask,
        )

        target_counts = target_density.sum(dim=(1, 2, 3))
        is_empty = target_counts < 0.5

        if is_empty.any():
            indices = is_empty.nonzero(as_tuple=True)[0]
            add_bucket(
                empty_bucket,
                target_density[indices],
                coarse_density[indices],
                refined_density[indices],
                mask[indices],
            )

        if (~is_empty).any():
            indices = (~is_empty).nonzero(as_tuple=True)[0]
            add_bucket(
                nonempty_bucket,
                target_density[indices],
                coarse_density[indices],
                refined_density[indices],
                mask[indices],
            )

        processed_batches += 1
        if step == 1 or step % 10 == 0:
            print(f"Processed validation samples: {all_bucket['n']}")

    if processed_batches == 0:
        raise RuntimeError(
            "No validation samples were processed. Check --data-dir and "
            "the validation manifest."
        )

    print("\n" + "=" * 72)
    print("REFINER OUTPUT DIAGNOSTICS")
    print("=" * 72)

    if gain_values:
        gain = torch.cat(gain_values)
        add = torch.cat(add_values)
        print_tensor_stats("MULTIPLICATIVE GAIN (target identity = 1)", gain, 1.0)
        print_tensor_stats("ADDITIVE CORRECTION (target identity = 0)", add, 0.0)
        print(f"\nTotal additive correction mass: {add.sum().item():.10f}")
    else:
        print("No refiner outputs were collected.")

    print("\n" + "=" * 72)
    print("STAGE-1 vs STAGE-2 PREDICTION DIAGNOSTIC")
    print("=" * 72)
    print_bucket("ALL VALIDATION SAMPLES", all_bucket)
    print_bucket("EMPTY CROPS (GT count < 0.5)", empty_bucket)
    print_bucket("NONEMPTY CROPS (GT count >= 0.5)", nonempty_bucket)

    print("\nInterpretation guide:")
    print(
        "- Gain near 1 and additive correction near 0 mean the refiner "
        "is close to identity."
    )
    print(
        "- A very small gain spread means the multiplicative branch "
        "is producing nearly identical corrections."
    )
    print(
        "- Additive values near zero mean the additive branch contributes "
        "very little density."
    )
    print(
        "- Large additive values or gain values far from 1 may indicate "
        "aggressive corrections."
    )
    print(
        "- Compare coarse and refined MAE separately for empty and "
        "non-empty crops."
    )
    print(
        "\nDiagnostic complete. No checkpoints or training files were modified."
    )


if __name__ == "__main__":
    main()
