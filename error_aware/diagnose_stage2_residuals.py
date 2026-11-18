"""
Inspect the Refiner's signed residuals on validation crops.

Run from the project root:
    python error_aware/diagnose_stage2_residuals.py

This is inference-only. It does not train or modify checkpoints.
"""

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F

from data import get_loader
from model import ESRNet, select_topk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose signed Stage-2 residuals."
    )
    parser.add_argument("--data-dir", default="error_aware/processed")
    parser.add_argument(
        "--stage1-checkpoint",
        default="error_aware/checkpoints/coarse_best.pth",
    )
    parser.add_argument(
        "--stage2-checkpoint",
        default="error_aware/checkpoints_stage2_stability/stage2_best.pth",
    )
    parser.add_argument("--max-validation-batches", type=int, default=50)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def state_dict_from(checkpoint):
    return checkpoint.get(
        "model_state_dict",
        checkpoint.get("model", checkpoint),
    )


def make_stats():
    return {
        "samples": 0,
        "selected_cells": 0,
        "raw_sum": 0.0,
        "raw_abs_sum": 0.0,
        "raw_positive": 0,
        "raw_negative": 0,
        "raw_zero": 0,
        "raw_min": float("inf"),
        "raw_max": float("-inf"),
        "effective_sum": 0.0,
        "effective_abs_sum": 0.0,
        "effective_positive": 0,
        "effective_negative": 0,
        "effective_min": float("inf"),
        "effective_max": float("-inf"),
        "crop_correction_sum": 0.0,
        "coarse_count_sum": 0.0,
        "refined_count_sum": 0.0,
        "gt_count_sum": 0.0,
    }


def add_values(stats, raw, effective, crop_correction, coarse_count, refined_count, gt_count):
    # raw/effective are [number_of_selected_cells, 1, 4, 4]
    raw = raw.detach().float()
    effective = effective.detach().float()

    stats["samples"] += 1
    stats["selected_cells"] += raw.shape[0]
    stats["raw_sum"] += raw.sum().item()
    stats["raw_abs_sum"] += raw.abs().sum().item()
    stats["raw_positive"] += int((raw > 0).sum().item())
    stats["raw_negative"] += int((raw < 0).sum().item())
    stats["raw_zero"] += int((raw == 0).sum().item())
    stats["raw_min"] = min(stats["raw_min"], raw.min().item())
    stats["raw_max"] = max(stats["raw_max"], raw.max().item())

    stats["effective_sum"] += effective.sum().item()
    stats["effective_abs_sum"] += effective.abs().sum().item()
    stats["effective_positive"] += int((effective > 0).sum().item())
    stats["effective_negative"] += int((effective < 0).sum().item())
    stats["effective_min"] = min(
        stats["effective_min"], effective.min().item()
    )
    stats["effective_max"] = max(
        stats["effective_max"], effective.max().item()
    )

    stats["crop_correction_sum"] += float(crop_correction)
    stats["coarse_count_sum"] += float(coarse_count)
    stats["refined_count_sum"] += float(refined_count)
    stats["gt_count_sum"] += float(gt_count)


def print_stats(name, stats):
    print(f"\n{name}")
    if stats["selected_cells"] == 0:
        print("  No selected cells.")
        return

    pixels = stats["selected_cells"] * 16
    cells = stats["selected_cells"]
    print(f"  Samples containing selected cells: {stats['samples']}")
    print(f"  Selected cells:                     {cells}")
    print(
        f"  Raw residual mean:                  "
        f"{stats['raw_sum'] / pixels:.8f}"
    )
    print(
        f"  Raw residual mean absolute:         "
        f"{stats['raw_abs_sum'] / pixels:.8f}"
    )
    print(
        f"  Raw residual positive/negative/zero:"
        f" {stats['raw_positive'] / pixels:.2%} /"
        f" {stats['raw_negative'] / pixels:.2%} /"
        f" {stats['raw_zero'] / pixels:.2%}"
    )
    print(
        f"  Raw residual min/max:               "
        f"{stats['raw_min']:.8f} / {stats['raw_max']:.8f}"
    )
    print(
        f"  Effective correction mean:          "
        f"{stats['effective_sum'] / pixels:.8f}"
    )
    print(
        f"  Effective correction positive/negative:"
        f" {stats['effective_positive'] / pixels:.2%} /"
        f" {stats['effective_negative'] / pixels:.2%}"
    )
    print(
        f"  Effective correction min/max:       "
        f"{stats['effective_min']:.8f} / "
        f"{stats['effective_max']:.8f}"
    )
    print(
        f"  Mean signed count correction/crop:  "
        f"{stats['crop_correction_sum'] / max(stats['samples'], 1):.6f}"
    )
    print(
        f"  Mean coarse/refined/GT count:       "
        f"{stats['coarse_count_sum'] / max(stats['samples'], 1):.6f} / "
        f"{stats['refined_count_sum'] / max(stats['samples'], 1):.6f} / "
        f"{stats['gt_count_sum'] / max(stats['samples'], 1):.6f}"
    )


@torch.no_grad()
def main():
    args = parse_args()
    if not 0.0 <= args.budget <= 1.0:
        raise ValueError("--budget must be between 0 and 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    stage2_ckpt = torch.load(
        args.stage2_checkpoint, map_location="cpu", weights_only=False
    )
    ckpt_args = stage2_ckpt.get("args", {})
    refine_width = int(ckpt_args.get("refine_width", 96))

    model = ESRNet(
        pretrained=False,
        freeze_early=True,
        refine_width=refine_width,
    ).to(device)

    stage1_ckpt = torch.load(
        args.stage1_checkpoint, map_location=device, weights_only=False
    )
    model.coarse.load_state_dict(state_dict_from(stage1_ckpt), strict=True)
    model.load_state_dict(state_dict_from(stage2_ckpt), strict=True)
    model.eval()
    model.coarse.eval()

    print(f"Loaded Stage-1: {args.stage1_checkpoint}")
    print(f"Loaded Stage-2: {args.stage2_checkpoint}")
    print(f"Budget: {args.budget:.0%}")

    # Inspect the final residual-prediction convolution.
    final_layer = model.refiner.body[-1]
    if not isinstance(final_layer, torch.nn.Conv2d):
        raise TypeError(
            "Expected model.refiner.body[-1] to be the final Conv2d; "
            f"found {type(final_layer).__name__}. Inspect model.py."
        )

    print("\nFINAL REFINER CONVOLUTION")
    print(f"  Weight shape: {tuple(final_layer.weight.shape)}")
    print(
        f"  Weight mean/std/min/max: "
        f"{final_layer.weight.detach().float().mean().item():.8f} / "
        f"{final_layer.weight.detach().float().std().item():.8f} / "
        f"{final_layer.weight.detach().float().min().item():.8f} / "
        f"{final_layer.weight.detach().float().max().item():.8f}"
    )
    if final_layer.bias is None:
        print("  Bias: None")
    else:
        print(
            f"  Bias value: {final_layer.bias.detach().float().cpu().tolist()}"
        )

    loader = get_loader(
        data_dir=args.data_dir,
        split="validation",
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    groups = {
        "all": make_stats(),
        "empty": make_stats(),
        "nonempty": make_stats(),
    }
    amp_enabled = device.type == "cuda" and not args.no_amp
    print(f"\nAMP enabled: {amp_enabled}")

    for step, (images, target_density, _) in enumerate(loader, start=1):
        if (
            args.max_validation_batches is not None
            and step > args.max_validation_batches
        ):
            break

        images = images.to(device, non_blocking=True)
        target_density = target_density.to(device, non_blocking=True).float()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            coarse_outputs = model.coarse(images)
            coarse = coarse_outputs["density"]
            scores = model.error_head(
                coarse_outputs["context"], coarse
            )
            mask = select_topk(scores.float(), args.budget)

        selected = mask[:, 0].bool().nonzero(as_tuple=False)
        if selected.numel() == 0:
            continue

        # Reproduce apply_refine's window extraction to inspect the raw residual
        # before it is scattered into the full density map.
        density_s4 = F.interpolate(
            coarse.float() / 4.0,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        feat_windows, den_windows, batch_idx, cell_y, cell_x = (
            model._extract_selected_windows(
                coarse_outputs["stride4_features"],
                density_s4,
                selected,
            )
        )

        # Match normal inference: run RefineNet under autocast. The model's
        # final residual-prediction convolution handles its FP32 computation
        # internally, so calling RefineNet outside autocast causes a dtype mismatch.
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            residual_s4 = model.refiner(feat_windows, den_windows)

        residual_s8 = F.avg_pool2d(residual_s4.float(), 2, 2) * 4.0
        raw_central = residual_s8[:, :, 4:8, 4:8].float()

        # Compute the actual correction after the nonnegative density clamp.
        effective_cells = []
        for i in range(selected.shape[0]):
            b = int(batch_idx[i].item())
            y = int(cell_y[i].item()) * 4
            x = int(cell_x[i].item()) * 4
            coarse_cell = coarse[b:b + 1, :, y:y + 4, x:x + 4].float()
            refined_cell = (coarse_cell + raw_central[i:i + 1]).clamp_min(0.0)
            effective_cells.append(refined_cell - coarse_cell)
        effective_central = torch.cat(effective_cells, dim=0)

        coarse = coarse.float()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            refined = model.apply_refine(
                coarse,
                coarse_outputs["stride4_features"],
                mask,
                clamp_output=True,
            ).float()

        gt_count = target_density.sum().item()
        coarse_count = coarse.sum().item()
        refined_count = refined.sum().item()
        crop_correction = refined_count - coarse_count

        # One image per batch is used by this diagnostic's loader.
        group_name = "empty" if gt_count < 0.5 else "nonempty"
        add_values(
            groups["all"],
            raw_central,
            effective_central,
            crop_correction,
            coarse_count,
            refined_count,
            gt_count,
        )
        add_values(
            groups[group_name],
            raw_central,
            effective_central,
            crop_correction,
            coarse_count,
            refined_count,
            gt_count,
        )

        if step == 1 or step % 10 == 0:
            print(f"Processed samples: {step}")

    print("\n" + "=" * 72)
    print("SIGNED RESIDUAL DIAGNOSTIC")
    print("=" * 72)
    print_stats("ALL CROPS", groups["all"])
    print_stats("EMPTY CROPS (GT < 0.5)", groups["empty"])
    print_stats("NONEMPTY CROPS (GT >= 0.5)", groups["nonempty"])

    print("\nHow to read this:")
    print("- Positive raw residual mean and high positive fraction indicate an upward residual bias.")
    print("- If raw residual is positive but effective correction is smaller, the density clamp is limiting negative corrections.")
    print("- Compare empty vs nonempty crops to see whether the bias is concentrated on empty scenes.")
    print("- The final convolution bias is a parameter value, not by itself proof of the whole residual bias.")
    print("\nNo training or checkpoint modification was performed.")


if __name__ == "__main__":
    main()
