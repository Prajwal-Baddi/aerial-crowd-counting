
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import get_loader
from model_multiplicative import ESRNet, select_topk


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare gain-only, additive-only and combined Stage-2 refinement."
    )
    p.add_argument("--data-dir", default="error_aware/processed")
    p.add_argument(
        "--dataset-root",
        default="dataset/VisDrone2019-DET-train"
    )
    p.add_argument(
        "--coarse-checkpoint",
        default="error_aware/checkpoints/coarse_best.pth"
    )
    p.add_argument("--stage2-checkpoint", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--budget", type=float, default=0.25)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument(
        "--output",
        default="error_aware/refinement_branch_diagnostic.json"
    )
    return p.parse_args()


def load_coarse(model, checkpoint_path, device):
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )
    state = ckpt.get(
        "model_state_dict",
        ckpt.get("model", ckpt)
    )
    model.coarse.load_state_dict(state, strict=True)

    for parameter in model.coarse.parameters():
        parameter.requires_grad_(False)

    model.coarse.eval()
    print(f"Loaded coarse checkpoint: {checkpoint_path}")


def load_stage2(model, checkpoint_path, device):
    if checkpoint_path is None:
        print(
            "WARNING: No Stage-2 checkpoint supplied. "
            "Using a freshly initialized refiner."
        )
        return

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )
    state = ckpt.get("model_state_dict", ckpt)

    model.load_state_dict(state, strict=True)
    print(f"Loaded Stage-2 checkpoint: {checkpoint_path}")

    if "epoch" in ckpt:
        print(f"Stage-2 checkpoint epoch: {ckpt['epoch']}")


def scatter_cells(coarse, selected, replacement_cells):
    """Replace selected 4x4 density cells in a [B,1,64,64] map."""
    result = coarse.clone().float()

    if selected.numel() == 0:
        return result

    batch_idx, cell_y, cell_x = selected.unbind(dim=1)

    cell_size = 4
    offsets = torch.arange(
        cell_size,
        device=coarse.device
    )

    rows = cell_y[:, None] * cell_size + offsets[None, :]
    cols = cell_x[:, None] * cell_size + offsets[None, :]

    result[
        batch_idx[:, None, None],
        0,
        rows[:, :, None],
        cols[:, None, :]
    ] = replacement_cells[:, 0]

    return result


def update_metrics(storage, prediction, target, is_empty):
    predicted_counts = prediction.float().sum(
        dim=(1, 2, 3)
    ).cpu().numpy()

    errors = np.abs(predicted_counts - target)

    storage["all"].extend(errors.tolist())

    if is_empty.any():
        storage["empty"].extend(
            errors[is_empty].tolist()
        )

    if (~is_empty).any():
        storage["nonempty"].extend(
            errors[~is_empty].tolist()
        )

    storage["predicted_counts"].extend(
        predicted_counts.tolist()
    )


def summarize(storage, coarse_counts):
    result = {}

    for category in ("all", "empty", "nonempty"):
        values = storage[category]

        result[category + "_mae"] = (
            float(np.mean(values)) if values else None
        )

    predicted = np.asarray(storage["predicted_counts"])
    coarse_counts = np.asarray(coarse_counts)

    result["mean_count_delta"] = (
        float(np.mean(predicted - coarse_counts))
        if len(predicted) else None
    )

    return result


@torch.no_grad()
def run_diagnostic(model, loader, device, budget, max_batches):
    model.eval()
    model.coarse.eval()

    names = (
        "coarse",
        "gain_only",
        "additive_only",
        "combined"
    )

    metrics = {
        name: {
            "all": [],
            "empty": [],
            "nonempty": [],
            "predicted_counts": []
        }
        for name in names
    }

    gt_counts = []
    coarse_counts = []
    selected_counts = []

    for step, (images, gt_density, counts) in enumerate(loader, 1):
        if max_batches is not None and step > max_batches:
            break

        images = images.to(device, non_blocking=True)
        gt_density = gt_density.to(
            device,
            non_blocking=True
        ).float()

        counts_np = counts.float().cpu().numpy()

        coarse_output = model.coarse(images)

        coarse = coarse_output["density"].float()
        features = coarse_output["stride4_features"]

        # Oracle selection: choose cells with the largest
        # absolute coarse count error against the ground truth.
        coarse_cells = coarse.unfold(
            2, 4, 4
        ).unfold(3, 4, 4).sum(dim=(-1, -2))

        gt_cells = gt_density.unfold(
            2, 4, 4
        ).unfold(3, 4, 4).sum(dim=(-1, -2))

        oracle_error = (coarse_cells - gt_cells).abs()

        mask = select_topk(
            oracle_error,
            budget
        )

        # Run the refiner once. All branches use the
        # exact same selected cells and predicted corrections.
        _, aux = model.apply_refine(
            coarse,
            features,
            mask,
            return_aux=True
        )

        if aux is None:
            gain_map = coarse.clone()
            add_map = coarse.clone()
            combined_map = coarse.clone()
        else:
            selected = aux["selected"]
            base_cells = aux["coarse_cells"]
            gain = aux["gain"]
            add = aux["add"]

            gain_cells = base_cells * gain
            additive_cells = base_cells + add
            combined_cells = base_cells * gain + add

            gain_map = scatter_cells(
                coarse, selected, gain_cells
            )
            add_map = scatter_cells(
                coarse, selected, additive_cells
            )
            combined_map = scatter_cells(
                coarse, selected, combined_cells
            )

        outputs = {
            "coarse": coarse,
            "gain_only": gain_map,
            "additive_only": add_map,
            "combined": combined_map
        }

        is_empty = counts.float().cpu().numpy() == 0

        gt_counts.extend(counts_np.tolist())

        coarse_batch_counts = coarse.sum(
            dim=(1, 2, 3)
        ).cpu().numpy()

        coarse_counts.extend(
            coarse_batch_counts.tolist()
        )

        selected_counts.extend(
            mask.float().sum(dim=(1, 2, 3)).cpu().tolist()
        )

        for name, prediction in outputs.items():
            update_metrics(
                metrics[name],
                prediction,
                counts_np,
                is_empty
            )

        if step == 1 or step % 50 == 0:
            print(
                f"Processed {step} batches | "
                f"selected cells: "
                f"{mask.float().mean().item():.3f}"
            )

    result = {
        "samples": len(gt_counts),
        "budget": budget,
        "selection": "oracle",
        "mean_selected_cells": (
            float(np.mean(selected_counts))
            if selected_counts else None
        ),
        "results": {}
    }

    for name in names:
        result["results"][name] = summarize(
            metrics[name],
            coarse_counts
        )

    return result


def print_results(result):
    print("\n" + "=" * 76)
    print("ORACLE REFINEMENT BRANCH DIAGNOSTIC")
    print("=" * 76)

    print(f"Samples: {result['samples']}")
    print(f"Oracle budget: {result['budget']:.2f}")
    print(
        f"Mean selected cells: "
        f"{result['mean_selected_cells']:.2f}"
    )

    print(
        f"\n{'Branch':<18}"
        f"{'Overall MAE':>14}"
        f"{'Empty MAE':>13}"
        f"{'Nonempty MAE':>15}"
        f"{'Mean dCount':>15}"
    )

    print("-" * 76)

    for name, values in result["results"].items():
        def fmt(value):
            return "n/a" if value is None else f"{value:.4f}"

        print(
            f"{name:<18}"
            f"{fmt(values['all_mae']):>14}"
            f"{fmt(values['empty_mae']):>13}"
            f"{fmt(values['nonempty_mae']):>15}"
            f"{fmt(values['mean_count_delta']):>15}"
        )

    baseline = result["results"]["coarse"]["all_mae"]

    print("\nImprovement relative to coarse MAE:")

    for name in ("gain_only", "additive_only", "combined"):
        value = result["results"][name]["all_mae"]

        if value is not None:
            print(
                f"{name:<18}: "
                f"{baseline - value:+.4f}"
            )

    print("=" * 76)


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    model = ESRNet(
        pretrained=False,
        freeze_early=True,
        refine_width=96
    ).to(device)

    load_coarse(
        model,
        args.coarse_checkpoint,
        device
    )

    load_stage2(
        model,
        args.stage2_checkpoint,
        device
    )

    loader = get_loader(
        data_dir=args.data_dir,
        split="validation",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        dataset_root=args.dataset_root
    )

    result = run_diagnostic(
        model,
        loader,
        device,
        args.budget,
        args.max_batches
    )

    print_results(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as file:
        json.dump(result, file, indent=2)

    print(f"\nSaved diagnostic results to: {output_path}")


if __name__ == "__main__":
    main()
