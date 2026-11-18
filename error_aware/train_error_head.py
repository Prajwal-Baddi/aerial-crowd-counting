"""Train the 3-way ErrorHead while keeping the Stage-1 coarse counter frozen.

Classes:
    0: fine        - coarse count is within DELTA of ground truth
    1: increase    - coarse count underestimates ground truth
    2: decrease    - coarse count overestimates ground truth

Diagnostics-only example (does not train or save a checkpoint):
    python error_aware/train_error_head.py ^
      --diagnostics-only ^
      --manifest error_aware/processed/stage2_train20_holdout.jsonl ^
      --checkpoint error_aware/checkpoints_stage1_holdout80/coarse_best.pth ^
      --cells-cache error_aware/analysis/cells.npz
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from data import ErrorAwareCrowdDataset
from head_metrics import (
    class_distribution,
    head_report,
    print_class_distribution,
    print_head_report,
)
from model_gated import ESRNet, cell_sum
from selection_baselines import DELTA


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or diagnose the ErrorHead with a frozen Stage-1 counter."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cells-cache", default=None)
    parser.add_argument("--data-dir", default="error_aware/processed")
    parser.add_argument(
        "--dataset-root", default="dataset/VisDrone2019-DET-train"
    )
    parser.add_argument(
        "--output-dir", default="error_aware/checkpoints_error_head"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples-per-epoch", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha-fine", type=float, default=0.05)
    parser.add_argument("--val-subset", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Report class balance and ErrorHead output diagnostics without training.",
    )
    return parser.parse_args()


def make_labels(coarse_cells, gt_cells):
    """Return labels with shape [batch, cells]."""
    error = (coarse_cells - gt_cells)[:, 0]
    labels = torch.zeros_like(error, dtype=torch.long)
    labels[error <= -DELTA] = 1
    labels[error >= DELTA] = 2
    return labels


def focal_loss(logits, labels, gamma, alpha_fine):
    """Focal loss; normalize by the number of material-error cells."""
    log_prob = F.log_softmax(logits.float(), dim=1)
    log_prob_target = log_prob.gather(1, labels[:, None])[:, 0]

    alpha = torch.where(
        labels == 0,
        torch.full_like(log_prob_target, alpha_fine),
        torch.ones_like(log_prob_target),
    )
    per_cell_loss = (
        -alpha
        * (1.0 - log_prob_target.exp()).pow(gamma)
        * log_prob_target
    )
    return per_cell_loss.sum() / (labels > 0).sum().clamp(min=1)


def load_cells_cache(cache_path, expected_crops):
    """Load cached coarse/GT cell counts and validate their dimensions."""
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cells cache not found: {path}")

    cache = np.load(path, allow_pickle=False)
    if "coarse" not in cache or "gt" not in cache:
        raise KeyError(
            f"{path} must contain 'coarse' and 'gt'; found {list(cache.files)}"
        )

    coarse = np.asarray(cache["coarse"])
    gt = np.asarray(cache["gt"])

    if coarse.shape != gt.shape or coarse.ndim != 2:
        raise RuntimeError(
            "Expected matching 2-D [crops, cells] arrays; "
            f"got coarse={coarse.shape}, gt={gt.shape}"
        )
    if coarse.shape[0] != expected_crops:
        raise RuntimeError(
            f"Cache contains {coarse.shape[0]} crops, but the manifest contains "
            f"{expected_crops}. Regenerate the cache for this exact manifest "
            "and ordering."
        )

    return coarse, gt


def build_train_loader(args):
    dataset = ErrorAwareCrowdDataset(
        Path(args.manifest),
        dataset_root=args.dataset_root,
        crop_size=512,
        density_stride=8,
        gaussian_sigma=3.0,
    )

    if args.cells_cache:
        coarse, gt = load_cells_cache(args.cells_cache, len(dataset))
        crop_has_error = (np.abs(coarse - gt) >= DELTA).any(axis=1)
        sampling_description = "crops containing a material-error cell"
    else:
        crop_has_error = np.array(
            [record["count"] > 0 for record in dataset.records],
            dtype=bool,
        )
        sampling_description = "non-empty crops (no coarse-error cache supplied)"

    positive_count = int(crop_has_error.sum())
    negative_count = int((~crop_has_error).sum())

    # Equal total sampling weight for positive and negative crop groups.
    weights = np.where(
        crop_has_error,
        0.5 / max(positive_count, 1),
        0.5 / max(negative_count, 1),
    )
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=args.samples_per_epoch,
        replacement=True,
    )

    print(
        f"Training crops: {len(dataset)} | 50/50 sampling of "
        f"{sampling_description}: {positive_count} vs {negative_count}"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    return dataset, loader


def build_validation_loader(args, subset_size):
    dataset = ErrorAwareCrowdDataset(
        Path(args.data_dir) / "validation.jsonl",
        dataset_root=args.dataset_root,
        crop_size=512,
        density_stride=8,
        gaussian_sigma=3.0,
    )

    if subset_size > 0 and subset_size < len(dataset):
        indices = np.random.RandomState(0).permutation(len(dataset))
        indices = sorted(indices[:subset_size].tolist())
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def collect_predictions(model, loader, device, amp, max_batches=None):
    """Collect coarse/GT cell counts and ErrorHead probabilities."""
    model.eval()
    coarse_all = []
    gt_all = []
    probabilities_all = []

    for step, (images, gt_density, _) in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break

        images = images.to(device, non_blocking=True)
        gt_density = gt_density.to(device, non_blocking=True).float()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            coarse_output = model.coarse(images)

        coarse_density = coarse_output["density"].float()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            logits = model.error_head(
                coarse_output["context"],
                coarse_density,
                coarse_output["stride4_features"],
            )

        coarse_all.append(
            cell_sum(coarse_density).flatten(1).cpu().numpy()
        )
        gt_all.append(
            cell_sum(gt_density).flatten(1).cpu().numpy()
        )
        probabilities_all.append(
            logits.float().softmax(dim=1).flatten(2).cpu().numpy()
        )

    if not coarse_all:
        raise RuntimeError(
            "No validation batches were collected. Check validation.jsonl."
        )

    return (
        np.concatenate(coarse_all, axis=0),
        np.concatenate(gt_all, axis=0),
        np.concatenate(probabilities_all, axis=0),
    )


def safe_correlation(x, y):
    """Pearson correlation, returning NaN for constant inputs."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.size != y.size or x.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def print_error_head_output_diagnostics(coarse, gt, probabilities):
    """Print probability, spatial variation and coarse-density diagnostics."""
    coarse = np.asarray(coarse, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if coarse.shape != gt.shape:
        raise ValueError(
            f"Coarse/GT shape mismatch: {coarse.shape} vs {gt.shape}"
        )
    if probabilities.ndim != 3 or probabilities.shape[1] != 3:
        raise ValueError(
            "Expected probabilities with shape [crops, 3, cells], "
            f"got {probabilities.shape}"
        )
    if (
        probabilities.shape[0] != coarse.shape[0]
        or probabilities.shape[2] != coarse.shape[1]
    ):
        raise ValueError(
            f"Incompatible coarse/probability shapes: "
            f"{coarse.shape} vs {probabilities.shape}"
        )

    class_names = ("fine", "increase", "decrease")
    predicted_class = probabilities.argmax(axis=1).reshape(-1)

    print("\nERRORHEAD PROBABILITY DISTRIBUTION")
    print("-" * 78)
    print(
        f"{'Class':<12}{'Mean':>12}{'Std':>12}"
        f"{'Min':>12}{'Max':>12}{'Argmax %':>14}"
    )
    print("-" * 74)

    for class_id, class_name in enumerate(class_names):
        values = probabilities[:, class_id, :].reshape(-1)
        argmax_percentage = 100.0 * (predicted_class == class_id).mean()
        print(
            f"{class_name:<12}{values.mean():>12.6f}{values.std():>12.6f}"
            f"{values.min():>12.6f}{values.max():>12.6f}"
            f"{argmax_percentage:>13.3f}%"
        )

    entropy = -(
        probabilities
        * np.log(np.clip(probabilities, 1e-12, 1.0))
    ).sum(axis=1)
    print(f"\nMean predictive entropy: {entropy.mean():.6f} nats")
    print(
        "Mean maximum class probability: "
        f"{probabilities.max(axis=1).mean():.6f}"
    )
    print(
        "Mean probability sum per cell: "
        f"{probabilities.sum(axis=1).mean():.6f}"
    )

    print("\nSPATIAL VARIATION (standard deviation across cells per crop)")
    for class_id, class_name in enumerate(class_names):
        per_crop_std = probabilities[:, class_id, :].std(axis=1)
        print(
            f"  {class_name:<9} mean={per_crop_std.mean():.8f} "
            f"median={np.median(per_crop_std):.8f} "
            f"p95={np.percentile(per_crop_std, 95):.8f}"
        )

    coarse_flat = coarse.reshape(-1)
    p_fine = probabilities[:, 0, :].reshape(-1)
    p_increase = probabilities[:, 1, :].reshape(-1)
    p_decrease = probabilities[:, 2, :].reshape(-1)
    p_error = (1.0 - probabilities[:, 0, :]).reshape(-1)

    print("\nCORRELATION WITH COARSE-DENSITY HEURISTICS")
    print(
        "  corr(P(increase), -coarse density): "
        f"{safe_correlation(p_increase, -coarse_flat):.6f}"
    )
    print(
        "  corr(P(decrease), +coarse density): "
        f"{safe_correlation(p_decrease, coarse_flat):.6f}"
    )
    print(
        "  corr(P(error), |coarse density|): "
        f"{safe_correlation(p_error, np.abs(coarse_flat)):.6f}"
    )
    print(
        "  corr(P(fine), |coarse density|): "
        f"{safe_correlation(p_fine, np.abs(coarse_flat)):.6f}"
    )

    error = coarse - gt
    true_increase = error <= -DELTA
    true_decrease = error >= DELTA
    true_fine = ~(true_increase | true_decrease)
    probabilities_by_cell = probabilities.transpose(0, 2, 1)

    print("\nMEAN PREDICTED PROBABILITY BY TRUE CLASS")
    print(
        f"{'True class':<14}{'P(fine)':>12}"
        f"{'P(increase)':>14}{'P(decrease)':>14}"
    )
    print("-" * 54)

    for class_name, mask in (
        ("fine", true_fine),
        ("increase", true_increase),
        ("decrease", true_decrease),
    ):
        if mask.any():
            means = probabilities_by_cell[mask].mean(axis=0)
            print(
                f"{class_name:<14}{means[0]:>12.6f}"
                f"{means[1]:>14.6f}{means[2]:>14.6f}"
            )
        else:
            print(
                f"{class_name:<14}{'n/a':>12}{'n/a':>14}{'n/a':>14}"
            )


def report_validation(
    model,
    loader,
    device,
    amp,
    max_batches=None,
):
    coarse, gt, probabilities = collect_predictions(
        model,
        loader,
        device,
        amp,
        max_batches=max_batches,
    )

    print_error_head_output_diagnostics(coarse, gt, probabilities)

    distribution = class_distribution(coarse, gt)
    print_class_distribution(
        distribution,
        "VALIDATION CLASS DISTRIBUTION",
    )

    report = head_report(coarse, gt, probabilities)
    print("\nVALIDATION ERRORHEAD METRICS")
    print_head_report(report)
    return report


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.samples_per_epoch < 1:
        raise ValueError("--samples-per-epoch must be at least 1.")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("--max-train-batches must be at least 1.")
    if args.val_subset < 0:
        raise ValueError("--val-subset cannot be negative.")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be at least 1.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    amp = device.type == "cuda" and not args.no_amp

    output_dir = Path(args.output_dir)
    if not args.diagnostics_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Device {device} | AMP {amp} | "
        f"material-error threshold {DELTA} counts"
    )

    model = ESRNet(
        pretrained=False,
        freeze_early=True,
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    coarse_state = checkpoint.get("model_state_dict", checkpoint)
    model.coarse.load_state_dict(coarse_state, strict=True)

    for parameter in model.coarse.parameters():
        parameter.requires_grad_(False)
    model.coarse.eval()

    train_dataset, train_data_loader = build_train_loader(args)

    if args.cells_cache:
        train_coarse, train_gt = load_cells_cache(
            args.cells_cache,
            len(train_dataset),
        )
        train_distribution = class_distribution(
            train_coarse,
            train_gt,
        )
        print_class_distribution(
            train_distribution,
            "TRAINING CACHE CLASS DISTRIBUTION",
        )
    else:
        print(
            "\nTraining direction counts unavailable. Supply --cells-cache "
            "generated for this exact training manifest."
        )

    validation_subset = 0 if args.diagnostics_only else args.val_subset
    validation_data_loader = build_validation_loader(
        args,
        validation_subset,
    )

    if args.diagnostics_only:
        print(
            "\nDiagnostics-only mode: no training, optimizer creation, "
            "or checkpoint saving."
        )
        report_validation(
            model,
            validation_data_loader,
            device,
            amp,
        )
        return

    trainable_parameters = list(model.error_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    epochs = 1 if args.smoke_test else args.epochs
    steps_per_epoch = min(
        len(train_data_loader),
        args.max_train_batches or len(train_data_loader),
    )
    total_steps = max(1, epochs * steps_per_epoch)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            min(1.0, (step + 1) / max(args.warmup_steps, 1))
            * max(
                0.02,
                0.5
                * (
                    1.0
                    + math.cos(
                        math.pi * min(1.0, step / total_steps)
                    )
                ),
            )
        ),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=128.0,
        enabled=amp,
    )

    best_ap = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.error_head.train()
        model.coarse.eval()

        epoch_start = time.time()
        running_loss = 0.0
        batches_seen = 0
        skipped_updates = 0

        for step, (images, gt_density, _) in enumerate(
            train_data_loader,
            start=1,
        ):
            if (
                args.max_train_batches is not None
                and step > args.max_train_batches
            ):
                break

            images = images.to(device, non_blocking=True)
            gt_density = gt_density.to(
                device,
                non_blocking=True,
            ).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp,
                ):
                    coarse_output = model.coarse(images)

            coarse_density = coarse_output["density"].float()

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                logits = model.error_head(
                    coarse_output["context"],
                    coarse_density,
                    coarse_output["stride4_features"],
                )

            labels = make_labels(
                cell_sum(coarse_density),
                cell_sum(gt_density),
            )
            loss = focal_loss(
                logits,
                labels,
                args.gamma,
                args.alpha_fine,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, step {step}."
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=5.0,
            )

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            if scaler.get_scale() < scale_before:
                skipped_updates += 1
            else:
                scheduler.step()

            running_loss += float(loss.item())
            batches_seen += 1

            if step == 1 or step % args.log_interval == 0:
                fine_count = int((labels == 0).sum().item())
                increase_count = int((labels == 1).sum().item())
                decrease_count = int((labels == 2).sum().item())

                print(
                    f"Epoch {epoch:02d} step {step:04d} | "
                    f"focal {loss.item():.4f} | "
                    f"fine {fine_count} | "
                    f"increase {increase_count} | "
                    f"decrease {decrease_count} | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}"
                )

        coarse_val, gt_val, probabilities_val = collect_predictions(
            model,
            validation_data_loader,
            device,
            amp,
            max_batches=20 if args.smoke_test else None,
        )
        validation_report = head_report(
            coarse_val,
            gt_val,
            probabilities_val,
        )
        material_ap = validation_report["ap"]["material"]

        average_train_loss = running_loss / max(batches_seen, 1)
        elapsed_minutes = (time.time() - epoch_start) / 60.0

        print(
            f"\nEpoch {epoch:02d} | "
            f"train focal {average_train_loss:.4f} | "
            f"skipped {skipped_updates}/{batches_seen} | "
            f"{elapsed_minutes:.1f} min"
        )
        print_class_distribution(
            validation_report["class_distribution"],
            "VALIDATION CLASS DISTRIBUTION",
        )
        print_head_report(validation_report)

        history.append(
            {
                "epoch": epoch,
                "train_loss": average_train_loss,
                "report": validation_report,
            }
        )

        if args.smoke_test:
            continue

        if material_ap["head"] > best_ap:
            best_ap = material_ap["head"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "error_head_state_dict": model.error_head.state_dict(),
                    "val_ap_material": best_ap,
                    "val_ap_density": material_ap["density"],
                    "delta": DELTA,
                    "args": vars(args),
                },
                output_dir / "error_head_best.pth",
            )
            print(
                f"Saved best ErrorHead: material AP {best_ap:.4f} "
                f"(baseline {material_ap['density']:.4f})"
            )
        else:
            epochs_without_improvement += 1

        with open(
            output_dir / "error_head_history.json",
            "w",
            encoding="utf-8",
        ) as history_file:
            json.dump(history, history_file, indent=2)

        if epochs_without_improvement >= args.patience:
            print("Early stopping.")
            break

    if args.smoke_test:
        print("SMOKE TEST COMPLETED. No checkpoint was saved.")
        return

    best_checkpoint_path = output_dir / "error_head_best.pth"
    if not best_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best ErrorHead checkpoint was not created: {best_checkpoint_path}"
        )

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.error_head.load_state_dict(
        best_checkpoint["error_head_state_dict"]
    )

    print("\nFinal evaluation of the best ErrorHead on the FULL validation set")
    full_validation_loader = build_validation_loader(args, 0)
    final_report = report_validation(
        model,
        full_validation_loader,
        device,
        amp,
    )

    with open(
        output_dir / "error_head_full_validation.json",
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(final_report, report_file, indent=2)


if __name__ == "__main__":
    main()
