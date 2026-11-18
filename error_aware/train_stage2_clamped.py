"""

Stage 2 training for the Error-Aware Selective Refinement architecture.

Run from the project root:

    python error_aware/train_stage2.py --smoke-test --max-train-batches 2 --log-interval 1

Full training:

    python error_aware/train_stage2.py --epochs 30 --log-interval 100

The Stage-1 coarse counter is loaded from coarse_best.pth and remains frozen.

Only ErrorHead and RefineNet parameters are optimized.

"""

import argparse

import json

import random

import time

from pathlib import Path

import torch

import torch.nn.functional as F

from data import get_loader

from model import ESRNet, cell_sum, select_topk, mask_to_pixels

def parse_args():

    parser = argparse.ArgumentParser(

        description="Stage-2 experiment: train using the inference-time clamped density map."

    )

    parser.add_argument("--data-dir", default="error_aware/processed")

    parser.add_argument(

        "--checkpoint",

        default="error_aware/checkpoints/coarse_best.pth",

    )

    parser.add_argument(

        "--output-dir",

        default="error_aware/checkpoints_stage2_clamped",

    )

    parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log-interval", type=int, default=100)

    parser.add_argument("--max-train-batches", type=int, default=None)

    parser.add_argument("--max-validation-batches", type=int, default=None)

    parser.add_argument("--smoke-test", action="store_true")

    parser.add_argument("--refine-width", type=int, default=96)
    parser.add_argument("--nonnegative-weight", type=float, default=0.1,
                        help="Weight for the selected-region negative-density penalty.")

    return parser.parse_args()

def set_seed(seed):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

def load_coarse_checkpoint(model, checkpoint_path, device):

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device,

        weights_only=False,

    )

    state_dict = checkpoint.get(

        "model_state_dict",

        checkpoint.get("model", checkpoint),

    )

    model.coarse.load_state_dict(state_dict, strict=True)

    for parameter in model.coarse.parameters():

        parameter.requires_grad_(False)

    model.coarse.eval()

    # Verify that the entire coarse model is frozen.

    assert all(

        not parameter.requires_grad

        for parameter in model.coarse.parameters()

    )

    print(f"Loaded Stage-1 checkpoint strictly: {checkpoint_path}")

    print("Stage-1 coarse counter frozen.")

def true_cell_error(coarse_density, target_density):

    """

    Generate the Error Head's supervision target.

    Each error-map cell corresponds to a 4x4 region in the stride-8

    density map, i.e. a 32x32 region in the input crop.

    Target error = absolute difference between predicted and true

    pedestrian counts in each cell.

    """

    predicted_cell_counts = cell_sum(coarse_density.detach().float())

    target_cell_counts = cell_sum(target_density.float())

    return (predicted_cell_counts - target_cell_counts).abs()

def ranking_loss(

    predicted_error,

    target_error,

    pairs=1024,

    margin_ignore=0.05,

):

    """

    Pairwise logistic ranking loss.

    If cell i has a larger true error than cell j, its predicted error

    score should also be larger. Pairs with nearly equal true errors

    are ignored.

    """

    predicted = predicted_error.flatten(1)

    target = target_error.flatten(1)

    batch_size, number_of_cells = predicted.shape

    if number_of_cells < 2:

        return predicted.sum() * 0.0

    # Sample pairs rather than constructing a full N x N matrix.

    index_i = torch.randint(

        0,

        number_of_cells,

        (batch_size, pairs),

        device=predicted.device,

    )

    index_j = torch.randint(

        0,

        number_of_cells,

        (batch_size, pairs),

        device=predicted.device,

    )

    target_i = target.gather(1, index_i)

    target_j = target.gather(1, index_j)

    difference = target_i - target_j

    valid_pairs = difference.abs() >= margin_ignore

    if not valid_pairs.any():

        return predicted.sum() * 0.0

    direction = difference.sign()

    predicted_i = predicted.gather(1, index_i)

    predicted_j = predicted.gather(1, index_j)

    pair_logits = direction * (predicted_i - predicted_j)

    return F.softplus(-pair_logits[valid_pairs]).mean()

def choose_training_mask(

    predicted_error,

    target_error,

    budget,

    teacher_probability,

):

    """

    Scheduled sampling:

      - with probability teacher_probability, select using true errors;

      - otherwise, select using the Error Head's predictions.

    """

    if random.random() < teacher_probability:

        selection_scores = target_error

    else:

        selection_scores = predicted_error.detach()

    return select_topk(selection_scores, budget)

def calculate_stage2_losses(

    predicted_error,

    target_error,

    refined_density,

    raw_refined_density,

    target_density,

    mask,
    nonnegative_weight=0.1,

):

    """

    Implements the Stage-2 objective from the project specification:

      L_error  = SmoothL1(ErrorHead prediction, true cell error)

      L_rank   = pairwise ranking loss

      L_refine = selected-region density MSE

                 + selected-cell count L1

      L_global = full-crop count L1

      L_nonnegative = selected-region negative-density penalty

      L_total = L_error + 0.5*L_rank + L_refine + 0.1*L_global
                 + nonnegative_weight*L_nonnegative

    """

    predicted_error = predicted_error.float()

    target_error = target_error.float()

    # All task losses use the same nonnegative density map used at inference.
    refined_density = refined_density.float()
    raw_refined_density = raw_refined_density.float()

    target_density = target_density.float()

    mask = mask.float()

    # 1. Error prediction loss.

    error_loss = F.smooth_l1_loss(

        predicted_error,

        target_error,

    )

    # 2. Pairwise ranking loss.

    rank_loss = ranking_loss(

        predicted_error,

        target_error,

    )

    # Convert the 16x16 cell mask into a 64x64 density-map mask.

    pixel_mask = mask_to_pixels(mask, scale=4).float()

    selected_pixel_count = pixel_mask.sum().clamp_min(1.0)

    # 3a. Pixel-level refinement MSE, averaged over selected pixels only.

    squared_density_error = (

        (refined_density - target_density).square() * pixel_mask

    )

    selected_density_mse = (

        squared_density_error.sum() / selected_pixel_count

    )

    # 3b. Selected-cell count L1 loss.

    predicted_cell_counts = cell_sum(refined_density)

    target_cell_counts = cell_sum(target_density)

    selected_cell_count = mask.sum().clamp_min(1.0)

    selected_count_l1 = (

        (

            (predicted_cell_counts - target_cell_counts).abs()

            * mask

        ).sum()

        / selected_cell_count

    )

    # Penalize negative values in the raw pre-clamp map separately.
    # The density/count objectives above use the clamped inference output;
    # this auxiliary term supplies gradients for negative raw predictions.
    negative_density = F.relu(-raw_refined_density) * pixel_mask
    nonnegative_loss = negative_density.square().sum() / selected_pixel_count

    # L_refine combines the two selected-region objectives.

    refine_loss = selected_density_mse + selected_count_l1

    # 4. Global count loss: compare the complete crop counts.

    # This is deliberately NOT restricted to selected cells.

    predicted_global_count = refined_density.sum(dim=(1, 2, 3))

    target_global_count = target_density.sum(dim=(1, 2, 3))

    global_loss = F.l1_loss(

        predicted_global_count,

        target_global_count,

    )

    total_loss = (

        error_loss

        + 0.5 * rank_loss

        + refine_loss

        + 0.1 * global_loss
        + float(nonnegative_weight) * nonnegative_loss

    )

    metrics = {

        "total": total_loss,

        "error": error_loss,

        "rank": rank_loss,

        "refine": refine_loss,

        "selected_mse": selected_density_mse,

        "selected_count_l1": selected_count_l1,

        "global": global_loss,
        "nonnegative": nonnegative_loss,

    }

    return total_loss, metrics

def train_one_epoch(

    model,

    loader,

    optimizer,

    scaler,

    device,

    epoch,

    args,

):

    model.train()

    # model.train() recursively changes child modules, so explicitly

    # keep the frozen Stage-1 model in evaluation mode.

    model.coarse.eval()

    # Teacher forcing decays linearly from 0.8 to 0.2.

    progress = (epoch - 1) / max(args.epochs - 1, 1)

    teacher_probability = 0.8 - 0.6 * progress

    metric_names = (

        "total",

        "error",

        "rank",

        "refine",

        "selected_mse",

        "selected_count_l1",

        "global",
        "nonnegative",

    )

    totals = {name: 0.0 for name in metric_names}

    number_of_batches = 0
    successful_updates = 0

    start_time = time.time()

    for step, (images, target_density, _) in enumerate(loader, start=1):

        if (

            args.max_train_batches is not None

            and step > args.max_train_batches

        ):

            break

        images = images.to(device, non_blocking=True)

        target_density = target_density.to(

            device,

            non_blocking=True,

        )

        optimizer.zero_grad(set_to_none=True)

        amp_enabled = device.type == "cuda"

        with torch.autocast(

            device_type=device.type,

            dtype=torch.float16,

            enabled=amp_enabled,

        ):

            outputs = model.encode(images)

            coarse_density = outputs["density"]

            predicted_error = outputs["error_scores"]

            # Error targets are calculated in float32 for numerical stability.

            target_error = true_cell_error(

                coarse_density.float(),

                target_density.float(),

            )

            # Random refinement budget between 10% and 40%.

            budget = random.uniform(0.10, 0.40)

            mask = choose_training_mask(

                predicted_error.float(),

                target_error,

                budget,

                teacher_probability,

            )

            raw_refined_density = model.apply_refine(

                coarse_density,

                outputs["stride4_features"],

                mask,
                clamp_output=False,

            )

        # Match inference exactly: density/count losses see the nonnegative map.
        refined_density = raw_refined_density.float().clamp_min(0.0)

        # Compute losses outside autocast in float32.

        loss, metrics = calculate_stage2_losses(

            predicted_error=predicted_error,

            target_error=target_error,

            refined_density=refined_density,

            raw_refined_density=raw_refined_density,

            target_density=target_density,

            mask=mask,
            nonnegative_weight=args.nonnegative_weight,

        )

        if not torch.isfinite(loss):

            raise RuntimeError(

                f"Non-finite loss at epoch {epoch}, step {step}: "

                f"{loss.item()}"

            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(

            list(model.error_head.parameters())

            + list(model.refiner.parameters()),

            max_norm=5.0,

        )

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        if scale_after < scale_before:
            print(
                f"WARNING: AMP skipped optimizer update at "
                f"epoch {epoch}, step {step}. "
                f"Scale: {scale_before} -> {scale_after}"
            )
        else:
            successful_updates += 1
            if step == 1 or step % args.log_interval == 0:
                print(
                    f"Optimizer update accepted at epoch {epoch}, "
                    f"step {step}. Scale: {scale_before} -> {scale_after}"
                )

        for name in metric_names:

            totals[name] += metrics[name].detach().item()

        number_of_batches += 1

        if step % args.log_interval == 0 or step == 1:

            print(

                f"Epoch {epoch:02d} Step {step:04d} | "

                f"loss {metrics['total'].item():.6f} | "

                f"err {metrics['error'].item():.6f} | "

                f"rank {metrics['rank'].item():.6f} | "

                f"refine {metrics['refine'].item():.6f} | "

                f"selected_mse {metrics['selected_mse'].item():.6f} | "

                f"selected_count_l1 "

                f"{metrics['selected_count_l1'].item():.6f} | "

                f"global {metrics['global'].item():.6f} | "

                f"budget {budget:.2f} | "

                f"teacher {teacher_probability:.2f}"

            )

    elapsed_seconds = time.time() - start_time

    averages = {

        name: value / max(number_of_batches, 1)

        for name, value in totals.items()

    }

    return averages, elapsed_seconds, successful_updates

@torch.no_grad()

def validate(

    model,

    loader,

    device,

    max_batches=None,

):

    """

    Validate using the predicted Error Head ranking and a fixed 25% budget.

    Reports crop-level count MAE and RMSE.

    """

    model.eval()

    model.coarse.eval()

    absolute_error_sum = 0.0

    squared_error_sum = 0.0

    number_of_samples = 0

    for step, (images, target_density, _) in enumerate(loader, start=1):

        if max_batches is not None and step > max_batches:

            break

        images = images.to(device, non_blocking=True)

        target_density = target_density.to(

            device,

            non_blocking=True,

        )

        amp_enabled = device.type == "cuda"

        with torch.autocast(

            device_type=device.type,

            dtype=torch.float16,

            enabled=amp_enabled,

        ):

            outputs = model.encode(images)

            mask = select_topk(

                outputs["error_scores"].float(),

                0.25,

            )

            final_density = model.apply_refine(

                outputs["density"],

                outputs["stride4_features"],

                mask,

            )

        predicted_count = final_density.float().sum(dim=(1, 2, 3))

        true_count = target_density.float().sum(dim=(1, 2, 3))

        count_errors = predicted_count - true_count

        absolute_error_sum += count_errors.abs().sum().item()

        squared_error_sum += count_errors.square().sum().item()

        number_of_samples += images.size(0)

    mae = absolute_error_sum / max(number_of_samples, 1)

    rmse = (

        squared_error_sum / max(number_of_samples, 1)

    ) ** 0.5

    return {

        "mae": mae,

        "rmse": rmse,

        "samples": number_of_samples,

    }

def save_checkpoint(

    path,

    epoch,

    model,

    optimizer,

    scheduler,

    best_val_mae,

    args,

):

    torch.save(

        {

            "epoch": epoch,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "scheduler_state_dict": scheduler.state_dict(),

            "best_val_mae": best_val_mae,

            "args": vars(args),

        },

        path,

    )

def main():

    args = parse_args()

    set_seed(args.seed)

    device = torch.device(

        "cuda" if torch.cuda.is_available() else "cpu"

    )

    print("EXPERIMENT: inference-time clamped density training")
    print("Selected density/count losses use clamp_min(0); negative raw residuals use an auxiliary penalty.")
    print(f"Device: {device}")

    if device.type == "cuda":

        print(f"GPU: {torch.cuda.get_device_name(0)}")

        total_vram = (

            torch.cuda.get_device_properties(0).total_memory

            / (1024 ** 3)

        )

        print(f"VRAM: {total_vram:.2f} GB")

    output_dir = Path(args.output_dir)

    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    model = ESRNet(

        pretrained=False,

        freeze_early=True,

        refine_width=args.refine_width,

    ).to(device)

    load_coarse_checkpoint(

        model,

        args.checkpoint,

        device,

    )

    train_loader = get_loader(

        data_dir=args.data_dir,

        split="train",

        batch_size=args.batch_size,

        num_workers=args.num_workers,

    )

    val_loader = get_loader(

        data_dir=args.data_dir,

        split="validation",

        batch_size=1,

        shuffle=False,

        num_workers=args.num_workers,

    )

    # Only Stage-2 components are passed to the optimizer.

    trainable_parameters = (

        list(model.error_head.parameters())

        + list(model.refiner.parameters())

    )

    optimizer = torch.optim.AdamW(

        trainable_parameters,

        lr=args.lr,

        weight_decay=args.weight_decay,

    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=args.epochs,

    )

    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=(device.type == "cuda"),

    )

    history = []

    best_val_mae = float("inf")

    epochs_to_run = 1 if args.smoke_test else args.epochs

    for epoch in range(1, epochs_to_run + 1):

        train_metrics, elapsed_seconds, successful_updates = train_one_epoch(

            model=model,

            loader=train_loader,

            optimizer=optimizer,

            scaler=scaler,

            device=device,

            epoch=epoch,

            args=args,

        )

        validation_limit = (

            2 if args.smoke_test

            else args.max_validation_batches

        )

        validation_metrics = validate(

            model=model,

            loader=val_loader,

            device=device,

            max_batches=validation_limit,

        )

        if successful_updates > 0:
            scheduler.step()
        else:
            print(
                f"WARNING: No successful optimizer updates in epoch {epoch}; "
                "learning-rate scheduler was not advanced."
            )

        teacher_probability = (

            0.8

            - 0.6

            * ((epoch - 1) / max(args.epochs - 1, 1))

        )

        epoch_record = {

            "epoch": epoch,

            "train": train_metrics,

            "validation": validation_metrics,

            "seconds": elapsed_seconds,

            "teacher_probability": teacher_probability,

            "learning_rate": optimizer.param_groups[0]["lr"],

        }

        history.append(epoch_record)

        print(

            f"Epoch {epoch:02d} complete | "

            f"train loss {train_metrics['total']:.6f} | "

            f"val MAE {validation_metrics['mae']:.6f} | "

            f"val RMSE {validation_metrics['rmse']:.6f} | "

            f"samples {validation_metrics['samples']} | "

            f"time {elapsed_seconds / 60:.2f} min"

        )

        if validation_metrics["mae"] < best_val_mae:

            best_val_mae = validation_metrics["mae"]

            best_path = output_dir / "stage2_best.pth"

            save_checkpoint(

                path=best_path,

                epoch=epoch,

                model=model,

                optimizer=optimizer,

                scheduler=scheduler,

                best_val_mae=best_val_mae,

                args=args,

            )

            print(f"Saved best checkpoint: {best_path}")

        last_path = output_dir / "stage2_last.pth"

        save_checkpoint(

            path=last_path,

            epoch=epoch,

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,

            best_val_mae=best_val_mae,

            args=args,

        )

        history_path = output_dir / "stage2_training_history.json"

        with history_path.open("w", encoding="utf-8") as file:

            json.dump(history, file, indent=2)

    print(

        "Stage-2 finished. "

        f"Best validation MAE: {best_val_mae:.6f}"

    )

if __name__ == "__main__":

    main()
