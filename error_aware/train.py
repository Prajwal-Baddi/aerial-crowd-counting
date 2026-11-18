import argparse

import json

import random

import time

from pathlib import Path



import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader



from data import ErrorAwareCrowdDataset

from model import CoarseCounter





def parse_args():

    parser = argparse.ArgumentParser(

        description="Stage 1: Train the error-aware coarse crowd counter."

    )



    parser.add_argument(

        "--data-dir",

        type=str,

        default="error_aware/processed",

    )



    parser.add_argument(

        "--train-manifest",

        type=str,

        default="train_balanced.jsonl",

    )



    parser.add_argument(

        "--output-dir",

        type=str,

        default="error_aware/checkpoints",

    )



    parser.add_argument("--epochs", type=int, default=15)

    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--num-workers", type=int, default=0)



    parser.add_argument("--backbone-lr", type=float, default=1e-5)

    parser.add_argument("--head-lr", type=float, default=1e-4)

    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--count-weight", type=float, default=0.01)



    parser.add_argument("--seed", type=int, default=42)



    parser.add_argument(

        "--unfreeze-early",

        action="store_true",

        help="Train all VGG16 blocks instead of freezing the first two.",

    )



    parser.add_argument(

        "--smoke-test",

        action="store_true",

    )



    parser.add_argument(

        "--log-interval",

        type=int,

        default=100,

    )



    # Diagnostic limits. None means no limit.
    parser.add_argument(

        "--max-train-batches",

        type=int,

        default=None,

        help="Limit training batches for a diagnostic run.",

    )



    parser.add_argument(

        "--max-validation-batches",

        type=int,

        default=None,

        help="Limit validation batches for a diagnostic run.",

    )



    return parser.parse_args()





def set_seed(seed):

    random.seed(seed)

    torch.manual_seed(seed)



    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)





def create_optimizer(model, args):

    """

    Separate trainable VGG16 parameters from newly added layers.

    Frozen parameters are excluded automatically.

    """



    backbone_params = []

    head_params = []



    for name, parameter in model.named_parameters():



        if not parameter.requires_grad:

            continue



        if "backbone" in name:

            backbone_params.append(parameter)

        else:

            head_params.append(parameter)



    if not backbone_params:

        raise RuntimeError(

            "No trainable backbone parameters found."

        )



    if not head_params:

        raise RuntimeError(

            "No trainable new-layer parameters found."

        )



    print(

        f"Trainable backbone tensors: {len(backbone_params)}"

    )



    print(

        f"Trainable new-layer tensors: {len(head_params)}"

    )



    optimizer = torch.optim.AdamW(

        [

            {

                "params": backbone_params,

                "lr": args.backbone_lr,

            },

            {

                "params": head_params,

                "lr": args.head_lr,

            },

        ],

        weight_decay=args.weight_decay,

    )



    return optimizer





def create_loader(

    manifest_path,

    batch_size,

    num_workers,

    shuffle,

):

    dataset = ErrorAwareCrowdDataset(manifest_path)



    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available(),

        drop_last=False,

        persistent_workers=(num_workers > 0),

    )



    return dataset, loader





def calculate_losses(

    pred_density,

    target_density,

    target_counts,

    count_weight,

):

    """

    Density maps are scaled by 10 for the density MSE term.

    Count predictions remain in their original units.

    """



    pred_density = pred_density.float()

    target_density = target_density.float()

    target_counts = target_counts.float().view(-1)



    density_loss = F.mse_loss(

        pred_density * 10.0,

        target_density * 10.0,

    )



    predicted_counts = pred_density.sum(

        dim=(1, 2, 3)

    )



    count_loss = F.l1_loss(

        predicted_counts,

        target_counts,

    )



    total_loss = (

        density_loss

        + count_weight * count_loss

    )



    return (

        total_loss,

        density_loss,

        count_loss,

        predicted_counts,

    )





def get_activation_statistics(model, context):

    """

    Inspect density-head activations without running the backbone again.



    The model forward pass uses mixed precision, so context may be FP16.

    Convert it to FP32 before passing it through the FP32 density head.

    """

    with torch.no_grad():

        context = context.detach().float()



        hidden = model.density_head[0](context)

        hidden = model.density_head[1](hidden)



        pre_activation = model.density_head[-2](hidden)



        positive_percentage = (

            (pre_activation > 0).float().mean().item() * 100.0

        )



        activation_min = pre_activation.min().item()

        activation_max = pre_activation.max().item()



        final_bias = model.density_head[-2].bias.mean().item()



    return {

        "positive_percentage": positive_percentage,

        "activation_min": activation_min,

        "activation_max": activation_max,

        "final_bias": final_bias,

    }



def run_training_epoch(

    model,

    loader,

    optimizer,

    scaler,

    device,

    args,

    epoch,

    max_batches=None,

):

    model.train()



    total_loss_sum = 0.0

    density_loss_sum = 0.0

    count_loss_sum = 0.0



    batch_count = 0

    start_time = time.time()



    for batch_index, (images, densities, counts) in enumerate(loader):



        if max_batches is not None and batch_index >= max_batches:

            break



        images = images.to(

            device,

            non_blocking=True,

        )



        densities = densities.to(

            device,

            non_blocking=True,

        )



        counts = counts.to(

            device,

            non_blocking=True,

        )



        optimizer.zero_grad(set_to_none=True)



        with torch.autocast(

            device_type=device.type,

            dtype=torch.float16,

            enabled=(device.type == "cuda"),

        ):

            outputs = model(images)

            predicted_density = outputs["density"]



        (

            loss,

            density_loss,

            count_loss,

            predicted_counts,

        ) = calculate_losses(

            predicted_density,

            densities,

            counts,

            args.count_weight,

        )



        if not torch.isfinite(loss):

            raise RuntimeError(

                f"Non-finite loss at epoch {epoch}, "

                f"batch {batch_index}."

            )



        # Collect activation statistics before updating parameters.
        activation_stats = None



        if batch_index % args.log_interval == 0:

            activation_stats = get_activation_statistics(

                model,

                outputs["context"],

            )



        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()



        total_loss_sum += loss.item()

        density_loss_sum += density_loss.item()

        count_loss_sum += count_loss.item()



        batch_count += 1



        if activation_stats is not None:



            predicted_count_value = (

                predicted_counts.mean().item()

            )



            print(

                f"Epoch {epoch:02d} | "

                f"Batch {batch_index:05d}/{len(loader):05d} | "

                f"Loss {loss.item():.6f} | "

                f"Density {density_loss.item():.6f} | "

                f"Count {count_loss.item():.4f} | "

                f"Predicted count {predicted_count_value:.8f} | "

                f"Positive activations "

                f"{activation_stats['positive_percentage']:.2f}% | "

                f"PreAct min {activation_stats['activation_min']:.8e} | "

                f"PreAct max {activation_stats['activation_max']:.8e} | "

                f"Bias {activation_stats['final_bias']:.8e}"

            )



    if batch_count == 0:

        raise RuntimeError(

            "No training batches were processed."

        )



    elapsed = time.time() - start_time



    return {

        "loss": total_loss_sum / batch_count,

        "density_loss": density_loss_sum / batch_count,

        "count_loss": count_loss_sum / batch_count,

        "seconds": elapsed,

        "batches": batch_count,

    }





@torch.no_grad()

def evaluate(

    model,

    loader,

    device,

    max_batches=None,

):

    model.eval()



    absolute_errors = []

    squared_errors = []



    empty_absolute_errors = []

    nonempty_absolute_errors = []



    processed = 0



    for batch_index, (images, densities, counts) in enumerate(loader):



        if max_batches is not None and batch_index >= max_batches:

            break



        images = images.to(

            device,

            non_blocking=True,

        )



        counts = counts.to(

            device,

            non_blocking=True,

        ).float().view(-1)



        with torch.autocast(

            device_type=device.type,

            dtype=torch.float16,

            enabled=(device.type == "cuda"),

        ):

            outputs = model(images)

            predicted_density = outputs["density"]



        predicted_counts = predicted_density.float().sum(

            dim=(1, 2, 3)

        )



        errors = (predicted_counts - counts).abs()



        absolute_errors.extend(

            errors.cpu().tolist()

        )



        squared_errors.extend(

            (errors ** 2).cpu().tolist()

        )



        empty_mask = counts == 0

        nonempty_mask = counts > 0



        if empty_mask.any():

            empty_absolute_errors.extend(

                errors[empty_mask].cpu().tolist()

            )



        if nonempty_mask.any():

            nonempty_absolute_errors.extend(

                errors[nonempty_mask].cpu().tolist()

            )



        processed += images.size(0)



    if processed == 0:

        raise RuntimeError(

            "No validation samples were processed."

        )



    mae = (

        sum(absolute_errors)

        / len(absolute_errors)

    )



    rmse = (

        sum(squared_errors)

        / len(squared_errors)

    ) ** 0.5



    empty_mae = (

        sum(empty_absolute_errors)

        / len(empty_absolute_errors)

        if empty_absolute_errors

        else None

    )



    nonempty_mae = (

        sum(nonempty_absolute_errors)

        / len(nonempty_absolute_errors)

        if nonempty_absolute_errors

        else None

    )



    return {

        "mae": mae,

        "rmse": rmse,

        "empty_mae": empty_mae,

        "nonempty_mae": nonempty_mae,

        "samples": processed,

    }





def save_checkpoint(

    path,

    model,

    optimizer,

    scheduler,

    scaler,

    epoch,

    validation_metrics,

    args,

):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "scaler_state_dict": scaler.state_dict(),

        "validation_metrics": validation_metrics,

        "config": vars(args),

    }



    torch.save(checkpoint, path)





def main():

    args = parse_args()



    if args.max_train_batches is not None and args.max_train_batches <= 0:

        raise ValueError("--max-train-batches must be positive.")



    if args.max_validation_batches is not None and args.max_validation_batches <= 0:

        raise ValueError("--max-validation-batches must be positive.")



    set_seed(args.seed)



    device = torch.device(

        "cuda" if torch.cuda.is_available() else "cpu"

    )



    print("=" * 65)

    print("ERROR-AWARE CROWD COUNTING — STAGE 1")

    print("=" * 65)



    print(f"Device: {device}")



    if device.type == "cuda":

        print(

            f"GPU: {torch.cuda.get_device_name(0)}"

        )



        torch.cuda.reset_peak_memory_stats()



    data_dir = Path(args.data_dir)

    output_dir = Path(args.output_dir)



    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    train_manifest = data_dir / args.train_manifest

    validation_manifest = data_dir / "validation.jsonl"



    if not train_manifest.exists():

        raise FileNotFoundError(

            f"Training manifest not found: {train_manifest}"

        )



    if not validation_manifest.exists():

        raise FileNotFoundError(

            f"Validation manifest not found: {validation_manifest}"

        )



    print(f"Training manifest: {train_manifest}")

    print(f"Validation manifest: {validation_manifest}")



    train_dataset, train_loader = create_loader(

        train_manifest,

        args.batch_size,

        args.num_workers,

        shuffle=True,

    )



    validation_dataset, validation_loader = create_loader(

        validation_manifest,

        args.batch_size,

        args.num_workers,

        shuffle=False,

    )



    print(f"Training crops: {len(train_dataset)}")

    print(f"Validation crops: {len(validation_dataset)}")

    print(f"Batch size: {args.batch_size}")

    print(f"Epochs: {args.epochs}")



    freeze_early = not args.unfreeze_early



    print(f"Freeze early VGG16 blocks: {freeze_early}")



    diagnostic_mode = (

        args.max_train_batches is not None

        or args.max_validation_batches is not None

    )



    if diagnostic_mode:

        print("\nLIMITED-BATCH DIAGNOSTIC MODE")

        print("Partial runs will not save checkpoints.")



    print("\nLoading pretrained coarse counter...")



    model = CoarseCounter(

        pretrained=True,

        freeze_early=freeze_early,

    ).to(device)



    optimizer = create_optimizer(

        model,

        args,

    )



    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=args.epochs,

    )



    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=(device.type == "cuda"),

    )



    best_mae = float("inf")



    if args.smoke_test:

        print(

            "\nSMOKE TEST MODE: "

            "one training batch and one validation batch."

        )



        train_batch_limit = 1

        validation_batch_limit = 1

        epochs_to_run = 1



    else:

        train_batch_limit = args.max_train_batches

        validation_batch_limit = args.max_validation_batches

        epochs_to_run = 1 if diagnostic_mode else args.epochs



    history = []



    for epoch in range(1, epochs_to_run + 1):



        print("\n" + "-" * 65)

        print(f"Epoch {epoch}/{epochs_to_run}")

        print("-" * 65)



        train_metrics = run_training_epoch(

            model=model,

            loader=train_loader,

            optimizer=optimizer,

            scaler=scaler,

            device=device,

            args=args,

            epoch=epoch,

            max_batches=train_batch_limit,

        )



        validation_metrics = evaluate(

            model=model,

            loader=validation_loader,

            device=device,

            max_batches=validation_batch_limit,

        )



        if not diagnostic_mode and not args.smoke_test:

            scheduler.step()



        print("\nEpoch results:")



        print(

            f"Train loss:       {train_metrics['loss']:.6f}"

        )



        print(

            f"Density loss:     {train_metrics['density_loss']:.6f}"

        )



        print(

            f"Count loss:       {train_metrics['count_loss']:.4f}"

        )



        print(

            f"Validation MAE:   {validation_metrics['mae']:.4f}"

        )



        print(

            f"Validation RMSE:  {validation_metrics['rmse']:.4f}"

        )



        if validation_metrics["empty_mae"] is not None:

            print(

                f"Empty-crop MAE:   "

                f"{validation_metrics['empty_mae']:.4f}"

            )



        if validation_metrics["nonempty_mae"] is not None:

            print(

                f"Nonempty-crop MAE: "

                f"{validation_metrics['nonempty_mae']:.4f}"

            )



        print(

            f"Training time:    {train_metrics['seconds']:.2f}s"

        )



        if device.type == "cuda":

            peak_memory = (

                torch.cuda.max_memory_allocated()

                / (1024 ** 3)

            )



            print(

                f"Peak GPU memory:  {peak_memory:.3f} GB"

            )



        epoch_record = {

            "epoch": epoch,

            "train": train_metrics,

            "validation": validation_metrics,

        }



        history.append(epoch_record)



        # Do not save partial diagnostic runs as complete epochs.
        if not diagnostic_mode and not args.smoke_test:



            save_checkpoint(

                output_dir / "coarse_last.pth",

                model,

                optimizer,

                scheduler,

                scaler,

                epoch,

                validation_metrics,

                args,

            )



            if validation_metrics["mae"] < best_mae:



                best_mae = validation_metrics["mae"]



                save_checkpoint(

                    output_dir / "coarse_best.pth",

                    model,

                    optimizer,

                    scheduler,

                    scaler,

                    epoch,

                    validation_metrics,

                    args,

                )



                print(

                    f"New best checkpoint saved. "

                    f"Validation MAE: {best_mae:.4f}"

                )



            history_path = (

                output_dir / "coarse_training_history.json"

            )



            with open(

                history_path,

                "w",

                encoding="utf-8",

            ) as file:

                json.dump(

                    history,

                    file,

                    indent=2,

                )



    print("\n" + "=" * 65)



    if args.smoke_test:

        print("SMOKE TEST COMPLETED.")

        print("No training checkpoint was saved.")



    elif diagnostic_mode:

        print("LIMITED-BATCH DIAGNOSTIC COMPLETED.")

        print("No checkpoint was saved.")



    else:

        print("STAGE 1 TRAINING COMPLETED.")

        print(f"Best validation MAE: {best_mae:.4f}")

        print(

            f"Best checkpoint: {output_dir / 'coarse_best.pth'}"

        )



    print("=" * 65)





if __name__ == "__main__":

    main()
