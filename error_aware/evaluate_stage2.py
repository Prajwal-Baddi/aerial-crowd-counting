import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch

from data import get_loader
from model import ESRNet, select_topk


def parse_args():
    p = argparse.ArgumentParser(
        description="Test-set budget sweep for Error-Aware Selective Refinement."
    )
    p.add_argument("--data-dir", default="error_aware/processed")
    p.add_argument("--checkpoint", default="error_aware/checkpoints/stage2_best.pth")
    p.add_argument("--output-dir", default="error_aware/evaluation")
    p.add_argument("--budgets", nargs="+", type=float, default=[0.10, 0.25, 0.40])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-test-batches", type=int, default=None)
    p.add_argument("--refine-width", type=int, default=96)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    print(
        f"Loaded Stage-2 checkpoint: {path} | "
        f"epoch={checkpoint.get('epoch', 'unknown')} | "
        f"best_val_mae={checkpoint.get('best_val_mae', 'unknown')}"
    )


def new_store():
    return {
        "abs_sum": 0.0, "sq_sum": 0.0, "n": 0,
        "empty_abs_sum": 0.0, "empty_n": 0,
        "nonempty_abs_sum": 0.0, "nonempty_n": 0,
        "seconds": 0.0,
    }


def add_metrics(store, prediction, target, elapsed):
    errors = prediction.float().view(-1) - target.float().view(-1)
    absolute = errors.abs()
    target = target.float().view(-1)

    store["abs_sum"] += absolute.sum().item()
    store["sq_sum"] += errors.square().sum().item()
    store["n"] += errors.numel()
    store["seconds"] += elapsed

    empty = target == 0
    nonempty = target > 0
    if empty.any():
        store["empty_abs_sum"] += absolute[empty].sum().item()
        store["empty_n"] += int(empty.sum().item())
    if nonempty.any():
        store["nonempty_abs_sum"] += absolute[nonempty].sum().item()
        store["nonempty_n"] += int(nonempty.sum().item())


def finish_metrics(s):
    n = max(s["n"], 1)
    return {
        "mae": s["abs_sum"] / n,
        "rmse": (s["sq_sum"] / n) ** 0.5,
        "empty_mae": s["empty_abs_sum"] / s["empty_n"] if s["empty_n"] else None,
        "nonempty_mae": (
            s["nonempty_abs_sum"] / s["nonempty_n"] if s["nonempty_n"] else None
        ),
        "samples": s["n"],
        "empty_samples": s["empty_n"],
        "nonempty_samples": s["nonempty_n"],
        "mean_inference_ms_per_crop": s["seconds"] / n * 1000.0,
    }


@torch.inference_mode()
def evaluate(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    for budget in args.budgets:
        if not 0.0 < budget < 1.0:
            raise ValueError(f"Budget must be between 0 and 1, got {budget}")

    model = ESRNet(
        pretrained=False, freeze_early=True, refine_width=args.refine_width
    ).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    model.coarse.eval()

    # Use the complete, unbalanced test split. Never use train_balanced.jsonl here.
    loader = get_loader(
        data_dir=args.data_dir,
        split="test",
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    stores = {"coarse_only": new_store()}
    for budget in args.budgets:
        label = f"{int(round(budget * 100)):02d}%"
        stores[f"error_aware_{label}"] = new_store()
        stores[f"random_{label}"] = new_store()

    total = len(loader)
    limit = min(args.max_test_batches, total) if args.max_test_batches else total
    print(f"Test crops: {total}; evaluating: {limit}")
    print(f"Budgets: {[f'{b:.0%}' for b in args.budgets]}")

    for batch_index, (images, target_density, _counts) in enumerate(loader, start=1):
        if args.max_test_batches and batch_index > args.max_test_batches:
            break

        images = images.to(device, non_blocking=True)
        target_density = target_density.to(device, non_blocking=True).float()

        # One shared coarse forward pass per crop for a fair strategy comparison.
        sync(device)
        start = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            outputs = model.encode(images)
        sync(device)
        encode_seconds = time.perf_counter() - start

        coarse_density = outputs["density"]
        error_scores = outputs["error_scores"].float()
        stride4_features = outputs["stride4_features"]
        true_count = target_density.sum(dim=(1, 2, 3))
        coarse_count = coarse_density.float().sum(dim=(1, 2, 3))

        add_metrics(stores["coarse_only"], coarse_count, true_count, encode_seconds)

        for budget in args.budgets:
            label = f"{int(round(budget * 100)):02d}%"

            # Highest predicted-error cells.
            error_mask = select_topk(error_scores, budget)
            sync(device)
            start = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                error_density = model.apply_refine(
                    coarse_density, stride4_features, error_mask
                )
            sync(device)
            refine_seconds = time.perf_counter() - start
            error_count = error_density.float().sum(dim=(1, 2, 3))
            add_metrics(
                stores[f"error_aware_{label}"],
                error_count,
                true_count,
                encode_seconds + refine_seconds,
            )

            # Reproducible random-cell baseline, using the same refiner.
            generator = torch.Generator(device=device)
            generator.manual_seed(
                args.seed + batch_index * 1009 + int(round(budget * 10000))
            )
            random_scores = torch.rand(
                error_scores.shape, device=device, generator=generator
            )
            random_mask = select_topk(random_scores, budget)

            sync(device)
            start = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                random_density = model.apply_refine(
                    coarse_density, stride4_features, random_mask
                )
            sync(device)
            refine_seconds = time.perf_counter() - start
            random_count = random_density.float().sum(dim=(1, 2, 3))
            add_metrics(
                stores[f"random_{label}"],
                random_count,
                true_count,
                encode_seconds + refine_seconds,
            )

        if batch_index == 1 or batch_index % 250 == 0:
            print(f"Processed {batch_index}/{limit} crops")

    results = {name: finish_metrics(value) for name, value in stores.items()}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint": args.checkpoint,
        "test_split": "full" if args.max_test_batches is None else "partial",
        "seed": args.seed,
        "budgets": args.budgets,
        "results": results,
    }
    json_path = output_dir / "stage2_test_metrics.json"
    csv_path = output_dir / "stage2_test_metrics.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    columns = [
        "method", "mae", "rmse", "empty_mae", "nonempty_mae",
        "samples", "empty_samples", "nonempty_samples",
        "mean_inference_ms_per_crop",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for method, metrics in results.items():
            writer.writerow({"method": method, **metrics})

    print("\nTEST RESULTS (crop-level)")
    print("-" * 100)
    print(
        f"{'Method':<22} {'MAE':>10} {'RMSE':>10} "
        f"{'Empty MAE':>12} {'Nonempty MAE':>14} {'ms/crop':>12}"
    )
    print("-" * 100)
    for method, m in results.items():
        empty = f"{m['empty_mae']:.4f}" if m["empty_mae"] is not None else "N/A"
        nonempty = (
            f"{m['nonempty_mae']:.4f}" if m["nonempty_mae"] is not None else "N/A"
        )
        print(
            f"{method:<22} {m['mae']:>10.4f} {m['rmse']:>10.4f} "
            f"{empty:>12} {nonempty:>14} "
            f"{m['mean_inference_ms_per_crop']:>12.2f}"
        )
    print("-" * 100)
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    if args.max_test_batches is not None:
        print("WARNING: Partial test run; do not report these as final test metrics.")


if __name__ == "__main__":
    evaluate(parse_args())
