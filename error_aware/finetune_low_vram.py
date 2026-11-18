"""Fine-tune the deployed counter on a 4 GB GPU without overwriting its weights."""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aerial-crowd-backend"))
from backend.inference import CrowdCounterService
from error_aware.data import ErrorAwareCrowdDataset


class ScaleAugmentedDataset(ErrorAwareCrowdDataset):
    """Keep native crops and add larger pedestrian scales, with transformed labels."""
    zoom_probability = 0.6

    def __getitem__(self, index):
        record = self.records[index]
        x, y = record["crop_x"], record["crop_y"]
        with Image.open(self.dataset_root / record["image"]) as source:
            image = source.convert("RGB").crop((x, y, x + 512, y + 512))
        points = np.asarray(record["points"], dtype=np.float32).reshape(-1, 2)
        scale = random.choice((1.5, 2.0, 3.0)) if random.random() < self.zoom_probability else 1.0
        side = round(512 / scale)
        left, top = random.randint(0, 512 - side), random.randint(0, 512 - side)
        # Sample near a labeled person often enough to retain sparse positive crops.
        if len(points) and scale > 1 and random.random() < 0.75:
            px, py = points[random.randrange(len(points))]
            left = round(np.clip(px - random.uniform(0.25, 0.75) * side, 0, 512 - side))
            top = round(np.clip(py - random.uniform(0.25, 0.75) * side, 0, 512 - side))
        inside = (points[:, 0] >= left) & (points[:, 0] < left + side)
        inside &= (points[:, 1] >= top) & (points[:, 1] < top + side)
        points = (points[inside] - np.array([left, top])) * (512 / side)
        image = image.crop((left, top, left + side, top + side)).resize((512, 512), Image.Resampling.BILINEAR)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            points[:, 0] = np.clip(512 - points[:, 0], 0, 512 - 1e-4)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return (torch.from_numpy(array.transpose(2, 0, 1).copy()),
                self._generate_density_map(points), torch.tensor(float(len(points))))


def image_batches(dataset, records, batch_size=4):
    """Decode each source image once while keeping only one batch in memory."""
    cached_path, cached_image = None, None
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        inputs = []
        for record in chunk:
            path = dataset.dataset_root / record["image"]
            if path != cached_path:
                with Image.open(path) as image:
                    cached_image = image.convert("RGB")
                cached_path = path
            x, y = record["crop_x"], record["crop_y"]
            array = np.asarray(cached_image.crop((x, y, x + 512, y + 512)), dtype=np.float32) / 255.0
            inputs.append(torch.from_numpy(array.transpose(2, 0, 1).copy()))
        yield torch.stack(inputs), chunk


@torch.inference_mode()
def evaluate(model, dataset, device, limit=None, batch_size=4):
    model.eval()
    predicted, targets = [], []
    image_pred, image_gt = defaultdict(float), defaultdict(float)
    records = dataset.records[:limit] if limit is not None else dataset.records
    for tensor, chunk in image_batches(dataset, records, batch_size):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            density = model(tensor.to(device))["density"].float()
        counts = density.sum((1, 2, 3)).cpu().tolist()
        for record, count in zip(chunk, counts):
            predicted.append(count)
            targets.append(record["count"])
            image_pred[record["image"]] += count
            image_gt[record["image"]] += record["count"]
        if (len(predicted) - len(chunk)) % 1000 == 0:
            print(f"{dataset.manifest_path.stem}: {len(predicted)}/{len(records)}", flush=True)
    error = np.array(predicted) - np.array(targets)
    sparse = (np.array(targets) > 0) & (np.array(targets) <= 5)
    sparse_images = [k for k in image_gt if 0 < image_gt[k] <= 5]
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
            "sparse_mae": float(np.abs(error[sparse]).mean()),
            "sparse_image_mae": float(np.mean([abs(image_pred[k] - image_gt[k]) for k in sparse_images])) if sparse_images else None,
            "image_mae": float(np.mean([abs(image_pred[k] - image_gt[k]) for k in image_gt])),
            "crops": len(targets)}


@torch.inference_mode()
def rebalance_sparse_training(model, dataset, device, smoke=False):
    """Mine labeled empty crops only from the original Stage-1 training images."""
    allowed_images = {r["image"] for r in dataset.records}
    full = ErrorAwareCrowdDataset(ROOT / "error_aware/processed/train.jsonl")
    negatives = [r for r in full.records if r["image"] in allowed_images and r["count"] == 0]
    rng = random.Random(42)
    pool = rng.sample(negatives, min(32 if smoke else 2048, len(negatives)))
    pool.sort(key=lambda r: (r["image"], r["crop_y"], r["crop_x"]))
    scored = []
    model.eval()
    for tensor, chunk in image_batches(dataset, pool):
        with torch.autocast(device.type, dtype=torch.float16):
            counts = model(tensor.to(device))["density"].float().sum((1, 2, 3)).cpu().tolist()
        scored.extend(zip(counts, chunk))
        if len(scored) % 400 == 0:
            print(f"Mining training backgrounds: {len(scored)}/{len(pool)}", flush=True)
    scored.sort(key=lambda pair: pair[0], reverse=True)
    hard = [record for _, record in scored[:min(512, len(scored))]]
    key = lambda r: (r["image"], r["crop_x"], r["crop_y"])
    hard_keys = {key(r) for r in hard}
    remaining = [r for r in negatives if key(r) not in hard_keys]
    random_negatives = rng.sample(remaining, min(max(0, 2048 - len(hard)), len(remaining)))
    sparse = [r for r in dataset.records if 0 < r["count"] <= 5]
    dense = [r for r in dataset.records if r["count"] > 5]
    positives = rng.sample(sparse, min(1536, len(sparse))) + rng.sample(dense, min(512, len(dense)))
    dataset.records = positives + hard + random_negatives
    rng.shuffle(dataset.records)
    summary = {"positive_crops": len(positives), "hard_background_crops": len(hard),
               "random_background_crops": len(random_negatives), "mining_pool": len(pool),
               "largest_background_prediction": scored[0][0] if scored else 0.0,
               "images": len({r["image"] for r in dataset.records})}
    print("SPARSE_TRAINING", summary, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "error_aware/checkpoints_stage1_holdout80/coarse_best.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "error_aware/checkpoints_low_vram")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--count-weight", type=float, default=0.05)
    parser.add_argument("--empty-weight", type=float, default=1.0)
    parser.add_argument("--zoom-probability", type=float, default=0.6)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--sparse-rebalance", action="store_true")
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.accumulate, args.epochs) < 1:
        parser.error("batch size, accumulation, and epochs must be positive")
    if not 0 <= args.zoom_probability <= 1 or args.empty_weight <= 0 or args.head_lr <= 0:
        parser.error("zoom probability must be in [0,1]; loss weight and learning rate must be positive")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU training profile")
    # Keep CUDA allocations below 80% of physical VRAM, leaving room for the desktop/API.
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.cuda.reset_peak_memory_stats()
    service = CrowdCounterService(args.checkpoint, device_name="cuda")
    model, device = service.model, service.device
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU: {properties.name}; VRAM: {properties.total_memory / 2**30:.2f} GiB", flush=True)
    print(f"Microbatch {args.batch_size}; accumulation {args.accumulate}; AMP FP16; effective batch {args.batch_size * args.accumulate}", flush=True)
    training = ScaleAugmentedDataset(ROOT / "error_aware/processed/stage1_train80_balanced.jsonl")
    training.zoom_probability = args.zoom_probability
    validation = ErrorAwareCrowdDataset(ROOT / "error_aware/processed/validation.jsonl")
    if {r["image"] for r in training.records} & {r["image"] for r in validation.records}:
        raise RuntimeError("Training/validation image overlap")
    mining = rebalance_sparse_training(model, training, device, args.smoke_test) if args.sparse_rebalance else None
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            if "backbone" in name:
                parameter.requires_grad_(False)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True)
    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and "backbone" in n], "lr": 5e-7},
              {"params": [p for n, p in model.named_parameters() if p.requires_grad and "backbone" not in n], "lr": args.head_lr}]
    optimizer = torch.optim.AdamW([group for group in groups if group["params"]], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    if args.reference_checkpoint:
        reference = CrowdCounterService(args.reference_checkpoint, device_name="cuda")
        baseline = evaluate(reference.model, validation, device, 8 if args.smoke_test else None)
        del reference
    else:
        baseline = evaluate(model, validation, device, 8 if args.smoke_test else None)
    print("BASELINE", baseline, flush=True)
    history = []
    best_mae = baseline["sparse_image_mae"] if args.sparse_rebalance else baseline["mae"]
    if not args.smoke_test:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if mining:
            (args.output_dir / "hard_negative_summary.json").write_text(json.dumps(mining, indent=2))
            (args.output_dir / "sparse_training_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in training.records))
    for epoch in range(1, (1 if args.smoke_test else args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        batches = min(8, len(loader)) if args.smoke_test else len(loader)
        for index, (images, density, count) in enumerate(loader):
            if index >= batches:
                break
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(images.to(device, non_blocking=True))["density"]
            output = output.float()
            count = count.to(device)
            weights = torch.where(count == 0, args.empty_weight, 1.0)
            loss_density = (F.mse_loss(output * 10, density.to(device) * 10, reduction="none").mean((1, 2, 3)) * weights).mean()
            loss_count = (torch.abs(output.sum((1, 2, 3)) - count) * weights).mean()
            loss = loss_density + args.count_weight * loss_count
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}, batch {index}")
            group_start = (index // args.accumulate) * args.accumulate
            group_size = min(args.accumulate, batches - group_start)
            scaler.scale(loss / group_size).backward()
            if (index + 1) % args.accumulate == 0 or index + 1 == batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if index % 250 == 0:
                print(f"Epoch {epoch}/{args.epochs} batch {index}/{batches}; loss={loss.item():.5f}; count_loss={loss_count.item():.4f}; peak_VRAM={torch.cuda.max_memory_reserved()/2**30:.2f} GiB", flush=True)
        metrics = evaluate(model, validation, device, 8 if args.smoke_test else None)
        record = {"epoch": epoch, "validation": metrics, "seconds": time.perf_counter() - start,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        history.append(record)
        print("EPOCH_RESULT", record, flush=True)
        if not args.smoke_test:
            checkpoint = {"model_state_dict": model.state_dict(), "epoch": epoch,
                          "validation": metrics, "source_checkpoint": str(args.checkpoint),
                          "training_config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
            torch.save(checkpoint, args.output_dir / "coarse_last.pth")
            torch.save(checkpoint, args.output_dir / f"coarse_epoch_{epoch}.pth")
            eligible = (metrics["sparse_image_mae"] < best_mae and metrics["image_mae"] <= baseline["image_mae"]) if args.sparse_rebalance else (
                metrics["mae"] < best_mae and metrics["sparse_mae"] <= baseline["sparse_mae"] and metrics["image_mae"] <= baseline["image_mae"])
            if eligible:
                best_mae = metrics["sparse_image_mae"] if args.sparse_rebalance else metrics["mae"]
                torch.save(checkpoint, args.output_dir / "coarse_best.pth")
                print("Saved improved validation checkpoint", flush=True)
            (args.output_dir / "training_history.json").write_text(json.dumps({"baseline": baseline,
                "reference_checkpoint": str(args.reference_checkpoint or args.checkpoint), "epochs": history}, indent=2))
    print("SMOKE TEST PASSED" if args.smoke_test else "TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
