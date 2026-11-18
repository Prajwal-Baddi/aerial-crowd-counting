"""Validate intermediate weights between the zoom and sparse-scene fine-tunes."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from PIL import Image

from finetune_low_vram import ROOT, CrowdCounterService, ErrorAwareCrowdDataset, evaluate


def average_states(first, second, weight):
    if not 0 <= weight <= 1 or first.keys() != second.keys():
        raise ValueError("Weight must be in [0,1] and checkpoint architectures must match")
    result = {}
    for key, tensor in first.items():
        other = second[key]
        if tensor.shape != other.shape or tensor.dtype != other.dtype:
            raise ValueError(f"Incompatible tensor: {key}")
        if tensor.is_floating_point():
            result[key] = torch.lerp(tensor, other, weight)
        elif torch.equal(tensor, other):
            result[key] = tensor.clone()
        else:
            raise ValueError(f"Cannot average discrete tensor: {key}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, default=ROOT / "error_aware/checkpoints_low_vram/coarse_last.pth")
    parser.add_argument("--second", type=Path, default=ROOT / "error_aware/checkpoints_sparse_low_vram/coarse_epoch_1.pth")
    parser.add_argument("--weight", type=float, default=0.4)
    parser.add_argument("--reference-metrics", type=Path, default=ROOT / "error_aware/checkpoints_sparse_low_vram/training_history.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "error_aware/checkpoints_balanced_low_vram")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--expected-count", type=int, default=3)
    args = parser.parse_args()
    first = torch.load(args.first, map_location="cpu", weights_only=False)["model_state_dict"]
    second = torch.load(args.second, map_location="cpu", weights_only=False)["model_state_dict"]
    state = average_states(first, second, args.weight)
    provenance = {"first": str(args.first), "second": str(args.second), "weight_second": args.weight,
                  "first_sha256": hashlib.sha256(args.first.read_bytes()).hexdigest(),
                  "second_sha256": hashlib.sha256(args.second.read_bytes()).hexdigest()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"coarse_average_{args.weight:.3f}.pth"
    torch.save({"model_state_dict": state, "epoch": "averaged", "provenance": provenance}, checkpoint_path)
    service = CrowdCounterService(checkpoint_path)
    dataset = ErrorAwareCrowdDataset(ROOT / "error_aware/processed/validation.jsonl")
    metrics = evaluate(service.model, dataset, service.device)
    reference = json.loads(args.reference_metrics.read_text())["baseline"]
    eligible = metrics["image_mae"] <= reference["image_mae"] and metrics["sparse_image_mae"] < reference["sparse_image_mae"]
    sample = None
    if args.image:
        with Image.open(args.image) as image:
            sample = {k: v for k, v in service.predict(image).items() if k != "density_map"}
        eligible = eligible and sample["predicted_count"] == args.expected_count
    report = {"provenance": provenance, "reference": reference, "validation": metrics,
              "sample": sample, "eligible": eligible}
    (args.output_dir / f"selection_{args.weight:.3f}.json").write_text(json.dumps(report, indent=2) + "\n")
    if eligible:
        shutil.copyfile(checkpoint_path, args.output_dir / "coarse_best.pth")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
