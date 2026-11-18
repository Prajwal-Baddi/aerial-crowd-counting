"""Fit sparse counting rules on validation, then freeze them for test evaluation."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from finetune_low_vram import ROOT, CrowdCounterService, ErrorAwareCrowdDataset, image_batches
from backend.localization import component_masses, choose_count


@torch.inference_mode()
def collect(service, split, cache_dir):
    manifest = ROOT / "error_aware/processed" / f"{split}.jsonl"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    checkpoint_digest = service.checkpoint_digest
    cache_path = cache_dir / f"{split}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached["checkpoint_sha256"] == checkpoint_digest and cached["manifest_sha256"] == digest:
            print(f"Reusing {split} features", flush=True)
            return cached["images"]
    dataset = ErrorAwareCrowdDataset(manifest)
    images = []
    current_name, density, gt = None, None, 0

    def finish():
        masses = component_masses(density)
        images.append({"image": current_name, "target": gt,
                       "raw_count": float(density.sum(dtype=np.float64)),
                       "masses": masses[masses >= 0.01].tolist()})

    processed = 0
    for tensor, chunk in image_batches(dataset, dataset.records, 4):
        with torch.autocast(service.device.type, dtype=torch.float16, enabled=service.device.type == "cuda"):
            output = service.model(tensor.to(service.device))["density"].float().cpu().numpy()[:, 0]
        for tile_density, record in zip(output, chunk):
            if record["image"] != current_name:
                if current_name is not None:
                    finish()
                current_name = record["image"]
                density = np.zeros(((record["original_height"] + 7) // 8,
                                    (record["original_width"] + 7) // 8), dtype=np.float32)
                gt = 0
            y, x = record["crop_y"] // 8, record["crop_x"] // 8
            h, w = min(64, density.shape[0] - y), min(64, density.shape[1] - x)
            density[y:y+h, x:x+w] = tile_density[:h, :w]
            gt += record["count"]
        processed += len(chunk)
        if (processed - len(chunk)) % 1000 == 0:
            print(f"{split}: {processed}/{len(dataset)} crops", flush=True)
    if current_name is not None:
        finish()
    if len({r["image"] for r in images}) != len(images):
        raise ValueError("Manifest must group crops by source image")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"checkpoint_sha256": checkpoint_digest,
                                    "manifest_sha256": digest, "images": images}))
    return images


def metrics(images, policy):
    targets = np.array([r["target"] for r in images])
    counts = np.array([choose_count(r["raw_count"], r["masses"], policy)[0] for r in images])
    error = np.abs(counts - targets)
    sparse = (targets > 0) & (targets <= 5)
    return {"images": len(images), "rounded_image_mae": float(error.mean()),
            "sparse_image_mae": float(error[sparse].mean()),
            "empty_image_mae": float(error[targets == 0].mean()),
            "exact_images": int(np.sum(error == 0))}


def sample_features(service, path):
    tiles = []
    handle = service.model.register_forward_hook(
        lambda model, args, output: tiles.append(output["density"].float().cpu().numpy()[0, 0].copy()))
    try:
        with Image.open(path) as image:
            width, height = image.size
            result = service.predict(image)
    finally:
        handle.remove()
    density = np.zeros((((height + 511) // 512) * 64, ((width + 511) // 512) * 64), dtype=np.float32)
    columns = (width + 511) // 512
    for index, tile in enumerate(tiles):
        y, x = (index // columns) * 64, (index % columns) * 64
        density[y:y+64, x:x+64] = tile
    density = density[:(height + 7)//8, :(width + 7)//8]
    return {"image": str(path), "raw_count": result["raw_count"],
            "masses": component_masses(density).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "error_aware/checkpoints_balanced_low_vram/coarse_validated.pth")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/localization_cache")
    parser.add_argument("--report", type=Path, default=ROOT / "error_aware/evaluation/localization_counting_report.json")
    parser.add_argument("--objective", choices=("overall", "sparse"), default="overall")
    parser.add_argument("--max-component-mass", type=float, default=1.5,
                        help="Keep the integral when a blob may contain merged people")
    parser.add_argument("--max-test-overall-regression", type=float, default=0.0,
                        help="Explicit fractional tolerance for overall test MAE; default is strict")
    args = parser.parse_args()
    if not 0 <= args.max_test_overall_regression <= 0.01:
        parser.error("Overall test regression tolerance must be between zero and 1%")
    service = CrowdCounterService(args.checkpoint)
    service.checkpoint_digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    validation = collect(service, "validation", args.cache_dir)
    baseline = metrics(validation, None)
    examples = [(ROOT / "WhatsApp Image 2026-11-18 at 11.16.26.jpeg", 3),
                (ROOT / "WhatsApp Image 2026-11-18 at 08.59.23.jpeg", 5)]
    samples = [(sample_features(service, path), expected) for path, expected in examples]
    eligible = []
    for threshold, lower, upper, maximum in itertools.product(
            (0.2, 0.3, 0.4, 0.5, 0.6), (2.0, 3.0, 4.0), (8.0, 12.0, 20.0), (args.max_component_mass,)):
        policy = {"mass_threshold": threshold, "raw_min": lower, "raw_max": upper,
                  "max_component_mass": maximum}
        score = metrics(validation, policy)
        regressions_pass = all(choose_count(s["raw_count"], s["masses"], policy)[0] == target for s, target in samples)
        if (regressions_pass and score["rounded_image_mae"] <= baseline["rounded_image_mae"]
                and score["sparse_image_mae"] < baseline["sparse_image_mae"]):
            eligible.append((score["sparse_image_mae"], score["rounded_image_mae"], policy, score))
    if not eligible:
        raise RuntimeError("No localization policy improves validation and passes both regression images")
    _, _, policy, validation_score = min(eligible, key=(lambda r: (r[1], r[0])) if args.objective == "overall" else (lambda r: (r[0], r[1])))
    print("SELECTED", policy, validation_score, "BASELINE", baseline, flush=True)
    test = collect(service, "test", args.cache_dir)
    test_baseline, test_score = metrics(test, None), metrics(test, policy)
    accepted = (test_score["rounded_image_mae"] <= test_baseline["rounded_image_mae"] * (1 + args.max_test_overall_regression)
                and test_score["sparse_image_mae"] < test_baseline["sparse_image_mae"])
    report = {"checkpoint_sha256": service.checkpoint_digest, "policy": policy,
              "selection_objective": args.objective,
              "acceptance_criteria": {
                  "max_relative_test_overall_regression": args.max_test_overall_regression,
                  "strict_test_overall_nonregression": test_score["rounded_image_mae"] <= test_baseline["rounded_image_mae"]},
              "validation": {"baseline": baseline, "localized": validation_score},
              "test": {"baseline": test_baseline, "localized": test_score}, "accepted": accepted,
              "regressions": [{**s, "expected_count": target,
                               "predicted_count": choose_count(s["raw_count"], s["masses"], policy)[0]}
                              for s, target in samples]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
