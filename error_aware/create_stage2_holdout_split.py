"""
Create an image-level 80/20 split for Stage-1 / Stage-2 training.

Inputs in --data-dir:
  train.jsonl           full training crop manifest
  train_balanced.jsonl  balanced Stage-1 crop manifest

Outputs:
  stage1_train80_balanced.jsonl  Stage-1 crops from 80% of source images
  stage2_train20_holdout.jsonl   ALL crops from the held-out 20% of source images
  stage2_holdout_split_summary.json

The split is by original image, never by crop, to prevent crop-level leakage.
Validation and test manifests are not modified.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_no}: {exc}") from exc
    if not records:
        raise RuntimeError(f"No records found in {path}")
    return records


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="error_aware/processed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-fraction", type=float, default=0.20)
    args = p.parse_args()

    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must be between 0 and 1")

    data_dir = Path(args.data_dir)
    full_path = data_dir / "train.jsonl"
    balanced_path = data_dir / "train_balanced.jsonl"

    full_records = read_jsonl(full_path)
    balanced_records = read_jsonl(balanced_path)

    full_images = sorted({r["image"] for r in full_records})
    balanced_images = {r["image"] for r in balanced_records}

    if not full_images:
        raise RuntimeError("No image identifiers found in train.jsonl")

    rng = random.Random(args.seed)
    shuffled = full_images.copy()
    rng.shuffle(shuffled)

    n_holdout = max(1, round(len(shuffled) * args.holdout_fraction))
    holdout_images = set(shuffled[:n_holdout])
    stage1_images = set(shuffled[n_holdout:])

    # Stage 1 uses the existing balanced manifest, filtered by image.
    stage1_records = [r for r in balanced_records if r["image"] in stage1_images]

    # Stage 2 uses every crop from held-out original training images.
    stage2_records = [r for r in full_records if r["image"] in holdout_images]

    if not stage1_records:
        raise RuntimeError("Stage-1 manifest is empty after image-level filtering")
    if not stage2_records:
        raise RuntimeError("Stage-2 holdout manifest is empty after image-level filtering")

    stage1_record_images = {r["image"] for r in stage1_records}
    stage2_record_images = {r["image"] for r in stage2_records}
    overlap = stage1_record_images & stage2_record_images
    if overlap:
        raise RuntimeError(f"Image leakage detected: {len(overlap)} overlapping images")

    # Every held-out image must be represented in the full manifest, and no
    # held-out crop may be lost.
    expected_stage2 = sum(1 for r in full_records if r["image"] in holdout_images)
    if expected_stage2 != len(stage2_records):
        raise RuntimeError("Stage-2 manifest record-count verification failed")

    stage1_out = data_dir / "stage1_train80_balanced.jsonl"
    stage2_out = data_dir / "stage2_train20_holdout.jsonl"
    summary_out = data_dir / "stage2_holdout_split_summary.json"

    write_jsonl(stage1_out, stage1_records)
    write_jsonl(stage2_out, stage2_records)

    summary = {
        "seed": args.seed,
        "holdout_fraction_requested": args.holdout_fraction,
        "source_train_images": len(full_images),
        "stage1_image_count": len(stage1_images),
        "stage2_holdout_image_count": len(holdout_images),
        "stage1_balanced_manifest_source_records": len(balanced_records),
        "stage1_output_records": len(stage1_records),
        "stage2_full_manifest_source_records": len(full_records),
        "stage2_output_records": len(stage2_records),
        "stage1_unique_images_in_output": len(stage1_record_images),
        "stage2_unique_images_in_output": len(stage2_record_images),
        "image_overlap": len(overlap),
        "stage1_empty_crops": sum(int(r["count"]) == 0 for r in stage1_records),
        "stage1_nonempty_crops": sum(int(r["count"]) > 0 for r in stage1_records),
        "stage2_empty_crops": sum(int(r["count"]) == 0 for r in stage2_records),
        "stage2_nonempty_crops": sum(int(r["count"]) > 0 for r in stage2_records),
        "stage1_manifest": str(stage1_out),
        "stage2_manifest": str(stage2_out),
        "note": (
            "Split is by original image. Stage 1 sees only stage1 images; "
            "Stage 2 trains on all crops from held-out stage2 images. "
            "Validation/test manifests are untouched."
        ),
        "balanced_manifest_images_not_in_full_train": len(balanced_images - set(full_images)),
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nImage-level split verification: PASSED (0 overlapping images).")


if __name__ == "__main__":
    main()
