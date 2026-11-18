
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image


SEED = 42
CROP_SIZE = 512
DENSITY_STRIDE = 8
GAUSSIAN_SIGMA = 3.0

# Preserve the current dataset convention.
SELECTED_CATEGORY = 1

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10


def read_annotations(annotation_path):
    """
    Read VisDrone annotations.

    Expected columns:
    x, y, width, height, score, category, truncation, occlusion

    Only category 1 (pedestrian) is retained.
    """
    points = []
    malformed_rows = 0
    ignored_categories = 0

    if not annotation_path.exists():
        return points, 0, 0, 1

    with annotation_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().rstrip(",")

            if not line:
                continue

            values = [value.strip() for value in line.split(",")]

            if len(values) < 8:
                malformed_rows += 1
                continue

            try:
                x = float(values[0])
                y = float(values[1])
                width = float(values[2])
                height = float(values[3])
                category = int(float(values[5]))
            except (ValueError, IndexError):
                malformed_rows += 1
                continue

            if width <= 0 or height <= 0:
                malformed_rows += 1
                continue

            if category != SELECTED_CATEGORY:
                ignored_categories += 1
                continue

            # Current project convention: bounding-box center.
            center_x = x + width / 2.0
            center_y = y + height / 2.0

            points.append([center_x, center_y])

    return points, malformed_rows, ignored_categories, 0


def assign_split(image_paths, seed=SEED):
    """Split original images before generating crops."""
    shuffled = list(image_paths)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def generate_crop_records(
    image_path,
    annotation_path,
    dataset_root,
    split,
):
    """
    Generate metadata for non-overlapping 512x512 crops.

    Images are padded on the right and bottom to multiples of 512.
    Every pedestrian is assigned to exactly one crop using its center.
    """
    records = []

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    points, malformed, ignored, missing = read_annotations(
        annotation_path
    )

    if missing:
        raise FileNotFoundError(
            f"Missing annotation file: {annotation_path}"
        )

    # Ignore centers outside the original image boundaries.
    valid_points = [
        point
        for point in points
        if 0 <= point[0] < image_width
        and 0 <= point[1] < image_height
    ]

    padded_width = (
        (image_width + CROP_SIZE - 1) // CROP_SIZE
    ) * CROP_SIZE

    padded_height = (
        (image_height + CROP_SIZE - 1) // CROP_SIZE
    ) * CROP_SIZE

    relative_image_path = image_path.relative_to(
        dataset_root
    ).as_posix()

    crops_x = padded_width // CROP_SIZE
    crops_y = padded_height // CROP_SIZE

    # Assign each point to exactly one crop.
    points_by_crop = {}

    for center_x, center_y in valid_points:
        crop_column = int(center_x // CROP_SIZE)
        crop_row = int(center_y // CROP_SIZE)

        crop_index = (crop_row, crop_column)

        local_x = center_x - crop_column * CROP_SIZE
        local_y = center_y - crop_row * CROP_SIZE

        points_by_crop.setdefault(crop_index, []).append(
            [local_x, local_y]
        )

    for crop_row in range(crops_y):
        for crop_column in range(crops_x):
            x0 = crop_column * CROP_SIZE
            y0 = crop_row * CROP_SIZE

            crop_points = points_by_crop.get(
                (crop_row, crop_column), []
            )

            record = {
                "split": split,
                "image": relative_image_path,
                "original_width": image_width,
                "original_height": image_height,
                "padded_width": padded_width,
                "padded_height": padded_height,
                "crop_x": x0,
                "crop_y": y0,
                "crop_size": CROP_SIZE,
                "density_size": CROP_SIZE // DENSITY_STRIDE,
                "density_stride": DENSITY_STRIDE,
                "gaussian_sigma": GAUSSIAN_SIGMA,
                "points": crop_points,
                "count": len(crop_points),
            }

            records.append(record)

    statistics = {
        "malformed_rows": malformed,
        "ignored_categories": ignored,
        "valid_pedestrian_points": len(valid_points),
        "out_of_bounds_points": len(points) - len(valid_points),
    }

    return records, statistics


def main():
    parser = argparse.ArgumentParser(
        description="Prepare VisDrone crops for error-aware crowd counting."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/VisDrone2019-DET-train"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("error_aware/processed"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}"
        )

    if not annotations_dir.exists():
        raise FileNotFoundError(
            f"Annotations directory not found: {annotations_dir}"
        )

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise RuntimeError("No images found in the dataset.")

    annotation_paths = {
        path.stem: path
        for path in annotations_dir.glob("*.txt")
    }

    missing_annotations = [
        path.name
        for path in image_paths
        if path.stem not in annotation_paths
    ]

    if missing_annotations:
        raise RuntimeError(
            f"{len(missing_annotations)} images have no annotations. "
            f"First examples: {missing_annotations[:5]}"
        )

    print(f"Dataset images: {len(image_paths)}")
    print(f"Dataset root: {dataset_root}")
    print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")
    print(f"Density stride: {DENSITY_STRIDE}")
    print(f"Gaussian sigma: {GAUSSIAN_SIGMA}")

    splits = assign_split(image_paths, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_statistics = Counter()
    split_summary = {}
    total_crops = 0

    for split_name, split_images in splits.items():
        manifest_path = (
            args.output_dir / f"{split_name}.jsonl"
        )

        split_crops = 0
        split_pedestrians = 0

        print(
            f"\nPreparing {split_name}: "
            f"{len(split_images)} original images"
        )

        with manifest_path.open(
            "w", encoding="utf-8"
        ) as manifest:
            for image_index, image_path in enumerate(
                split_images, start=1
            ):
                annotation_path = (
                    annotation_paths[image_path.stem]
                )

                records, stats = generate_crop_records(
                    image_path=image_path,
                    annotation_path=annotation_path,
                    dataset_root=dataset_root,
                    split=split_name,
                )

                for record in records:
                    manifest.write(
                        json.dumps(record) + "\n"
                    )

                split_crops += len(records)
                split_pedestrians += (
                    stats["valid_pedestrian_points"]
                )

                all_statistics.update(stats)

                if image_index % 250 == 0:
                    print(
                        f"  Processed {image_index}/"
                        f"{len(split_images)} images"
                    )

        total_crops += split_crops

        split_summary[split_name] = {
            "original_images": len(split_images),
            "crops": split_crops,
            "pedestrian_points": split_pedestrians,
            "manifest": manifest_path.name,
        }

        print(f"  Crops: {split_crops}")
        print(f"  Pedestrian points: {split_pedestrians}")
        print(f"  Saved: {manifest_path}")

    summary = {
        "seed": args.seed,
        "dataset": "VisDrone2019-DET-train",
        "selected_category": SELECTED_CATEGORY,
        "point_convention": "bbox_center",
        "crop_size": CROP_SIZE,
        "density_stride": DENSITY_STRIDE,
        "density_size": CROP_SIZE // DENSITY_STRIDE,
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "splits": split_summary,
        "total_crops": total_crops,
        "annotation_statistics": dict(all_statistics),
    }

    summary_path = args.output_dir / "summary.json"

    with summary_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n========== PREPARATION COMPLETE ==========")
    print(f"Original images: {len(image_paths)}")
    print(f"Total crops: {total_crops}")
    print(f"Summary: {summary_path}")
    print("===========================================")


if __name__ == "__main__":
    main()
