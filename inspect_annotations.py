
from pathlib import Path
from collections import Counter
import random

DATASET = Path("dataset/VisDrone2019-DET-train")
IMAGE_DIR = DATASET / "images"
ANNOTATION_DIR = DATASET / "annotations"

if not IMAGE_DIR.exists() or not ANNOTATION_DIR.exists():
    raise FileNotFoundError(
        f"Expected images and annotations inside: {DATASET.resolve()}"
    )

image_extensions = {".jpg", ".jpeg", ".png"}
images = {
    p.stem: p
    for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
}

annotations = sorted(ANNOTATION_DIR.glob("*.txt"))
category_counts = Counter()
person_counts = Counter()
invalid_rows = []
empty_annotations = 0
total_rows = 0

print("=== SAMPLE ANNOTATIONS ===")

for path in annotations[:5]:
    print(f"\n{path.name}")
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    if not lines:
        print("  Empty annotation file")
        continue

    for line in lines[:5]:
        print(" ", line)

print("\n=== DATASET VALIDATION ===")

for path in annotations:
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    if not lines:
        empty_annotations += 1

    person_count = 0

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        # VisDrone DET normally uses comma-separated values.
        parts = [part.strip() for part in line.split(",")]

        if len(parts) != 8:
            invalid_rows.append((path.name, line_number, line))
            continue

        try:
            x, y, w, h = map(float, parts[:4])
            category = int(parts[5])
        except ValueError:
            invalid_rows.append((path.name, line_number, line))
            continue

        total_rows += 1
        category_counts[category] += 1

        # VisDrone categories 1 and 2:
        # pedestrian and people.
        if category in (1, 2) and w > 0 and h > 0:
            person_count += 1

    person_counts[person_count] += 1

missing_annotations = sorted(set(images) - {
    p.stem for p in annotations
})
missing_images = sorted({
    p.stem for p in annotations
} - set(images))

print(f"Images: {len(images)}")
print(f"Annotation files: {len(annotations)}")
print(f"Images without annotations: {len(missing_annotations)}")
print(f"Annotations without images: {len(missing_images)}")
print(f"Empty annotation files: {empty_annotations}")
print(f"Valid annotation rows: {total_rows}")
print(f"Malformed rows: {len(invalid_rows)}")

print("\nObject counts by category ID:")
for category, count in sorted(category_counts.items()):
    print(f"  Category {category}: {count}")

print("\nImages by number of person/person-group boxes:")
for count, frequency in sorted(person_counts.items()):
    print(f"  {count} boxes: {frequency} images")

if invalid_rows:
    print("\nFirst malformed rows:")
    for item in invalid_rows[:5]:
        print(" ", item)

print("\n=== SAMPLE IMAGE FILES ===")
for stem in list(images)[:5]:
    print(" ", images[stem].name)
