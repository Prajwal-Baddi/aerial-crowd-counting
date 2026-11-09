
from pathlib import Path
from collections import Counter

ROOT = Path("dataset")

if not ROOT.exists():
    print(f"Dataset folder not found: {ROOT.resolve()}")
    print("Extract your ZIP into the dataset folder first.")
    raise SystemExit(1)

print(f"Dataset location: {ROOT.resolve()}\n")

extensions = Counter()
image_files = []
annotation_files = []
other_files = []

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
annotation_exts = {".mat", ".json", ".csv", ".txt", ".npy", ".npz", ".xml"}

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    ext = path.suffix.lower()
    extensions[ext or "[no extension]"] += 1

    if ext in image_exts:
        image_files.append(path)
    elif ext in annotation_exts:
        annotation_files.append(path)
    else:
        other_files.append(path)

print("File counts by extension:")
for ext, count in extensions.most_common():
    print(f"  {ext}: {count}")

print(f"\nTotal images: {len(image_files)}")
print(f"Potential annotation files: {len(annotation_files)}")

print("\nSample image paths:")
for path in image_files[:10]:
    print(" ", path.relative_to(ROOT))

print("\nSample annotation paths:")
for path in annotation_files[:20]:
    print(" ", path.relative_to(ROOT))

print("\nTop-level folders and files:")
for path in sorted(ROOT.iterdir()):
    kind = "DIR " if path.is_dir() else "FILE"
    print(f"  [{kind}] {path.name}")
