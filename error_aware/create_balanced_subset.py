import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_manifest(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def write_manifest(path, records):
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="error_aware/processed/train.jsonl",
    )

    parser.add_argument(
        "--output",
        default="error_aware/processed/train_balanced.jsonl",
    )

    parser.add_argument("--empty", type=int, default=4000)
    parser.add_argument("--non-empty", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = read_manifest(input_path)

    empty_records = []
    nonempty_records = []

    for record in records:
        count = int(record["count"])

        if count == 0:
            empty_records.append(record)
        else:
            nonempty_records.append(record)

    print("=" * 60)
    print("BALANCED TRAINING SUBSET GENERATION")
    print("=" * 60)

    print(f"Original training crops: {len(records)}")
    print(f"Empty crops available: {len(empty_records)}")
    print(f"Non-empty crops available: {len(nonempty_records)}")

    if len(empty_records) < args.empty:
        raise ValueError("Not enough empty crops for requested subset.")

    if len(nonempty_records) < args.non_empty:
        raise ValueError("Not enough non-empty crops for requested subset.")

    selected_empty = random.sample(empty_records, args.empty)
    selected_nonempty = random.sample(nonempty_records, args.non_empty)

    selected_records = selected_empty + selected_nonempty

    random.shuffle(selected_records)

    write_manifest(output_path, selected_records)

    selected_counts = [
        int(record["count"]) for record in selected_records
    ]

    frequency = Counter(selected_counts)

    print("\nSelected subset:")
    print(f"Total crops: {len(selected_records)}")
    print(f"Empty crops: {sum(c == 0 for c in selected_counts)}")
    print(f"Non-empty crops: {sum(c > 0 for c in selected_counts)}")
    print(f"Average people per crop: {sum(selected_counts) / len(selected_counts):.3f}")
    print(f"Maximum people in a crop: {max(selected_counts)}")

    print("\nCount distribution:")
    for count in range(11):
        print(f"Count {count}: {frequency.get(count, 0)}")

    print(f"\nSaved manifest: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()