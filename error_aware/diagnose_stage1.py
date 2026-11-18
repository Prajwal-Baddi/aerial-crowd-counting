import torch
from torch.utils.data import DataLoader
from collections import Counter

from data import ErrorAwareCrowdDataset
from model import CoarseCounter


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ErrorAwareCrowdDataset(
        "error_aware/processed/train.jsonl"
    )

    validation_dataset = ErrorAwareCrowdDataset(
        "error_aware/processed/validation.jsonl"
    )

    # Inspect the count distribution.
    for name, dataset in [
        ("Training", train_dataset),
        ("Validation", validation_dataset),
    ]:
        counts = []

        for i in range(len(dataset)):
            _, _, count = dataset[i]
            counts.append(int(count))

        empty = sum(c == 0 for c in counts)
        nonempty = len(counts) - empty

        print(f"\n{name} dataset")
        print("-" * 45)
        print(f"Total crops:       {len(counts)}")
        print(f"Empty crops:       {empty} ({100 * empty / len(counts):.2f}%)")
        print(f"Non-empty crops:   {nonempty} ({100 * nonempty / len(counts):.2f}%)")
        print(f"Average count:     {sum(counts) / len(counts):.3f}")
        print(f"Maximum count:     {max(counts)}")
        print(f"Median count:      {sorted(counts)[len(counts) // 2]}")

        print("Count frequency (0–10):")
        frequency = Counter(counts)

        for count in range(11):
            print(f"  {count}: {frequency.get(count, 0)}")

    # Inspect predictions from the untrained model.
    print("\nUntrained model prediction check")
    print("-" * 45)

    loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = CoarseCounter(pretrained=True).to(device)
    model.eval()

    empty_predictions = []
    nonempty_predictions = []

    with torch.no_grad():
        for i, (images, densities, counts) in enumerate(loader):
            if i >= 30:
                break

            images = images.to(device)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                output = model(images)

            prediction = output["density"].float().sum().item()
            actual = counts.item()

            if actual == 0:
                empty_predictions.append(prediction)
            else:
                nonempty_predictions.append((actual, prediction))

            print(
                f"Sample {i:02d} | "
                f"Actual: {actual:6.1f} | "
                f"Predicted: {prediction:8.2f}"
            )

    if empty_predictions:
        print(
            "\nAverage prediction on empty crops:",
            sum(empty_predictions) / len(empty_predictions),
        )

    if nonempty_predictions:
        print("\nNon-empty crop predictions:")
        for actual, prediction in nonempty_predictions:
            print(f"Actual: {actual:6.1f} | Predicted: {prediction:8.2f}")


if __name__ == "__main__":
    main()