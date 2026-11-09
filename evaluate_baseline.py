
import torch
from torch.utils.data import DataLoader, Subset, random_split

from dataset_loader import VisDroneCrowdDataset
from model import CrowdCounter


SEED = 42
torch.manual_seed(SEED)

dataset = VisDroneCrowdDataset(
    image_size=(320, 320),
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

_, val_set = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)

# Match the smoke-test validation subset.
val_set = Subset(
    val_set,
    range(min(32, len(val_set))),
)

loader = DataLoader(
    val_set,
    batch_size=2,
    shuffle=False,
)

checkpoint = torch.load(
    "multiscale_crowdcounter.pth",
    map_location="cpu",
    weights_only=True,
)

model = CrowdCounter()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

absolute_error = 0.0
squared_error = 0.0
sample_count = 0

print("First 10 validation predictions:")

with torch.no_grad():
    for images, density_maps, counts in loader:
        predictions = model(images)

        predicted_counts = predictions.sum(dim=(1, 2, 3))

        for ground_truth, predicted in zip(
            counts,
            predicted_counts,
        ):
            gt = ground_truth.item()
            pred = predicted.item()

            if sample_count < 10:
                print(
                    f"Ground truth: {gt:7.1f} | "
                    f"Prediction: {pred:9.3f}"
                )

            error = pred - gt
            absolute_error += abs(error)
            squared_error += error ** 2
            sample_count += 1

mae = absolute_error / sample_count
rmse = (squared_error / sample_count) ** 0.5

print(f"\nMAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"Checkpoint epoch: {checkpoint['epoch']}")
