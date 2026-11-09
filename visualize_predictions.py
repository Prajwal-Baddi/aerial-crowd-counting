
import matplotlib.pyplot as plt
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

val_set = Subset(
    val_set,
    range(min(32, len(val_set))),
)

loader = DataLoader(
    val_set,
    batch_size=1,
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

fig, axes = plt.subplots(4, 3, figsize=(12, 14))

shown = 0

with torch.no_grad():
    for image, target, count in loader:
        prediction = model(image)
        predicted_count = prediction.sum().item()

        image_np = image[0].permute(1, 2, 0).numpy()
        target_np = target[0, 0].numpy()
        prediction_np = prediction[0, 0].numpy()

        row = shown

        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(
            f"Image | GT count: {count.item():.0f}"
        )

        axes[row, 1].imshow(target_np, cmap="hot")
        axes[row, 1].set_title("Ground-truth density")

        axes[row, 2].imshow(prediction_np, cmap="hot")
        axes[row, 2].set_title(
            f"Predicted density | Count: {predicted_count:.1f}"
        )

        for col in range(3):
            axes[row, col].axis("off")

        shown += 1

        if shown >= 4:
            break

plt.tight_layout()
plt.savefig("prediction_diagnostics.png", dpi=150)
plt.show()

print("Saved visualization to prediction_diagnostics.png")
