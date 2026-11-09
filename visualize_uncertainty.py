
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split

from dataset_loader import VisDroneCrowdDataset
from model import CrowdCounter
from uncertainty import estimate_uncertainty


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

CHECKPOINT_PATH = "mc_multiscale_crowdcounter.pth"

NUM_SAMPLES = 30
NUM_IMAGES = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

dataset = VisDroneCrowdDataset(
    root="dataset/VisDrone2019-DET-train",
    image_size=(320, 320)
)

total_size = len(dataset)

train_size = int(0.8 * total_size)
val_size = total_size - train_size

generator = torch.Generator().manual_seed(SEED)

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

dropout_rate = checkpoint.get(
    "dropout_rate",
    0.25
)

model = CrowdCounter(
    dropout_rate=dropout_rate
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])
print("Best validation MAE:", checkpoint["best_mae"])
print("Using device:", DEVICE)


# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    NUM_IMAGES,
    4,
    figsize=(16, 4 * NUM_IMAGES)
)

if NUM_IMAGES == 1:
    axes = axes.reshape(1, 4)

for index, (image, target, count) in enumerate(val_loader):

    if index >= NUM_IMAGES:
        break

    image_gpu = image.to(
        DEVICE,
        non_blocking=True
    )

    (
        mean_density,
        variance_map,
        predicted_count,
        count_variance
    ) = estimate_uncertainty(
        model,
        image_gpu,
        num_samples=NUM_SAMPLES
    )

    # Convert tensors to NumPy
    image_np = image[0].permute(
        1, 2, 0
    ).cpu().numpy()

    target_np = target[0, 0].cpu().numpy()

    mean_density_np = (
        mean_density[0, 0].cpu().numpy()
    )

    variance_np = (
        variance_map[0, 0].cpu().numpy()
    )

    gt_count = count.item()
    pred_count = predicted_count.item()

    absolute_error = abs(
        gt_count - pred_count
    )

    mean_variance = variance_np.mean()

    print(f"\nImage {index + 1}")
    print("Ground truth count:", round(gt_count, 2))
    print("Predicted count:", round(pred_count, 2))
    print("Absolute error:", round(absolute_error, 2))
    print("Mean pixel variance:", round(mean_variance, 6))
    print("Count variance:", round(count_variance.item(), 6))

    # Original image
    axes[index, 0].imshow(image_np)
    axes[index, 0].set_title("Original Image")

    # Ground truth
    axes[index, 1].imshow(
        target_np,
        cmap="jet"
    )
    axes[index, 1].set_title(
        f"Ground Truth (Count: {gt_count:.1f})"
    )

    # Mean prediction
    axes[index, 2].imshow(
        mean_density_np,
        cmap="jet"
    )
    axes[index, 2].set_title(
        f"Mean Prediction ({pred_count:.1f})"
    )

    # Uncertainty
    axes[index, 3].imshow(
        variance_np,
        cmap="hot"
    )
    axes[index, 3].set_title(
        "MC Dropout Uncertainty"
    )

    for column in range(4):
        axes[index, column].axis("off")


plt.tight_layout()

plt.savefig(
    "mc_uncertainty_visualization.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nVisualization saved as mc_uncertainty_visualization.png")
