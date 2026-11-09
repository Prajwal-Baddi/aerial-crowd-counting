
import torch

from dataset_loader import VisDroneCrowdDataset

dataset = VisDroneCrowdDataset(image_size=(320, 320))

print("Dataset size:", len(dataset))

for index in [0, 1, 2]:
    image, density, count = dataset[index]

    print(f"\nSample {index}")
    print("Image shape:", tuple(image.shape))
    print("Density shape:", tuple(density.shape))
    print("Ground-truth count:", count.item())
    print("Density-map sum:", density.sum().item())

    assert image.shape == (3, 320, 320)
    assert density.shape == (1, 80, 80)
    assert torch.isfinite(image).all()
    assert torch.isfinite(density).all()

    assert abs(density.sum().item() - count.item()) < 1e-3

print("\nAll dataset loader checks passed.")
