
import torch
import torch.nn.functional as F

from dataset_loader import VisDroneCrowdDataset
from model import CrowdCounter, resize_density_target

torch.manual_seed(42)
torch.set_num_threads(2)

dataset = VisDroneCrowdDataset(image_size=(320, 320))
model = CrowdCounter()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Use a few samples and deliberately reuse them for this diagnostic.
batch = [dataset[i] for i in range(4)]
images = torch.stack([item[0] for item in batch])
targets = torch.stack([item[1] for item in batch])
counts = torch.stack([item[2] for item in batch])

target_small = resize_density_target(targets, (80, 80))

print("Image range:", images.min().item(), images.max().item())
print("Target density range:", target_small.min().item(),
      target_small.max().item())
print("Target sums:", target_small.sum((1, 2, 3)).tolist())
print("Ground-truth counts:", counts.tolist())

def parameter_snapshot(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

before = parameter_snapshot(model)

model.train()
predictions = model(images)
print("Initial predicted counts:",
      predictions.sum((1, 2, 3)).detach().tolist())

loss = F.mse_loss(predictions, target_small)
optimizer.zero_grad()
loss.backward()

grad_norm = torch.sqrt(sum(
    p.grad.detach().square().sum()
    for p in model.parameters()
    if p.grad is not None
))

print("Initial loss:", loss.item())
print("Gradient norm:", grad_norm.item())

optimizer.step()
after = parameter_snapshot(model)

print("Parameter change after one update:",
      (after - before).abs().max().item())

model.eval()
with torch.no_grad():
    predictions_after = model(images)
    print("Counts after one update:",
          predictions_after.sum((1, 2, 3)).tolist())
    print("Loss after one update:",
          F.mse_loss(predictions_after, target_small).item())
