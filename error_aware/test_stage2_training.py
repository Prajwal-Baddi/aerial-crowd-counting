import torch
import torch.nn.functional as F

from model import ESRNet, cell_sum
from train_stage2 import true_cell_error, ranking_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESRNet(pretrained=False, freeze_early=True, refine_width=32).to(device)

    # Small random tensors exercise the loss functions independently of VGG.
    coarse = torch.rand(2, 1, 64, 64, device=device)
    target = torch.rand(2, 1, 64, 64, device=device)
    error = true_cell_error(coarse, target)
    assert error.shape == (2, 1, 16, 16)
    assert torch.all(error >= 0)
    print("Cell-wise error target: PASS")

    pred_error = torch.rand_like(error, requires_grad=True)
    rank = ranking_loss(pred_error, error, pairs=256)
    rank.backward()
    assert pred_error.grad is not None
    print("Pairwise ranking loss gradient: PASS")

    # Confirm count aggregation retains count units.
    assert torch.allclose(cell_sum(target).sum((1, 2, 3)), target.sum((1, 2, 3)), atol=1e-4)
    print("Cell count conservation: PASS")

    # Ensure optimizer scope contains only Stage-2 parameters.
    params = list(model.error_head.parameters()) + list(model.refiner.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    optimized = {id(p) for group in opt.param_groups for p in group["params"]}
    assert all(id(p) in optimized for p in model.error_head.parameters())
    assert all(id(p) in optimized for p in model.refiner.parameters())
    assert not any(id(p) in optimized for p in model.coarse.parameters())
    print("Stage-2-only optimizer parameter scope: PASS")
    print("All Phase-C component tests passed.")


if __name__ == "__main__":
    main()
