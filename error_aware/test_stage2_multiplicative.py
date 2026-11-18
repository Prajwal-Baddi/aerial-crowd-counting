"""Unit checks for the multiplicative-additive Stage-2 model and losses."""
import torch

from model_multiplicative import ESRNet, RefineNet, select_topk
from refine_losses_multiplicative import build_training_mask, refinement_loss


def main():
    torch.manual_seed(7)

    # Refiner output is nonnegative and starts near identity.
    refiner = RefineNet(width=24)
    feat = torch.randn(2, 256, 24, 24)
    den = torch.rand(2, 1, 24, 24) * 0.02
    gain, add = refiner(feat, den)
    assert gain.shape == (2, 1, 4, 4), gain.shape
    assert add.shape == (2, 1, 4, 4), add.shape
    assert torch.all(gain > 0) and torch.all(gain < 3)
    assert torch.all(add >= 0)
    assert abs(gain.mean().item() - 1.0) < 0.02, gain.mean().item()
    assert add.mean().item() < 0.01, add.mean().item()

    # The per-pixel additive term must remain negligible after summing a full crop.
    # With all 256 cells selected, the initial count drift should stay below 0.01.
    full_crop_add_mass = add.mean().item() * 256 * 16
    assert full_crop_add_mass < 0.01, full_crop_add_mass

    # Selection threshold suppresses cells below threshold.
    scores = torch.tensor([[[[0.01, 0.2], [0.3, 0.04]]]])
    selected = select_topk(scores, 0.5, min_score=0.1)
    assert selected.sum().item() == 2, selected

    # End-to-end vectorized scatter on synthetic feature maps.
    model = ESRNet(pretrained=False, freeze_early=True, refine_width=24)
    coarse = torch.rand(2, 1, 64, 64)
    features = torch.randn(2, 256, 128, 128, requires_grad=True)
    mask = torch.zeros(2, 1, 16, 16, dtype=torch.bool)
    mask[0, 0, 2, 3] = True
    mask[1, 0, 5, 7] = True

    final, aux = model.apply_refine(coarse, features, mask, return_aux=True)
    assert final.shape == coarse.shape
    assert torch.all(final >= 0)
    pixel_mask = mask.repeat_interleave(4, -2).repeat_interleave(4, -1)
    assert torch.equal(final[~pixel_mask], coarse[~pixel_mask])
    assert aux["gain"].shape == (2, 1, 4, 4)
    assert aux["add"].shape == (2, 1, 4, 4)

    # Check the refinement objective and gradient flow.
    gt = coarse.detach().clone()
    gt[0, 0, 8:12, 12:16] *= 0.7
    gt[1, 0, 20:24, 28:32] += 0.01
    loss, parts = refinement_loss(final, coarse, gt, mask, aux)
    assert torch.isfinite(loss)
    assert all(torch.is_tensor(v) and torch.isfinite(v) for v in parts.values())
    loss.backward()
    grads = [p.grad for p in model.refiner.parameters() if p.grad is not None]
    assert grads, "No refiner gradients"
    assert all(torch.isfinite(g).all() for g in grads), "Non-finite refiner gradients"

    # Background sampling returns a boolean mask and can add cells beyond primary budget.
    err = torch.rand(2, 1, 16, 16)
    train_mask = build_training_mask(err, err, 0.1, p_true=1.0, background_frac=0.25)
    assert train_mask.dtype == torch.bool
    print("All multiplicative Stage-2 unit checks passed.")


if __name__ == "__main__":
    main()
