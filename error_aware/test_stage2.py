import argparse
import torch

from model import ESRNet, cell_sum, select_topk, mask_to_pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="error_aware/checkpoints/coarse_best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    net = ESRNet(pretrained=False, freeze_early=True).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    net.coarse.load_state_dict(checkpoint, strict=True)
    print("Stage-1 checkpoint loaded strictly.")

    for parameter in net.coarse.parameters():
        parameter.requires_grad = False
    net.coarse.eval()

    # Verify top-k and count-preserving cell aggregation.
    images = torch.rand(1, 3, 512, 512, device=device)
    with torch.no_grad():
        encoded = net.encode(images)
    assert encoded["density"].shape == (1, 1, 64, 64)
    assert encoded["context"].shape == (1, 128, 64, 64)
    assert encoded["stride4_features"].shape == (1, 256, 128, 128)
    assert encoded["error_scores"].shape == (1, 1, 16, 16)

    mask = select_topk(encoded["error_scores"], 0.25)
    assert int(mask.sum()) == 64
    assert mask_to_pixels(mask).shape == (1, 1, 64, 64)

    d = torch.rand(2, 1, 64, 64, device=device)
    assert torch.allclose(cell_sum(d).sum((1,2,3)), d.sum((1,2,3)), atol=1e-4)

    # A single selected cell should only alter its corresponding 4x4 density block.
    single = torch.zeros(1, 1, 16, 16, dtype=torch.bool, device=device)
    single[0, 0, 0, 0] = True
    coarse_density = torch.rand(1, 1, 64, 64, device=device)
    stride4 = torch.rand(1, 256, 128, 128, device=device)
    with torch.no_grad():
        single_refined = net.apply_refine(coarse_density, stride4, single)
    changed = (single_refined - coarse_density).abs().sum(dim=1)[0] > 1e-7
    expected = torch.zeros(64, 64, dtype=torch.bool, device=device)
    expected[:4, :4] = True
    assert torch.equal(changed, expected), "Unselected density cells changed."

    # Batched selection: 64 windows should be sent through RefineNet together.
    with torch.no_grad():
        refined = net.apply_refine(
            encoded["density"], encoded["stride4_features"], mask
        )
    assert refined.shape == (1, 1, 64, 64)
    assert torch.isfinite(refined).all()
    assert (refined >= 0).all()

    # Gradient-flow test: coarse frozen, ErrorHead and RefineNet trainable.
    net.train()
    net.coarse.eval()
    for parameter in net.coarse.parameters():
        parameter.requires_grad = False

    image = torch.rand(1, 3, 512, 512, device=device)
    gt = torch.rand(1, 1, 64, 64, device=device)
    outputs = net.encode(image)
    train_mask = select_topk(outputs["error_scores"], 0.10)
    refined_density = net.apply_refine(
        outputs["density"], outputs["stride4_features"], train_mask
    )
    loss = (outputs["error_scores"].mean()
            + torch.nn.functional.l1_loss(refined_density, gt))
    loss.backward()

    error_grads = [
        p.grad for p in net.error_head.parameters() if p.requires_grad
    ]
    refiner_grads = [
        p.grad for p in net.refiner.parameters() if p.requires_grad
    ]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in error_grads), "ErrorHead has no nonzero gradients."
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in refiner_grads), "RefineNet has no nonzero gradients."
    assert all(p.grad is None for p in net.coarse.parameters()), (
        "Coarse counter unexpectedly received gradients."
    )

    print("Shapes and top-k selection: PASS")
    print("Batched selected-window refinement: PASS")
    print("Unselected density cells unchanged: PASS")
    print("ErrorHead gradient flow: PASS")
    print("RefineNet gradient flow: PASS")
    print("Coarse counter frozen (no gradients): PASS")
    print("All Phase-B tests passed.")


if __name__ == "__main__":
    main()
