
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ErrorAwareCrowdDataset
from model import CoarseCounter


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = ErrorAwareCrowdDataset(
        "error_aware/processed/train_balanced.jsonl"
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = CoarseCounter(
        pretrained=True,
        freeze_early=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and "backbone" in n
                ],
                "lr": 1e-5
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and "backbone" not in n
                ],
                "lr": 1e-4
            }
        ],
        weight_decay=1e-4
    )

    final_conv = model.density_head[-2]

    model.train()

    print("Initial bias:", final_conv.bias.item())

    for index, (images, densities, counts) in enumerate(loader):

        if index >= 10:
            break

        if counts.item() == 0:
            continue

        images = images.to(device)
        densities = densities.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        prediction = outputs["density"]

        loss = (
            F.mse_loss(
                prediction.float() * 10,
                densities.float() * 10
            )
            + 0.01 * F.l1_loss(
                prediction.float().sum(dim=(1, 2, 3)),
                counts.to(device).float().view(-1)
            )
        )

        loss.backward()

        bias_before = final_conv.bias.item()
        grad_before = final_conv.bias.grad.item()

        optimizer.step()

        bias_after = final_conv.bias.item()

        with torch.no_grad():
            outputs_after = model(images)
            density_after = outputs_after["density"]

            hidden = model.stride4_backbone(images)
            hidden = model.stride8_backbone(hidden)
            hidden = model.context_module(hidden)

            pre_activation = final_conv(
                model.density_head[1](
                    model.density_head[0](hidden)
                )
            )

            positive_percentage = (
                (pre_activation > 0).float().mean().item() * 100
            )

            predicted_count = density_after.sum().item()

        print(f"\nUpdate {index}")
        print("Target count:", counts.item())
        print("Loss:", loss.item())
        print("Bias before:", bias_before)
        print("Bias gradient:", grad_before)
        print("Bias after:", bias_after)
        print("Predicted count:", predicted_count)
        print("Positive pre-activations:", positive_percentage)

        if predicted_count == 0:
            print("\nZERO-PREDICTION COLLAPSE DETECTED.")
            break

    print("\nDiagnostic finished. No checkpoint was saved.")


if __name__ == "__main__":
    main()
