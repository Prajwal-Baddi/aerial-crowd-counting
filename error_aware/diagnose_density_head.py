
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ErrorAwareCrowdDataset
from model import CoarseCounter


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

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

    model.eval()

    final_conv = model.density_head[-2]

    print("\nFinal convolution bias:")
    print(final_conv.bias.detach().cpu())

    for index, (images, densities, counts) in enumerate(loader):

        if index >= 8:
            break

        images = images.to(device)
        densities = densities.to(device)

        with torch.no_grad():

            # FP32 diagnostic
            features4 = model.stride4_backbone(images)
            features8 = model.stride8_backbone(features4)
            context = model.context_module(features8)

            hidden = model.density_head[0](context)
            hidden = model.density_head[1](hidden)

            pre_activation = final_conv(hidden)
            fp32_density = F.relu(pre_activation)

            # FP16 diagnostic
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                fp16_output = model(images)["density"]

        target_count = counts.item()
        predicted_count = fp32_density.sum().item()
        fp16_count = fp16_output.float().sum().item()

        positive_percentage = (
            (pre_activation > 0).float().mean().item() * 100
        )

        print(f"\nSample {index}")
        print("Target count:", target_count)
        print("FP32 predicted count:", predicted_count)
        print("FP16 predicted count:", fp16_count)

        print(
            "Pre-activation min/max:",
            pre_activation.min().item(),
            pre_activation.max().item()
        )

        print(
            "Positive pre-activation percentage:",
            positive_percentage
        )

        print(
            "FP32 density min/max:",
            fp32_density.min().item(),
            fp32_density.max().item()
        )

    # Gradient diagnostic on one non-empty crop
    model.train()
    model.zero_grad(set_to_none=True)

    for images, densities, counts in loader:

        if counts.item() > 0:
            images = images.to(device)
            densities = densities.to(device)

            output = model(images)["density"]

            loss = F.mse_loss(
                output.float() * 10,
                densities.float() * 10
            )

            loss.backward()

            print("\nGradient diagnostic")
            print("Target count:", counts.item())
            print("Loss:", loss.item())

            print(
                "Final weight gradient mean:",
                final_conv.weight.grad.abs().mean().item()
            )

            print(
                "Final bias gradient:",
                final_conv.bias.grad.item()
            )

            break

    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()
