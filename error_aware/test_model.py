
import torch

from model import CoarseCounter


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    # Load pretrained VGG16 weights.
    model = CoarseCounter(pretrained=True).to(device)
    model.eval()

    # One 512x512 image.
    dummy_image = torch.randn(
        1, 3, 512, 512,
        device=device,
    )

    # Mixed precision reduces inference memory consumption.
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                outputs = model(dummy_image)
        else:
            outputs = model(dummy_image)

    density = outputs["density"]
    context = outputs["context"]
    stride4 = outputs["stride4_features"]

    print("\nOutput shapes:")
    print("Input:", tuple(dummy_image.shape))
    print("Stride-4 features:", tuple(stride4.shape))
    print("Context features:", tuple(context.shape))
    print("Coarse density:", tuple(density.shape))

    assert stride4.shape == (1, 256, 128, 128)
    assert context.shape == (1, 128, 64, 64)
    assert density.shape == (1, 1, 64, 64)

    assert torch.isfinite(density).all()
    assert density.min().item() >= 0

    predicted_count = density.float().sum().item()

    print("\nPredicted count:", predicted_count)
    print("Density minimum:", density.min().item())
    print("Density maximum:", density.max().item())

    if device.type == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )

        print(f"Peak GPU memory: {peak_memory:.3f} GB")

    print("\nAll coarse counter architecture checks passed.")


if __name__ == "__main__":
    main()
