import torch
import torch.nn.functional as F

from data import ErrorAwareCrowdDataset
from model import CoarseCounter


def inspect_sample(model, dataset, index, device):
    image, target, count = dataset[index]

    image = image.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)

    captured = {}

    final_conv = model.density_head[-2]

    def capture_output(module, inputs, output):
        captured["pre_activation"] = output

    hook = final_conv.register_forward_hook(capture_output)

    model.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(device.type == "cuda"),
    ):
        output = model(image)
        prediction = output["density"]

    hook.remove()

    pre_activation = captured["pre_activation"].float()
    prediction = prediction.float()
    target = target.float()

    density_loss = F.mse_loss(prediction * 10, target * 10)
    count_loss = (
        prediction.sum() - float(count)
    ).abs()

    loss = density_loss + 0.01 * count_loss
    loss.backward()

    weight_grad = final_conv.weight.grad
    bias_grad = final_conv.bias.grad

    print("\n" + "=" * 55)
    print(f"Sample index: {index}")
    print(f"Ground-truth count: {int(count)}")
    print(f"Total predicted count: {prediction.sum().item():.6f}")

    print("\nFinal convolution pre-activation:")
    print(f"Minimum: {pre_activation.min().item():.8f}")
    print(f"Maximum: {pre_activation.max().item():.8f}")
    print(f"Mean:    {pre_activation.mean().item():.8f}")

    print("\nFinal density output:")
    print(f"Minimum: {prediction.min().item():.8f}")
    print(f"Maximum: {prediction.max().item():.8f}")
    print(f"Mean:    {prediction.mean().item():.8f}")

    zero_fraction = (
        (prediction == 0).float().mean().item() * 100
    )

    print(f"Zero density cells: {zero_fraction:.2f}%")

    print("\nLoss:")
    print(f"Density loss: {density_loss.item():.8f}")
    print(f"Count loss:   {count_loss.item():.8f}")
    print(f"Total loss:   {loss.item():.8f}")

    print("\nFinal convolution gradients:")

    if weight_grad is not None:
        print(f"Weight gradient mean: {weight_grad.abs().mean().item():.10e}")
        print(f"Weight gradient max:  {weight_grad.abs().max().item():.10e}")
    else:
        print("Weight gradient: None")

    if bias_grad is not None:
        print(f"Bias gradient: {bias_grad.item():.10e}")
    else:
        print("Bias gradient: None")

    print("=" * 55)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ErrorAwareCrowdDataset(
        "error_aware/processed/train.jsonl"
    )

    empty_index = None
    nonempty_index = None

    print("Searching for empty and non-empty crops...")

    for i in range(len(dataset)):
        _, _, count = dataset[i]

        if int(count) == 0 and empty_index is None:
            empty_index = i

        if int(count) > 0:
            nonempty_index = i
            break

    if empty_index is None or nonempty_index is None:
        raise RuntimeError("Could not find both sample types.")

    print(f"Empty crop index: {empty_index}")
    print(f"Non-empty crop index: {nonempty_index}")

    model = CoarseCounter(pretrained=True).to(device)
    model.train()

    inspect_sample(model, dataset, empty_index, device)
    inspect_sample(model, dataset, nonempty_index, device)


if __name__ == "__main__":
    main()