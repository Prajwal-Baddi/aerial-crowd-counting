
import torch
import torch.nn as nn


def enable_mc_dropout(model):

    # Disable BatchNorm training behavior
    model.eval()

    # Enable only dropout layers
    for module in model.modules():

        if isinstance(
            module,
            (nn.Dropout, nn.Dropout2d, nn.Dropout3d)
        ):
            module.train()


@torch.no_grad()
def estimate_uncertainty(
    model,
    image,
    num_samples=30
):

    if image.shape[0] != 1:
        raise ValueError(
            "MC uncertainty estimation currently expects batch size 1."
        )

    enable_mc_dropout(model)

    predictions = []

    for _ in range(num_samples):

        prediction = model(image)

        predictions.append(prediction)

    # [num_samples, batch, channel, height, width]
    predictions = torch.stack(
        predictions,
        dim=0
    )

    # Average density prediction
    mean_density = predictions.mean(dim=0)

    # Pixel-wise epistemic uncertainty estimate
    variance_map = predictions.var(
        dim=0,
        unbiased=False
    )

    # Count for each stochastic prediction
    predicted_counts = predictions.sum(
        dim=(2, 3, 4)
    ).squeeze(1)

    mean_count = predicted_counts.mean()

    count_variance = predicted_counts.var(
        unbiased=False
    )

    # Restore normal evaluation mode
    model.eval()

    return (
        mean_density,
        variance_map,
        mean_count,
        count_variance
    )
