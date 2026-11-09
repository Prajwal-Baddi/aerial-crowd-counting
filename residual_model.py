import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ResidualPatchRefiner(nn.Module):
    """
    Predicts a signed density correction.

    Inputs:
      image_crop: [B, 3, 160, 160], values in [0, 1]
      coarse_patch: [B, 1, 10, 10]
    Output:
      signed residual: [B, 1, 10, 10]

    The final layer is linear because residuals may be positive or negative.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(4, 16),
            nn.MaxPool2d(2),  # 160 -> 80
            conv_block(16, 24),
            nn.MaxPool2d(2),  # 80 -> 40
            conv_block(24, 32),
            nn.MaxPool2d(2),  # 40 -> 20
            conv_block(32, 48),
            nn.MaxPool2d(2),  # 20 -> 10
            conv_block(48, 48),
        )
        self.regression = nn.Sequential(
            conv_block(48, 32),
            nn.Conv2d(32, 1, kernel_size=1),  # No Softplus: signed residual.
        )

    def forward(self, image_crop, coarse_patch):
        if image_crop.ndim != 4 or image_crop.shape[1:] != (3, 160, 160):
            raise ValueError(
                f"Expected image_crop [B,3,160,160], got {tuple(image_crop.shape)}"
            )
        if coarse_patch.ndim != 4 or coarse_patch.shape[1:] != (1, 10, 10):
            raise ValueError(
                f"Expected coarse_patch [B,1,10,10], got {tuple(coarse_patch.shape)}"
            )
        if image_crop.shape[0] != coarse_patch.shape[0]:
            raise ValueError("Image and coarse-patch batch sizes do not match.")

        coarse_up = F.interpolate(
            coarse_patch, size=(160, 160), mode="bilinear", align_corners=False
        )
        combined = torch.cat([image_crop, coarse_up], dim=1)
        return self.regression(self.features(combined))
