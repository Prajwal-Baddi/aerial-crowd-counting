
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class CrowdCounter(nn.Module):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        # Initial feature extraction: 320 -> 160
        self.stem = nn.Sequential(
            conv_block(3, 16),
            nn.MaxPool2d(kernel_size=2)
        )

        # Fine-scale branch: 160 -> 80
        self.fine_branch = nn.Sequential(
            conv_block(16, 24),
            nn.MaxPool2d(kernel_size=2),
            conv_block(24, 24)
        )

        # Medium-scale branch: 80 -> 40
        self.medium_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            conv_block(24, 32),
            conv_block(32, 32)
        )

        # Coarse-scale branch: 40 -> 20
        self.coarse_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            conv_block(32, 48),
            conv_block(48, 48)
        )

        # Feature fusion at 80 x 80
        self.fusion = nn.Sequential(
            conv_block(24 + 32 + 48, 48),
            nn.Dropout2d(p=dropout_rate),
            conv_block(48, 24),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(24, 1, kernel_size=1)
        )

    def forward(self, x):

        x = self.stem(x)

        fine = self.fine_branch(x)

        medium = self.medium_branch(fine)

        coarse = self.coarse_branch(medium)

        # Bring medium and coarse features to 80 x 80
        medium_up = F.interpolate(
            medium,
            size=fine.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        coarse_up = F.interpolate(
            coarse,
            size=fine.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Combine all three scales
        combined = torch.cat(
            [fine, medium_up, coarse_up],
            dim=1
        )

        density_map = self.fusion(combined)

        # Density values must be non-negative
        density_map = F.softplus(density_map)

        return density_map
