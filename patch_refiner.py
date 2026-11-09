
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class PatchRefiner(nn.Module):
    """
    High-resolution patch refinement network.

    Input:  [B, 3, 160, 160]
    Output: [B, 1, 10, 10]
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            block(3, 16),
            nn.MaxPool2d(2),       # 160 -> 80

            block(16, 24),
            nn.MaxPool2d(2),       # 80 -> 40

            block(24, 32),
            nn.MaxPool2d(2),       # 40 -> 20

            block(32, 48),
            nn.MaxPool2d(2),       # 20 -> 10

            block(48, 48),
        )

        self.regression = nn.Sequential(
            block(48, 32),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.features(x)
        return self.regression(x)
