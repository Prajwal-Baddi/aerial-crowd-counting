import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

CELL = 4
DEN_SCALE = 1.0


def cell_sum(density, cell_size=CELL):
    if density.ndim != 4 or density.shape[-2] % cell_size or density.shape[-1] % cell_size:
        raise ValueError("density must be [B,C,H,W] with spatial sizes divisible by cell_size")
    return F.avg_pool2d(density, cell_size, cell_size) * (cell_size * cell_size)


def select_topk(scores, budget):
    if not 0.0 <= float(budget) <= 1.0:
        raise ValueError("budget must be between 0 and 1")
    if scores.ndim == 3:
        scores = scores.unsqueeze(1)
    if scores.ndim != 4 or scores.shape[1] != 1:
        raise ValueError("scores must have shape [B,1,H,W]")
    b, _, h, w = scores.shape
    total = h * w
    k = int(round(total * float(budget)))
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if k == 0:
        return mask
    if k >= total:
        return torch.ones_like(mask)
    indices = scores.flatten(1).topk(k, dim=1).indices
    mask.flatten(1).scatter_(1, indices, True)
    return mask


def mask_to_pixels(mask, scale=4):
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B,1,H,W]")
    return mask.repeat_interleave(scale, -2).repeat_interleave(scale, -1)


class MultiScaleContext(nn.Module):
    def __init__(self, in_channels=512, branch_channels=128):
        super().__init__()
        self.dilation_1 = nn.Conv2d(in_channels, branch_channels, 3, padding=1, dilation=1)
        self.dilation_2 = nn.Conv2d(in_channels, branch_channels, 3, padding=2, dilation=2)
        self.dilation_4 = nn.Conv2d(in_channels, branch_channels, 3, padding=4, dilation=4)
        self.activation = nn.ReLU(inplace=True)
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 3, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branches = [
            self.activation(self.dilation_1(x)),
            self.activation(self.dilation_2(x)),
            self.activation(self.dilation_4(x)),
        ]
        return self.fusion(torch.cat(branches, dim=1))


class CoarseCounter(nn.Module):
    """Stage-1 architecture. Keep module names unchanged for checkpoint compatibility."""
    def __init__(self, pretrained=True, freeze_early=True):
        super().__init__()
        weights = VGG16_Weights.DEFAULT if pretrained else None
        backbone = vgg16(weights=weights)
        self.stride4_backbone = nn.Sequential(*list(backbone.features.children())[:16])
        self.stride8_backbone = nn.Sequential(*list(backbone.features.children())[16:23])
        self.context_module = MultiScaleContext(512, 128)
        self.density_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Softplus(beta=1000, threshold=20),
        )
        final_conv = self.density_head[-2]
        nn.init.normal_(final_conv.weight, mean=0.0, std=1e-5)
        nn.init.constant_(final_conv.bias, -0.00225)
        self.freeze_early = freeze_early
        if freeze_early:
            for layer in list(self.stride4_backbone.children())[:10]:
                for parameter in layer.parameters():
                    parameter.requires_grad = False

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected input [B,3,H,W]")
        if image.shape[-2] % 32 or image.shape[-1] % 32:
            raise ValueError("Input height and width must be divisible by 32")
        stride4_features = self.stride4_backbone(image)
        stride8_features = self.stride8_backbone(stride4_features)
        context = self.context_module(stride8_features)
        density = self.density_head(context)
        return {
            "density": density,
            "context": context,
            "stride4_features": stride4_features,
        }


class ErrorHead(nn.Module):
    """Predict absolute count error on a 16x16 grid for 512x512 crops."""
    def __init__(self, context_channels=128, hidden_channels=64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(context_channels + 1, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.error_predictor = nn.Sequential(
            nn.Conv2d(33, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Softplus(),
        )

    def forward(self, context, coarse_density):
        context = context.detach()
        coarse_density = coarse_density.detach()
        if context.shape[-2:] != coarse_density.shape[-2:]:
            raise ValueError("Context and density spatial dimensions must match")
        features = self.feature_extractor(torch.cat([context, coarse_density], 1))
        pooled = F.avg_pool2d(features, CELL, CELL)
        coarse_cell_counts = cell_sum(coarse_density, CELL).detach()
        return self.error_predictor(torch.cat([pooled, coarse_cell_counts], 1))


class RefineNet(nn.Module):
    """96x96 crop represented by 24x24 stride-4 features."""
    def __init__(self, feature_channels=256, width=96):
        super().__init__()
        self.feature_reduce = nn.Conv2d(feature_channels, width, 1)
        self.body = nn.Sequential(
            nn.Conv2d(width + 1, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, stride4_window, coarse_density_window):
        features = self.feature_reduce(stride4_window)
        return self.body(torch.cat([features, coarse_density_window], 1))


class ESRNet(nn.Module):
    def __init__(self, pretrained=False, freeze_early=True, refine_width=96):
        super().__init__()
        self.coarse = CoarseCounter(pretrained, freeze_early)
        self.error_head = ErrorHead()
        self.refiner = RefineNet(width=refine_width)

    def encode(self, image):
        coarse = self.coarse(image)
        scores = self.error_head(coarse["context"], coarse["density"])
        return {**coarse, "error_scores": scores}

    def apply_refine(self, coarse_density, stride4_features, mask):
        if mask.shape != (coarse_density.shape[0], 1, 16, 16):
            raise ValueError("For 512px inputs, mask must have shape [B,1,16,16]")
        if coarse_density.shape[-2:] != (64, 64):
            raise ValueError("Expected coarse density [B,1,64,64]")
        if stride4_features.shape[-2:] != (128, 128):
            raise ValueError("Expected stride-4 features [B,256,128,128]")

        refined = coarse_density.clone()
        selected = mask[:, 0].bool().nonzero(as_tuple=False)
        if selected.numel() == 0:
            return refined

        # Distribute each stride-8 density value over its 2x2 stride-4 cells.
        density_s4 = F.interpolate(
            coarse_density / 4.0, scale_factor=2,
            mode="bilinear", align_corners=False
        )

        for batch_idx, cell_y, cell_x in selected.tolist():
            cy, cx = cell_y * 8, cell_x * 8
            y0, x0 = cy - 8, cx - 8
            y1, x1 = y0 + 24, x0 + 24

            ya, yb = max(0, y0), min(128, y1)
            xa, xb = max(0, x0), min(128, x1)
            feat = stride4_features[batch_idx:batch_idx+1, :, ya:yb, xa:xb]
            den = density_s4[batch_idx:batch_idx+1, :, ya:yb, xa:xb]

            feat = F.pad(feat, (max(0, -x0), max(0, x1-128),
                                max(0, -y0), max(0, y1-128)))
            den = F.pad(den, (max(0, -x0), max(0, x1-128),
                              max(0, -y0), max(0, y1-128)))

            residual_s4 = self.refiner(feat, den)
            residual_s8 = F.avg_pool2d(residual_s4, 2, 2) * 4.0
            central = residual_s8[:, :, 4:8, 4:8]

            dy, dx = cell_y * 4, cell_x * 4
            refined[batch_idx:batch_idx+1, :, dy:dy+4, dx:dx+4] = (
                refined[batch_idx:batch_idx+1, :, dy:dy+4, dx:dx+4] + central
            )

        return refined.clamp_min(0.0)

    def forward(self, image, budget=0.25, selection_mask=None):
        outputs = self.encode(image)
        if selection_mask is None:
            selection_mask = select_topk(outputs["error_scores"], budget)
        final_density = self.apply_refine(
            outputs["density"], outputs["stride4_features"], selection_mask
        )
        return {
            **outputs,
            "selection_mask": selection_mask,
            "final_density": final_density,
        }
