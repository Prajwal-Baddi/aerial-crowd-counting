import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

CELL = 4
GAIN_MAX = 3.0            # refined cell may scale coarse density by a factor in (0, GAIN_MAX)
ADD_BIAS_INIT = 0.0
ADD_SCALE = 2.5e-4     # near-zero additive mass; limits accumulation across a full 512px crop
DENSITY_INPUT_SCALE = 10.0  # coarse density per pixel is ~0.01, so scale it up for the refiner input


def cell_sum(density, cell_size=CELL):
    if density.ndim != 4 or density.shape[-2] % cell_size or density.shape[-1] % cell_size:
        raise ValueError("density must be [B,C,H,W] with spatial sizes divisible by cell_size")
    return F.avg_pool2d(density, cell_size, cell_size) * (cell_size * cell_size)


def select_topk(scores, budget, min_score=None):
    """Top-`budget` fraction of cells per image. If `min_score` is given, cells whose predicted
    error is below it are never selected (so empty / easy crops select nothing)."""
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
        mask = torch.ones_like(mask)
    else:
        indices = scores.flatten(1).topk(k, dim=1).indices
        mask.flatten(1).scatter_(1, indices, True)
    if min_score is not None:
        mask = mask & (scores >= float(min_score))
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
    """Stage-1 architecture; module names preserved for checkpoint compatibility. (UNCHANGED)"""

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
        return {"density": density, "context": context, "stride4_features": stride4_features}


class ErrorHead(nn.Module):
    """(UNCHANGED)"""

    def __init__(self, context_channels=128, hidden_channels=64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(context_channels + 1, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.error_predictor = nn.Sequential(
            nn.Conv2d(33, 32, 1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1), nn.Softplus(),
        )

    def forward(self, context, coarse_density):
        context = context.detach()
        coarse_density = coarse_density.detach()
        if context.shape[-2:] != coarse_density.shape[-2:]:
            raise ValueError("Context and density spatial sizes must match")
        features = self.feature_extractor(torch.cat([context, coarse_density], 1))
        pooled = F.avg_pool2d(features, CELL, CELL)
        counts = cell_sum(coarse_density, CELL).detach()
        return self.error_predictor(torch.cat([pooled, counts], 1))


class RefineNet(nn.Module):
    """Predicts, for the central 32x32 px cell of every 96x96 px window:
         gain in (0, GAIN_MAX)   -> multiplicative correction of the coarse density
         add  >= 0               -> additive density (new people the coarse model missed)
       D_final = D_coarse * gain + add. Non-negative by construction => no clamp, no asymmetry.
       Initialised to the identity (gain = 1, add ~ 0)."""

    def __init__(self, feature_channels=256, width=96):
        super().__init__()
        self.feature_reduce = nn.Conv2d(feature_channels, width, 1)
        self.body = nn.Sequential(
            nn.Conv2d(width + 1, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, 2, 1),      # channel 0: gain logit, channel 1: additive logit
        )
        final = self.body[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)   # small but not vanishing gradients
        with torch.no_grad():
            final.bias[0] = -math.log(GAIN_MAX - 1.0)       # sigmoid(.)*GAIN_MAX == 1
            final.bias[1] = ADD_BIAS_INIT

    def forward(self, stride4_windows, coarse_density_windows):
        features = self.feature_reduce(stride4_windows)
        density = (coarse_density_windows * DENSITY_INPUT_SCALE).to(features.dtype)
        x = torch.cat([features, density], dim=1)
        for layer in self.body[:-1]:
            x = layer(x)
        final = self.body[-1]
        # Final projection and non-linearities in FP32 (FP16 bias-gradient overflow, as before).
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = F.conv2d(x.float(), final.weight.float(), final.bias.float(),
                           stride=final.stride, padding=final.padding,
                           dilation=final.dilation, groups=final.groups)      # [N,2,24,24]
            gain_logit = F.avg_pool2d(out[:, :1], 2, 2)                       # [N,1,12,12]
            add_s8 = (
    F.avg_pool2d(F.relu(out[:, 1:]), 2, 2)
    * 4.0
    * ADD_SCALE
)         # sum-pool s4 -> s8
            gain = GAIN_MAX * torch.sigmoid(gain_logit)
        return gain[:, :, 4:8, 4:8], add_s8[:, :, 4:8, 4:8]                   # central cell [N,1,4,4]


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

    @staticmethod
    def _extract_selected_windows(features, density_s4, selected):
        """Selected 24x24 stride-4 windows as a batch (unfold is a view; only selected are copied)."""
        features_pad = F.pad(features, (8, 8, 8, 8))
        density_pad = F.pad(density_s4, (8, 8, 8, 8))
        feature_windows = features_pad.unfold(2, 24, 8).unfold(3, 24, 8)
        density_windows = density_pad.unfold(2, 24, 8).unfold(3, 24, 8)
        batch_idx, cell_y, cell_x = selected.unbind(dim=1)
        selected_features = feature_windows[batch_idx, :, cell_y, cell_x].contiguous()
        selected_density = density_windows[batch_idx, :, cell_y, cell_x].contiguous()
        return selected_features, selected_density, batch_idx, cell_y, cell_x

    def apply_refine(self, coarse_density, stride4_features, mask,
                     clamp_output=None, return_aux=False):
        """`clamp_output` is accepted for backward compatibility and ignored: the output is
        non-negative by construction, so training and inference use the identical function."""
        if mask.ndim != 4 or mask.shape[1:] != (1, 16, 16):
            raise ValueError("For 512px crops, mask must be [B,1,16,16]")
        if coarse_density.shape[1:] != (1, 64, 64):
            raise ValueError("Expected coarse density [B,1,64,64]")
        if stride4_features.shape[1:] != (256, 128, 128):
            raise ValueError("Expected stride-4 features [B,256,128,128]")

        selected = mask[:, 0].bool().nonzero(as_tuple=False)
        if selected.numel() == 0:
            return (coarse_density.clone(), None) if return_aux else coarse_density.clone()

        density_s4 = F.interpolate(coarse_density / 4.0, scale_factor=2,
                                   mode="bilinear", align_corners=False)
        feat_w, den_w, b, cy, cx = self._extract_selected_windows(
            stride4_features, density_s4, selected)

        gain, add = self.refiner(feat_w, den_w)                       # [N,1,4,4] each (fp32)

        # Coarse density of each selected cell, [N,1,4,4].
        cells = coarse_density.float().unfold(2, CELL, CELL).unfold(3, CELL, CELL)  # [B,1,16,16,4,4]
        coarse_cells = cells[b, 0, cy, cx].unsqueeze(1)
        final_cells = coarse_cells * gain + add                       # >= 0 by construction

        # Vectorised scatter of the refined cells (no Python loop, no in-place accumulation).
        ar = torch.arange(CELL, device=coarse_density.device)
        rows = (cy * CELL)[:, None] + ar[None]
        cols = (cx * CELL)[:, None] + ar[None]
        refined = coarse_density.clone().to(final_cells.dtype)
        refined[b[:, None, None], 0, rows[:, :, None], cols[:, None, :]] = final_cells[:, 0]
        if return_aux:
            return refined, {"gain": gain, "add": add, "selected": selected,
                             "coarse_cells": coarse_cells, "final_cells": final_cells}
        return refined

    def forward(self, image, budget=0.25, selection_mask=None, min_score=None, return_aux=False):
        outputs = self.encode(image)
        if selection_mask is None:
            selection_mask = select_topk(outputs["error_scores"], budget, min_score)
        result = self.apply_refine(outputs["density"], outputs["stride4_features"],
                                   selection_mask, return_aux=return_aux)
        final_density, aux = result if return_aux else (result, None)
        out = {**outputs, "selection_mask": selection_mask, "final_density": final_density}
        if return_aux:
            out["refine_aux"] = aux
        return out