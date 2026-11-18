"""Gated multiplicative-additive refinement with a 3-way ErrorHead.  (new file; model_multiplicative.py untouched)

ErrorHead   -> per 32x32 px cell: P(fine), P(needs increase: miss/under), P(needs decrease: false positive/over)
RefineNet   -> gain in (1/8, 8) in log space (exactly 1 at init)  +  additive density gated by P(needs increase)
D_final(cell) = D_coarse * gain + add * P(increase)     (non-negative by construction, no clamp)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

CELL = 4
NUM_CLASSES = 3                      # 0 fine | 1 needs increase | 2 needs decrease
GAIN_LOG_RANGE = math.log(8.0)       # gain in (1/8, 8): covers the 95th percentile (7.9x) of under-counted cells
ADD_CAP = 0.2                        # max additive density per stride-8 pixel before gating (4x4 cell: 3.2 counts)
ADD_BIAS_INIT = -7.0                 # sigmoid(-7)*0.2 ~ 1.8e-4 per pixel at init, further multiplied by P(increase)
DENSITY_INPUT_SCALE = 10.0


def cell_sum(density, cell_size=CELL):
    if density.ndim != 4 or density.shape[-2] % cell_size or density.shape[-1] % cell_size:
        raise ValueError("density must be [B,C,H,W] with spatial sizes divisible by cell_size")
    return F.avg_pool2d(density, cell_size, cell_size) * (cell_size * cell_size)


def select_topk(scores, budget, min_score=None):
    """Top-`budget` fraction of cells per image; cells scoring below `min_score` are never selected."""
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
    """3-way cell classifier. Unlike the coarse density head it also sees the stride-4 VGG features, because
    missed people (coarse density ~ 0) can only be found from finer evidence. All inputs are detached."""

    def __init__(self, context_channels=128, stride4_channels=256, hidden_channels=64):
        super().__init__()
        self.stride4 = nn.Sequential(
            nn.Conv2d(stride4_channels, 32, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True))      # stride 4 -> stride 8
        self.trunk = nn.Sequential(
            nn.Conv2d(context_channels + 1 + 32, hidden_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 48, 3, padding=2, dilation=2), nn.ReLU(inplace=True))
        # avg- and max-pooled trunk features (2*48) + coarse cell count + coarse max density
        self.classifier = nn.Sequential(
            nn.Conv2d(2 * 48 + 2, 64, 1), nn.ReLU(inplace=True), nn.Conv2d(64, NUM_CLASSES, 1))
        with torch.no_grad():                                       # prior ~ the 0.3% base rate of material errors
            self.classifier[-1].bias.copy_(torch.tensor([0.0, -5.0, -5.0]))

    def forward(self, context, coarse_density, stride4_features):
        ctx, dens, f4 = context.detach(), coarse_density.detach(), stride4_features.detach()
        f4s = self.stride4(f4)
        if f4s.shape[-2:] != ctx.shape[-2:] or ctx.shape[-2:] != dens.shape[-2:]:
            raise ValueError("Context, density and stride-4 features must align at stride 8")
        x = self.trunk(torch.cat([ctx, dens.to(ctx.dtype) * DENSITY_INPUT_SCALE, f4s], 1))
        pooled = torch.cat([F.avg_pool2d(x, CELL), F.max_pool2d(x, CELL)], 1)
        stats = torch.cat([cell_sum(dens.float()), F.max_pool2d(dens.float(), CELL) * DENSITY_INPUT_SCALE], 1)
        return self.classifier(torch.cat([pooled, stats.to(pooled.dtype)], 1))   # logits [B,3,H/32,W/32]


class RefineNet(nn.Module):
    """For the central 32x32 px cell of every 96x96 px window predict:
         gain = exp(L*tanh(s))                 in (1/8, 8); exactly 1 at init (final weights are zero)
         add  = ADD_CAP * sigmoid(a)           per stride-8 pixel, gated outside by P(needs increase)"""

    def __init__(self, feature_channels=256, width=96):
        super().__init__()
        self.feature_reduce = nn.Conv2d(feature_channels, width, 1)
        self.body = nn.Sequential(
            nn.Conv2d(width + 1 + NUM_CLASSES, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, 2, 1),               # channel 0: gain logit, channel 1: additive logit
        )
        final = self.body[-1]
        nn.init.zeros_(final.weight)              # exact identity at init; weights move after the first update
        with torch.no_grad():
            final.bias[0] = 0.0
            final.bias[1] = ADD_BIAS_INIT

    def forward(self, stride4_windows, coarse_density_windows, prob_windows):
        features = self.feature_reduce(stride4_windows)
        extra = torch.cat([coarse_density_windows * DENSITY_INPUT_SCALE, prob_windows], 1).to(features.dtype)
        x = torch.cat([features, extra], dim=1)
        for layer in self.body[:-1]:
            x = layer(x)
        final = self.body[-1]
        with torch.autocast(device_type=x.device.type, enabled=False):         # final projection in FP32
            out = F.conv2d(x.float(), final.weight.float(), final.bias.float(),
                           stride=final.stride, padding=final.padding,
                           dilation=final.dilation, groups=final.groups)      # [N,2,24,24]
            s = F.avg_pool2d(out[:, :1], 2, 2)                                # [N,1,12,12]
            gain = torch.exp(GAIN_LOG_RANGE * torch.tanh(s))
            add_s8 = ADD_CAP * F.avg_pool2d(torch.sigmoid(out[:, 1:]), 2, 2)  # density per stride-8 pixel
        return gain[:, :, 4:8, 4:8], add_s8[:, :, 4:8, 4:8]                   # central cell [N,1,4,4]


class ESRNet(nn.Module):
    def __init__(self, pretrained=False, freeze_early=True, refine_width=96):
        super().__init__()
        self.coarse = CoarseCounter(pretrained, freeze_early)
        self.error_head = ErrorHead()
        self.refiner = RefineNet(width=refine_width)

    def encode(self, image):
        coarse = self.coarse(image)
        logits = self.error_head(coarse["context"], coarse["density"], coarse["stride4_features"])
        probs = logits.float().softmax(dim=1)                                  # [B,3,16,16]
        return {**coarse, "error_logits": logits, "error_probs": probs,
                "error_scores": 1.0 - probs[:, :1]}                            # P(material error), [B,1,16,16]

    @staticmethod
    def _windows(x, b, cy, cx):
        """24x24 (stride-4 grid) windows centred on the selected 32x32 px cells -> [N,C,24,24]."""
        xp = F.pad(x, (8, 8, 8, 8))
        return xp.unfold(2, 24, 8).unfold(3, 24, 8)[b, :, cy, cx].contiguous()

    def apply_refine(self, coarse_density, stride4_features, mask, probs=None, return_aux=False):
        """probs: [B,3,16,16] class probabilities from the ErrorHead (or one-hot teacher labels in training).
        None -> 'everything is fine', i.e. the additive branch is gated fully off."""
        if mask.ndim != 4 or mask.shape[1:] != (1, 16, 16):
            raise ValueError("For 512px crops, mask must be [B,1,16,16]")
        if coarse_density.shape[1:] != (1, 64, 64):
            raise ValueError("Expected coarse density [B,1,64,64]")
        if stride4_features.shape[1:] != (256, 128, 128):
            raise ValueError("Expected stride-4 features [B,256,128,128]")
        selected = mask[:, 0].bool().nonzero(as_tuple=False)
        if selected.numel() == 0:
            return (coarse_density.clone(), None) if return_aux else coarse_density.clone()
        if probs is None:
            probs = torch.zeros(mask.shape[0], NUM_CLASSES, 16, 16, device=mask.device)
            probs[:, 0] = 1.0
        probs = probs.float().detach()
        b, cy, cx = selected.unbind(dim=1)

        density_s4 = F.interpolate(coarse_density / 4.0, scale_factor=2, mode="bilinear", align_corners=False)
        prob_s4 = F.interpolate(probs, scale_factor=8, mode="nearest")        # same grid as the stride-4 features
        gain, add_raw = self.refiner(self._windows(stride4_features, b, cy, cx),
                                     self._windows(density_s4, b, cy, cx),
                                     self._windows(prob_s4, b, cy, cx))       # [N,1,4,4] each (fp32)
        p_inc = probs[b, 1, cy, cx].view(-1, 1, 1, 1)
        add = add_raw * p_inc                                                  # gate: add only where an increase is expected

        cells = coarse_density.float().unfold(2, CELL, CELL).unfold(3, CELL, CELL)    # [B,1,16,16,4,4]
        coarse_cells = cells[b, 0, cy, cx].unsqueeze(1)
        final_cells = coarse_cells * gain + add                                # >= 0 by construction

        ar = torch.arange(CELL, device=coarse_density.device)
        rows, cols = (cy * CELL)[:, None] + ar[None], (cx * CELL)[:, None] + ar[None]
        refined = coarse_density.clone().to(final_cells.dtype)
        refined[b[:, None, None], 0, rows[:, :, None], cols[:, None, :]] = final_cells[:, 0]
        if return_aux:
            return refined, {"gain": gain, "add": add, "p_inc": p_inc, "selected": selected,
                             "coarse_cells": coarse_cells, "final_cells": final_cells}
        return refined

    def forward(self, image, budget=0.25, selection_mask=None, min_score=None, return_aux=False):
        outputs = self.encode(image)
        if selection_mask is None:
            selection_mask = select_topk(outputs["error_scores"], budget, min_score)
        result = self.apply_refine(outputs["density"], outputs["stride4_features"], selection_mask,
                                   probs=outputs["error_probs"], return_aux=return_aux)
        final_density, aux = result if return_aux else (result, None)
        out = {**outputs, "selection_mask": selection_mask, "final_density": final_density}
        if return_aux:
            out["refine_aux"] = aux
        return out