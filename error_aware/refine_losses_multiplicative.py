
"""Training-time selection and refinement losses for multiplicative-additive Stage 2."""

import random

import torch
import torch.nn.functional as F

# Support both package imports and direct script execution.
try:
    from .model_multiplicative import (
        cell_sum,
        select_topk,
        mask_to_pixels,
    )
except ImportError:
    from model_multiplicative import (
        cell_sum,
        select_topk,
        mask_to_pixels,
    )


def build_training_mask(
    err_true,
    err_pred,
    budget,
    p_true,
    background_frac=0.25,
):
    """Build a scheduled true/predicted-error selection mask.

    The union can exceed `budget`: background_frac is an additional
    sampling fraction, not a replacement for the primary selection.
    """
    base = err_true if random.random() < p_true else err_pred

    mask = select_topk(base, budget)

    if background_frac > 0:
        mask = mask | select_topk(
            torch.rand_like(base),
            background_frac,
        )

    return mask


def refinement_loss(
    final,
    coarse,
    gt,
    mask,
    aux,
    w_cell=1.0,
    w_pix=1.0,
    w_harm=0.5,
    w_empty=3.0,
    w_add=0.1,
    w_glob=0.05,
    empty_thresh=0.05,
):
    """Compute selected-cell, pixel, do-no-harm, additive-mass
    and global refinement losses.

    Args:
        final: Refined density map [B, 1, 64, 64].
        coarse: Coarse density map [B, 1, 64, 64].
        gt: Ground-truth density map [B, 1, 64, 64].
        mask: Selected-cell mask [B, 1, 16, 16].
        aux: Auxiliary dictionary returned by
             apply_refine(..., return_aux=True).

    Returns:
        total: Combined refinement loss.
        parts: Dictionary containing individual loss components.
    """
    final = final.float()
    coarse = coarse.float().detach()
    gt = gt.float()
    m = mask.float()

    n = m.sum().clamp(min=1.0)

    # Aggregate density into 4x4 stride-8 cells.
    fc = cell_sum(final)
    cc = cell_sum(coarse)
    gc = cell_sum(gt)

    err_f = (fc - gc).abs()
    err_c = (cc - gc).abs()

    # Identify empty ground-truth cells.
    empty = (gc < empty_thresh).float()

    # 1. Selected-cell count error.
    # Empty cells receive additional weight.
    l_cell = (
        F.smooth_l1_loss(
            fc,
            gc,
            beta=0.5,
            reduction="none",
        )
        * (1.0 + w_empty * empty)
        * m
    ).sum() / n

    # 2. Pixel-level density loss.
    # Uses the Stage-1 x10 density scaling convention.
    mp = mask_to_pixels(m)

    l_pix = (
        (((final - gt) * 10.0) ** 2) * mp
    ).sum() / mp.sum().clamp(min=1.0)

    # 3. Do-no-harm loss.
    # Penalizes selected cells when refinement increases count error.
    l_harm = (
        F.relu(err_f - err_c) * m
    ).sum() / n

    # 4. Additive-mass regularization.
    # aux["add"] contains only selected central cells.
    if aux is not None and aux.get("add") is not None:
        l_add = aux["add"].float().sum((1, 2, 3)).mean()
    else:
        l_add = final.new_zeros(())

    # 5. Image-level count consistency.
    l_glob = (
        final.sum((1, 2, 3)) - gt.sum((1, 2, 3))
    ).abs().mean()

    # Combined refinement objective.
    total = (
        w_cell * l_cell
        + w_pix * l_pix
        + w_harm * l_harm
        + w_add * l_add
        + w_glob * l_glob
    )

    parts = {
        "cell": l_cell,
        "pix": l_pix,
        "harm": l_harm,
        "add": l_add,
        "glob": l_glob,
    }

    return total, parts
