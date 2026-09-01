"""
Image quality metrics for evaluating deformable registration results.

Provides differentiable, mask-aware implementations of standard image
quality metrics that can be computed directly on PyTorch tensors:

* :func:`PSNR` – Peak Signal-to-Noise Ratio, optionally restricted to a
  foreground region defined by a binary mask.

Author : Florian Scalvini
"""

# --- Standard library ---
import math
from typing import Optional, Union

# --- Third-party ---
import torch
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────────────────────

def PSNR(
    y_pred: Tensor,
    y: Tensor,
    fg_mask: Optional[Tensor] = None,
    max_val: Union[int, float] = 1.0,
) -> Tensor:
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two image volumes.

    PSNR is defined as:

        PSNR = 20 · log10(max_val) − 10 · log10(MSE)

    where MSE is computed only over foreground voxels when *fg_mask* is
    provided, making the metric robust to large background regions that
    would otherwise dominate the error.

    Args:
        y_pred:  Predicted (warped) image of shape ``(B, C, D, H, W)``.
        y:       Ground-truth image of the same shape as *y_pred*.
        fg_mask: Optional binary foreground mask of shape ``(1, C, D, H, W)``
                 or ``(B, C, D, H, W)``.  A single-batch mask is broadcast
                 across the full batch.  If ``None``, all voxels are included
                 (equivalent to a mask of ones).
        max_val: Maximum possible pixel / voxel value of the signal.
                 Use ``1.0`` for images normalised to ``[0, 1]`` and ``255``
                 for 8-bit images.  Defaults to ``1.0``.

    Returns:
        Tensor of shape ``(B, C)`` containing the per-sample, per-channel
        PSNR in decibels.  Higher values indicate better reconstruction
        quality (identical images → ``+∞``).
    """
    B = y_pred.shape[0]
    if fg_mask is None:
        fg_mask = torch.ones_like(y_pred, dtype=torch.float)
    else:
        if fg_mask.shape[0] == 1:
            fg_mask = fg_mask.repeat(B, 1, 1, 1, 1)
    assert fg_mask.shape == y_pred.shape

    mse_out = (fg_mask * ((y_pred - y) ** 2)).sum(dim=(2, 3, 4)) \
                / fg_mask.sum(dim=(2, 3, 4))
    return 20 * math.log10(max_val) - 10 * torch.log10(mse_out)
