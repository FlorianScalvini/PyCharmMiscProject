"""Topology-aware centerline Dice (clDice), implemented with PyTorch."""

import torch
import torch.nn.functional as F
from torch import Tensor

from ._surface import validate_masks


def _max_pool(mask: Tensor) -> Tensor:
    if mask.ndim == 4:
        return F.max_pool2d(mask, 3, stride=1, padding=1)
    return F.max_pool3d(mask, 3, stride=1, padding=1)


def _erode(mask: Tensor) -> Tensor:
    padding = (1, 1) * (mask.ndim - 2)
    complement = F.pad(1.0 - mask, padding, value=1.0)
    if mask.ndim == 4:
        return 1.0 - F.max_pool2d(complement, 3, stride=1)
    return 1.0 - F.max_pool3d(complement, 3, stride=1)


def _skeletonize(mask: Tensor) -> Tensor:
    """Compute a morphological skeleton using PyTorch pooling operations."""
    current = mask.float()
    skeleton = torch.zeros_like(mask)
    for _ in range(max(mask.shape[2:])):
        if not torch.any(current):
            break
        eroded = _erode(current)
        opened = _max_pool(eroded)
        skeleton |= current.bool() & ~opened.bool()
        current = eroded
    return skeleton


def cldice(prediction: Tensor, target: Tensor, smooth: float = 1e-8) -> Tensor:
    """Compute clDice for two binary 2-D or 3-D masks.

    clDice is the harmonic mean of topology precision and topology sensitivity.
    It is especially useful for thin, connected or highly folded structures.
    Two empty masks score 1; exactly one empty mask scores 0.
    """
    prediction, target = validate_masks(prediction, target)
    if smooth < 0:
        raise ValueError("smooth must be non-negative")

    prediction_skeleton = _skeletonize(prediction)
    target_skeleton = _skeletonize(target)
    spatial_dims = tuple(range(2, prediction.ndim))
    topology_precision = ((prediction_skeleton & target).sum(dim=spatial_dims) + smooth) / (
        prediction_skeleton.sum(dim=spatial_dims) + smooth
    )
    topology_sensitivity = ((target_skeleton & prediction).sum(dim=spatial_dims) + smooth) / (
        target_skeleton.sum(dim=spatial_dims) + smooth
    )
    score = (
        2.0 * topology_precision * topology_sensitivity
        / (topology_precision + topology_sensitivity)
    )
    prediction_empty = ~prediction.any(dim=spatial_dims)
    target_empty = ~target.any(dim=spatial_dims)
    score = torch.where(prediction_empty & target_empty, 1.0, score)
    return torch.where(prediction_empty ^ target_empty, 0.0, score)
