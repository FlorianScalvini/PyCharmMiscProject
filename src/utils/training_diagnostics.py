"""Detached training metrics; expensive component gradients are sampled periodically."""

import torch

from utils.registration import phi_to_displacement_voxel
from utils.utils import compute_jacobian_determinant_3d


def tensor_norm(tensors, reference):
    """Global L2 norm, accumulated in float32 (including mixed-precision runs)."""
    squared = reference.new_zeros((), dtype=torch.float32)
    for tensor in tensors:
        if tensor is not None:
            squared = squared + tensor.detach().float().square().sum()
    return squared.sqrt()


def component_gradient_metrics(losses, parameters):
    """Measure weighted parameter gradients without changing optimizer .grad buffers.

    Call only on a single device: DDP does not support autograd.grad for its
    parameter gradients. Total backward/update metrics remain available in DDP.
    """
    parameters = tuple(parameters)
    if not parameters:
        return {}
    reference = parameters[0]
    gradients, norms, metrics = {}, {}, {}
    for name, loss in losses.items():
        gradients[name] = (
            torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            if loss.requires_grad else (None,) * len(parameters)
        )
        norms[name] = tensor_norm(gradients[name], reference)
        metrics[f"grad_norm_{name}"] = norms[name]
    if "seg" in gradients and "jac" in gradients:
        denominator = norms["seg"] * norms["jac"]
        if denominator.item() > 0:
            dot = reference.new_zeros((), dtype=torch.float32)
            for seg, jac in zip(gradients["seg"], gradients["jac"]):
                if seg is not None and jac is not None:
                    dot = dot + (seg.detach().float() * jac.detach().float()).sum()
            metrics["grad_cosine_seg_jac"] = (dot / denominator).clamp(-1, 1)
    return metrics


@torch.no_grad()
def deformation_metrics(trajectory, grid):
    """Summarize follow-ups in model/voxel space; exclude the identity baseline."""
    reference = trajectory[0]
    min_jac = reference.new_tensor(float("inf"))
    folds = reference.new_zeros(())
    outside = reference.new_zeros(())
    max_displacement = reference.new_zeros(())
    voxel_count = 0
    followups = trajectory.shape[0] - 1
    for phi in trajectory[1:]:
        displacement = phi_to_displacement_voxel(phi, grid)
        determinant = compute_jacobian_determinant_3d(displacement)
        min_jac = torch.minimum(min_jac, determinant.min())
        folds = folds + (determinant <= 0).sum()
        outside = outside + (phi.abs() > 1).any(dim=1).sum()
        max_displacement = torch.maximum(
            max_displacement, displacement.square().sum(dim=1).sqrt().max()
        )
        voxel_count += determinant.numel()
    return {
        "jac_min": min_jac,
        "jac_neg_count": folds / followups,
        "jac_neg_fraction": folds / voxel_count,
        "out_of_bounds_fraction": outside / voxel_count,
        "max_displacement_voxel": max_displacement,
    }
