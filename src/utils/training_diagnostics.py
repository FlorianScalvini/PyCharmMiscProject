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
def optimizer_step_metrics(optimizer, parameters, before_update, reference):
    """Inspect the actual update and post-step Adam state without modifying it.

    Call after optimizer.step(), when Lightning AMP has unscaled gradients.
    Adam moments use each parameter's own step count for bias correction.
    Denominator statistics include epsilon and the AMSGrad maximum when active.
    """
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    is_adam = isinstance(raw_optimizer, (torch.optim.Adam, torch.optim.AdamW))
    groups = raw_optimizer.param_groups
    group_by_parameter = {id(p): group for group in groups for p in group["params"]}
    update_squared = reference.new_zeros((), dtype=torch.float32)
    parameter_squared = torch.zeros_like(update_squared)
    gradient_squared = torch.zeros_like(update_squared)
    update_gradient_dot = torch.zeros_like(update_squared)
    moment_squared = torch.zeros_like(update_squared)
    moment_gradient_squared = torch.zeros_like(update_squared)
    moment_gradient_dot = torch.zeros_like(update_squared)
    variance_sum = torch.zeros_like(update_squared)
    denominator_min = reference.new_tensor(float("inf"), dtype=torch.float32)
    state_elements = 0
    for parameter, before in zip(parameters, before_update):
        update = (parameter.detach() - before).float()
        parameter_squared += before.float().square().sum()
        update_squared += update.square().sum()
        gradient = parameter.grad.detach().float() if parameter.grad is not None else None
        if gradient is not None:
            gradient_squared += gradient.square().sum()
            update_gradient_dot += (update * gradient).sum()
        if not is_adam:
            continue
        state = raw_optimizer.state.get(parameter, {})
        if "exp_avg" not in state or "exp_avg_sq" not in state:
            continue  # For example, AMP skipped the first optimizer step.
        group = group_by_parameter[id(parameter)]
        beta1, beta2 = group["betas"]
        step = state["step"]
        step = float(step.item()) if torch.is_tensor(step) else float(step)
        if step <= 0:
            continue
        moment = state["exp_avg"].detach().float() / (1 - beta1 ** step)
        second_moment = state["max_exp_avg_sq"] if group["amsgrad"] else state["exp_avg_sq"]
        variance = second_moment.detach().float() / (1 - beta2 ** step)
        denominator = variance.sqrt() + group["eps"]
        # Ignore coordinates whose first moment is zero: they produce no
        # adaptive update, but otherwise force a misleading epsilon minimum.
        denominator_min = torch.minimum(
            denominator_min,
            torch.where(moment != 0, denominator, float("inf")).amin(),
        )
        moment_squared += moment.square().sum()
        variance_sum += variance.sum()
        state_elements += variance.numel()
        if gradient is not None:
            moment_gradient_squared += gradient.square().sum()
            moment_gradient_dot += (moment * gradient).sum()

    def cosine(dot, squared_a, squared_b):
        denominator = squared_a.sqrt() * squared_b.sqrt()
        defined = (denominator > 0) & torch.isfinite(denominator) & torch.isfinite(dot)
        value = torch.where(defined, dot / denominator.clamp_min(1e-30), torch.zeros_like(dot))
        return value.clamp(-1, 1), defined.float()

    update_cosine, update_defined = cosine(-update_gradient_dot, update_squared, gradient_squared)
    update_norm = update_squared.sqrt()
    parameter_norm = parameter_squared.sqrt()
    metrics = {
        "parameter_update_norm": update_norm,
        "parameter_norm_before_update": parameter_norm,
        "parameter_relative_update_norm": update_norm / parameter_norm.clamp_min(1e-30),
        "update_cosine_negative_gradient": update_cosine,
        "update_direction_defined": update_defined,
        "optimizer_is_adam": float(is_adam),
    }
    if is_adam:
        moment_cosine, moment_defined = cosine(moment_gradient_dot, moment_squared, moment_gradient_squared)
        metrics.update({
            "adam_amsgrad_enabled": float(all(group["amsgrad"] for group in groups)),
            "adam_state_available": float(state_elements > 0),
            "adam_first_moment_norm": moment_squared.sqrt(),
            "adam_sqrt_second_moment_rms": (variance_sum / max(state_elements, 1)).sqrt(),
            "adam_denominator_min_active": torch.where(
                torch.isfinite(denominator_min), denominator_min, torch.zeros_like(denominator_min)
            ),
            "adam_momentum_cosine_gradient": moment_cosine,
            "adam_momentum_direction_defined": moment_defined,
        })
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
