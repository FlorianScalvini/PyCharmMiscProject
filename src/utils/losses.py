"""
Custom loss functions and regularisation penalties for deformable image registration.

Provides ``torch.nn.Module`` subclasses used as terms in the total
registration loss:

* :class:`NormalizedDiffusionLoss` – velocity diffusion in normalised coordinates.
* :class:`Grad3d` – spatial smoothness regularisation on a displacement field
  via first-order finite differences (L1 or L2 penalty).
* :class:`NonDetJacobianPenalty` – penalises Jacobian determinants below a
  positive safety margin with a mean Smooth L1 penalty.
* :class:`LogDetJacobianPenalty` – encourages volume-preserving deformations
  by penalising the log of the Jacobian determinant.
* :class:`CorticalMeanCurvatureLoss` – drives the cortical plication by
  matching the mean curvature of the transported white surface to that of the
  target session.

Author : Florian Scalvini
"""

# --- Standard library ---
from typing import Sequence

# --- Third-party ---
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Local ---
import utils.utils as utils


# ──────────────────────────────────────────────────────────────────────────────
#  Regularisation losses
# ──────────────────────────────────────────────────────────────────────────────

class NormalizedDiffusionLoss(nn.Module):
    """Sum spatial derivative energies on the normalised [-1, 1]^3 domain.

    Input velocities have shape (B, 3, D, H, W) and normalised units.
    Derivative spacing follows D/H/W; vector channels are weighted equally,
    so their order does not affect the energy. Each squared derivative is
    averaged over batch, channels and voxels, then summed over spatial axes.
    """

    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        if velocity.ndim != 5 or velocity.shape[1] != 3:
            raise ValueError("velocity must have shape (B, 3, D, H, W)")
        if any(size < 2 for size in velocity.shape[2:]):
            raise ValueError("each spatial dimension must contain at least two voxels")
        spacing = tuple(2.0 / (size - 1) for size in velocity.shape[2:])
        gradients = torch.gradient(velocity, spacing=spacing, dim=(2, 3, 4))
        return sum(gradient.square().mean() for gradient in gradients)


class Grad3d(nn.Module):
    """Spatial smoothness regulariser based on first-order finite differences.

    Computes the mean absolute (L1) or mean squared (L2) gradient of the
    predicted displacement field across all three spatial axes and returns
    their average.  Adding this term to the training loss penalises
    spatially irregular deformations.

    Args:
        penalty: Gradient penalty type.  ``'l1'`` uses absolute differences;
                 ``'l2'`` uses squared differences.  Defaults to ``'l1'``.

    Raises:
        ValueError: If *penalty* is not ``'l1'`` or ``'l2'``.
    """

    def __init__(self, penalty: str = 'l1') -> None:
        super().__init__()
        if penalty not in ['l1', 'l2']:
            raise ValueError(f"Unknown penalty type: {penalty}")
        self.penalty = penalty

    def forward(self, x_pred: torch.Tensor) -> torch.Tensor:
        """Compute the smoothness penalty for a displacement field.

        Args:
            x_pred: Displacement field of shape ``(B, 3, D, H, W)``.

        Returns:
            Scalar tensor — mean gradient magnitude across all axes.
        """
        # Compute gradients in each direction
        dx = torch.abs(x_pred[:, :, 1:, :, :] - x_pred[:, :, :-1, :, :])
        dy = torch.abs(x_pred[:, :, :, 1:, :] - x_pred[:, :, :, :-1, :])
        dz = torch.abs(x_pred[:, :, :, :, 1:] - x_pred[:, :, :, :, :-1])
        # Apply penalty (squared for L2 penalty)
        if self.penalty == 'l2':
            dy, dx, dz = dy**2, dx**2, dz**2
        grad = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0
        return grad


class NonDetJacobianPenalty(nn.Module):
    """Smooth L1 penalty on Jacobian determinants below a positive margin.

    Apply Smooth L1 with beta=0.1 to ``relu(0.05 - det(J))``. The penalty
    and its determinant derivative are zero at and above the margin. Below
    it, the slope increases continuously in magnitude before saturating at
    one. This encourages folding prevention without guaranteeing it.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        displacement: torch.Tensor,
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> torch.Tensor:
        """Compute the below-margin Jacobian penalty for a displacement field.

        Args:
            displacement: Displacement field of shape ``(B, 3, D, H, W)``
                          in voxel units.
            spacing: Physical voxel spacing ``(dz, dy, dx)`` used when
                     computing finite-difference Jacobian derivatives.
                     Defaults to ``(1.0, 1.0, 1.0)``.

        Returns:
            Scalar tensor — mean Smooth L1 margin penalty over all voxels.
        """
        det_j = utils.compute_jacobian_determinant_3d(displacement, spacing)
        epsilon = 0.05
        delta = 0.1
        violation = F.relu(epsilon - det_j)
        return F.smooth_l1_loss(
            violation, torch.zeros_like(violation), beta=delta, reduction="mean"
        )


class LogDetJacobianPenalty(nn.Module):
    """Penalty based on the log of the Jacobian determinant.

    Encourages volume-preserving deformations by summing ``log(det(J))``
    over all voxels.  The determinant is clamped to ``1e-6`` before taking
    the log to avoid numerical instability near folded regions.  A purely
    volume-preserving deformation would yield ``det(J) = 1`` everywhere
    (i.e. ``log(det(J)) = 0``), so minimising this term in absolute value
    pushes the field towards incompressibility.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        displacement: torch.Tensor,
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> torch.Tensor:
        """Compute the log-determinant Jacobian penalty for a displacement field.

        Args:
            displacement: Displacement field of shape ``(B, 3, D, H, W)``
                          in voxel units.
            spacing: Physical voxel spacing ``(dz, dy, dx)`` used when
                     computing finite-difference Jacobian derivatives.
                     Defaults to ``(1.0, 1.0, 1.0)``.

        Returns:
            Scalar tensor — sum of ``log(clamp(det(J), min=1e-6))`` over
            all voxels.
        """
        det_j = utils.compute_jacobian_determinant_3d(displacement, spacing)
        log_det_j = torch.log(torch.clamp(det_j, min=1e-6))
        return torch.sum(log_det_j)
