"""
Custom loss functions and regularisation penalties for deformable image registration.

Provides four ``torch.nn.Module`` subclasses used as terms in the total
registration loss:

* :class:`Grad3d` – spatial smoothness regularisation on a displacement field
  via first-order finite differences (L1 or L2 penalty).
* :class:`NonDetJacobianPenalty` – penalises folding (non-positive Jacobian
  determinants) by summing the negative part of ``det(J)``.
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
    """Smooth barrier on small Jacobian determinants (folding prevention).

    Uses a temperature-scaled softplus approximation of
    ``relu(epsilon - det(J))``.  This penalises folds and creates a small
    positive safety margin before the determinant reaches zero.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        displacement: torch.Tensor,
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> torch.Tensor:
        """Compute the non-positive Jacobian penalty for a displacement field.

        Args:
            displacement: Displacement field of shape ``(B, 3, D, H, W)``
                          in voxel units.
            spacing: Physical voxel spacing ``(dz, dy, dx)`` used when
                     computing finite-difference Jacobian derivatives.
                     Defaults to ``(1.0, 1.0, 1.0)``.

        Returns:
            Scalar tensor — mean smooth barrier over all voxels.
        """
        det_j = utils.compute_jacobian_determinant_3d(displacement, spacing)
        epsilon = 0.05
        temperature = 0.01
        return temperature * F.softplus(
            (epsilon - det_j) / temperature
        ).mean()


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


# ──────────────────────────────────────────────────────────────────────────────
#  Surface losses
# ──────────────────────────────────────────────────────────────────────────────

class CorticalMeanCurvatureLoss(nn.Module):
    """Mean-curvature agreement between a transported white surface and a target.

    The dense terms of the registration loss constrain intensities and label
    overlap, neither of which says anything about how *folded* the cortex is:
    the deformation can inflate the brain instead of plicating it.  This term
    closes that gap.  The white surface of the source session is transported by
    the deformation field, its per-vertex mean curvature is recomputed from the
    deformed geometry, and it is compared to the curvature of the target
    session, read from a precomputed volume at the transported positions.

    Source and target surfaces share no vertex correspondence — they are
    independent marching-cubes outputs with different vertex counts — hence the
    detour through a volumetric field, sampled trilinearly.  That sampling is
    also what makes the term differentiable with respect to the vertex
    positions, and therefore with respect to the deformation field.

    The comparison is restricted to a soft band around the target surface: a
    curvature sampled far from the surface has no meaning.  A Gaussian weight
    is used rather than a hard mask so that vertices which have drifted fade out
    smoothly instead of having their gradient cut.  A light point-to-surface
    anchor keeps the transported surface where the curvature comparison is
    valid; it is not meant to do the registration itself, which the similarity
    and segmentation terms already handle.

    All quantities are in **voxel units** — curvature in inverse voxels,
    distances in voxels — matching the displacement field.

    Args:
        band:            Standard deviation of the Gaussian band around the
                         target surface, in voxels.
        w_anchor:        Weight of the point-to-surface anchor term.
        huber_delta:     Transition point of the Huber loss on the curvature
                         residual.  Marching-cubes staircase artefacts give the
                         curvature heavy tails, and a plain L1 or L2 would let a
                         handful of outlier vertices dominate the gradient.
        cot_clamp:       Bound on the cotangent weights, guarding against
                         triangles the deformation has squashed flat.
        curvature_clamp: Bound on the computed curvature, in inverse voxels.
    """

    def __init__(
        self,
        band: float = 4.0,
        w_anchor: float = 0.1,
        huber_delta: float = 0.05,
        cot_clamp: float = 10.0,
        curvature_clamp: float = 2.0,
    ) -> None:
        super().__init__()
        self.band = band
        self.w_anchor = w_anchor
        self.huber_delta = huber_delta
        self.cot_clamp = cot_clamp
        self.curvature_clamp = curvature_clamp

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        curvature_volume: torch.Tensor,
        distance_volume: torch.Tensor,
    ) -> torch.Tensor:
        """Compare the curvature of a transported surface to the target field.

        Args:
            vertices:         Transported vertex positions ``(V, 3)`` in voxel
                              coordinates — the output of
                              :func:`utils.mesh.warp_vertices`.
            faces:            Triangle indices ``(F, 3)``, wound outward.  The
                              topology of the source surface, unchanged by the
                              deformation.
            curvature_volume: Target mean curvature ``(1, 1, D, H, W)``.
            distance_volume:  Signed distance to the target surface,
                              ``(1, 1, D, H, W)``, in voxels.

        Returns:
            Scalar tensor — weighted curvature residual plus anchor.
        """
        curvature = mesh.mean_curvature(
            vertices,
            faces,
            cot_clamp=self.cot_clamp,
            clamp=self.curvature_clamp,
        )
        target_curvature = mesh.sample_volume(curvature_volume, vertices)[:, 0]
        distance = mesh.sample_volume(distance_volume, vertices)[:, 0]

        weight = torch.exp(-(distance / self.band) ** 2)
        residual = F.smooth_l1_loss(
            curvature, target_curvature, beta=self.huber_delta, reduction='none'
        )
        loss_curvature = (weight * residual).sum() / weight.sum().clamp(min=1e-6)

        loss_anchor = F.smooth_l1_loss(
            distance, torch.zeros_like(distance), beta=1.0
        )
        return loss_curvature + self.w_anchor * loss_anchor
