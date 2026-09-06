"""
Neural ODE-based longitudinal brain MRI registration model.

Architecture overview
---------------------
The module is built around three tightly coupled classes that together
implement a continuous-time deformable registration pipeline:

1. **VelocityNet** – a time-conditioned 3-D U-Net that takes the
   concatenation of the source image, the currently warped image, and the
   target image as input and predicts a dense 3-D velocity field ``v(t)``.
   Temporal context (current integration time *t*, start age *tA*, end age
   *tB*) is encoded with sinusoidal embeddings and injected into every
   encoder / decoder block through a shared time MLP.

2. **ODEFunction** – wraps :class:`VelocityNet` as the right-hand side
   ``f(t, φ_t)`` of the neural ODE ``dφ/dt = v(t, φ_t)``.  At each
   solver evaluation it also accumulates a velocity regularisation loss
   (e.g. :class:`monai.losses.DiffusionLoss`) along the trajectory.

3. **LongitudinalODERegistration** – the top-level ``nn.Module`` consumed
   by the Lightning training loop.  It integrates :class:`ODEFunction`
   from *ages[0]* to *ages[-1]* using the RK4 solver from
   `torchdiffeq <https://github.com/rtqichen/torchdiffeq>`_ and returns
   the full deformation trajectory (one field per acquisition age) together
   with the cumulative regularisation loss.

Coordinate convention
---------------------
All grids and displacement fields follow the shape ``(B, 3, D, H, W)``.
The identity grid passed as *grid* to :class:`LongitudinalODERegistration`
must be in normalised ``[-1, 1]`` coordinates (as produced by
:func:`utils.registration.generate_grid3d_tensor`).

Author : Florian Scalvini
"""

# --- Third-party ---
import monai
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

# --- Local ---
import utils.registration as registration
import utils.losses as losses
from utils.utils import *
from .unet import EncoderUnet, UnetUpBlock
from .time_encoding import SinusoidalPositionEmbeddings


# ──────────────────────────────────────────────────────────────────────────────
#  Top-level registration model
# ──────────────────────────────────────────────────────────────────────────────

class LongitudinalODERegistration(nn.Module):
    """Longitudinal registration model driven by a neural ODE.

    The first image is the source and the last image is the target.
    Acquisition ages are mapped to relative times between ``0`` and ``1``,
    preserving intermediate visits for trajectory supervision.

    Parameters
    ----------
    shape : list of int
        Spatial dimensions ``[H, W, D]`` of the input volumes.
    step_time : float
        Fixed step size passed to the RK4 ODE solver.  Smaller values
        increase accuracy at the cost of more :class:`VelocityNet` forward
        passes per training step.
    """

    def __init__(
        self,
        shape: list[int] = [192, 224, 192],
        step_time: float = 0.05,
        temporal_conditioning: str = "endpoint_ages",
        age_span: float = 1.0,
        duration_scale: float = 4.0,
    ) -> None:
        super().__init__()
        self.velocity_net = VelocityNet(
            shape=shape,
            temporal_conditioning=temporal_conditioning,
            age_span=age_span,
            duration_scale=duration_scale,
        )
        self.jacobian_loss = losses.NonDetJacobianPenalty()
        self.step_time = step_time

    def forward(
        self,
        imageA: torch.Tensor,
        imageB: torch.Tensor,
        ages: torch.Tensor,
        grid: torch.Tensor,
        loss_v: nn.Module = monai.losses.DiffusionLoss(normalize=True), # type: ignore
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Integrate the velocity field over *ages* and return deformation trajectories.

        Parameters
        ----------
        imageA : torch.Tensor
            Source (baseline) image of shape ``(B, 1, H, W, D)``.
        imageB : torch.Tensor
            Last-visit target image of shape ``(B, 1, H, W, D)``.
        ages : torch.Tensor
            Sorted integration times of shape ``(N,)``.  ``ages[0]`` is the
            starting age *t₀* and ``ages[-1]`` is the final age.
        grid : torch.Tensor
            Identity grid of shape ``(B, 3, D, H, W)`` with coordinates in
            ``[-1, 1]``, used as the initial deformation state ``φ₀``.
        loss_v : nn.Module
            Velocity regularisation loss applied at each ODE evaluation
            step (default: :class:`monai.losses.DiffusionLoss`).

        Returns
        -------
        phi_traj : torch.Tensor
            Deformation field at each integration time, shape
            ``(N, B, 3, D, H, W)``.
        loss_reg : torch.Tensor
            Cumulative regularisation loss accumulated up to the final
            time step (scalar).
        """
        duration = ages[-1] - ages[0]
        if torch.any(duration <= 0):
            raise ValueError("the last acquisition age must be greater than the first")
        relative_ages = (ages - ages[0]) / duration
        ode_func = ODEFunction(
            self.velocity_net,
            imageA,
            imageB,
            identity_grid=grid,
            loss_jac=self.jacobian_loss,
            loss_v=loss_v,
            source_age_normalized=ages[0],
            target_age_normalized=ages[-1],
        )
        zero = imageA.new_zeros(())
        phi_traj, loss_reg_traj, loss_jac_traj = odeint( # type: ignore
            ode_func,
            (
                grid,
                zero,
                zero.clone(),
            ),  # initial state: (phi₀, loss_reg₀, loss_jac₀)
            relative_ages,
            method="rk4",
            options={"step_size": self.step_time},
        )
        return phi_traj, loss_reg_traj[-1], loss_jac_traj[-1]


# ──────────────────────────────────────────────────────────────────────────────
#  ODE right-hand side
# ──────────────────────────────────────────────────────────────────────────────

class ODEFunction(nn.Module):
    """Right-hand side of the neural ODE: ``f(t, φ_t) = v(t, φ_t)``.

    Wraps :class:`VelocityNet` and accumulates a scalar velocity
    regularisation loss along the ODE trajectory so it can be returned
    as part of the state vector and differentiated through by
    ``odeint_adjoint``.

    Parameters
    ----------
    vnet : nn.Module
        Velocity network used to predict ``v(t, φ_t)``.
    imageA : torch.Tensor
        Source image ``(B, 1, H, W, D)``, kept constant during integration.
    imageB : torch.Tensor
        Target image ``(B, 1, H, W, D)``, kept constant during integration.
    loss_v : nn.Module
        Velocity regularisation loss module (e.g. diffusion or bending
        energy) evaluated at each ODE step.
    """

    def __init__(
        self,
        vnet: nn.Module,
        imageA: torch.Tensor,
        imageB: torch.Tensor,
        identity_grid: torch.Tensor,
        loss_jac: nn.Module,
        source_age_normalized: torch.Tensor,
        target_age_normalized: torch.Tensor,
        loss_v: nn.Module = monai.losses.DiffusionLoss(normalize=True), # type: ignore
    ) -> None:
        super().__init__()
        self.vnet = vnet
        self.imageA = imageA
        self.imageB = imageB
        self.identity_grid = identity_grid
        self.loss_v = loss_v
        self.loss_jac = loss_jac
        self.source_age_normalized = source_age_normalized
        self.target_age_normalized = target_age_normalized
        self.duration_normalized = target_age_normalized - source_age_normalized

    def forward(
        self,
        t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the ODE right-hand side at time *t*.

        Parameters
        ----------
        t : torch.Tensor
            Current integration time (scalar tensor).
        state : tuple of torch.Tensor
            ``(phi_t, loss_reg_acc)`` — current deformation field of shape
            ``(B, 3, D, H, W)`` and the scalar accumulated regularisation
            loss.

        Returns
        -------
        v : torch.Tensor
            Predicted velocity field ``(B, 3, D, H, W)`` — the time
            derivative ``dφ/dt`` at the current state.
        loss_v : torch.Tensor
            Velocity regularisation loss at the current step (scalar),
            accumulated into the state for later retrieval.
        """
        phi_t = state[0]
        v = self.vnet(
            t,
            self.source_age_normalized,
            self.target_age_normalized,
            phi_t,
            self.imageA,
            self.imageB,
        )
        loss_v: torch.Tensor = self.loss_v(v)
        shape = phi_t.shape[2:]
        scale = phi_t.new_tensor(shape).view(1, 3, 1, 1, 1)
        displacement_voxel = (phi_t - self.identity_grid) * (scale - 1) / 2.0
        loss_jac: torch.Tensor = self.loss_jac(displacement_voxel)
        # The ODE is parameterized by relative time tau in [0, 1].
        # |d age / d tau| keeps accumulated losses positive in both temporal
        # directions and converts their integral back to the absolute-age scale.
        absolute_duration = torch.abs(self.duration_normalized)
        return v, loss_v * absolute_duration, loss_jac * absolute_duration


# ──────────────────────────────────────────────────────────────────────────────
#  Velocity network
# ──────────────────────────────────────────────────────────────────────────────

class VelocityNet(nn.Module):
    """Time-conditioned 3-D U-Net that predicts a dense velocity field.

    The network concatenates three volumes — the source image *imageA*, the
    current warped image ``warp(imageA, φ_t - grid)``, and the target image
    *imageB* — into a 3-channel input and processes it through a symmetric
    encoder–decoder with skip connections.

    Relative time and the dataset-normalized first and last ages are encoded
    with sinusoidal embeddings followed by a 3-layer SiLU MLP. Relative time
    runs from ``0`` to ``1`` while both endpoint ages stay fixed.

    Parameters
    ----------
    reg_head_chan : int
        Number of channels in the final registration head convolutions.
    shape : list of int
        Spatial dimensions ``[H, W, D]`` of the input volumes.
    t_dim : int
        Dimensionality of the time embedding fed to the U-Net blocks.
    t_dim_enc : int
        Dimensionality of the raw sinusoidal time encoding before the MLP.
    """

    grid: torch.Tensor

    def __init__(
        self,
        reg_head_chan: int = 16,
        shape: list[int] = [192, 224, 192],
        t_dim: int = 48,
        t_dim_enc: int = 16,
        temporal_conditioning: str = "endpoint_ages",
        age_span: float = 1.0,
        duration_scale: float = 4.0,
    ) -> None:
        super().__init__()
        if temporal_conditioning not in {"endpoint_ages", "relative_duration"}:
            raise ValueError(f"unknown temporal conditioning: {temporal_conditioning}")
        if age_span <= 0:
            raise ValueError("age_span must be positive")
        if duration_scale <= 0:
            raise ValueError("duration_scale must be positive")
        self.shape = shape
        self.temporal_conditioning = temporal_conditioning
        self.age_span = float(age_span)
        self.duration_scale = float(duration_scale)
        self.register_buffer(
            "grid",
            registration.generate_grid3d_tensor(self.shape),
            persistent=False,
        )
        self.t_dim_enc = t_dim_enc
        self.t_dim = t_dim
        self.encoder = EncoderUnet(
            in_channels=3, channels=[16, 32, 64, 128, 256], t_dim=self.t_dim
        )
        self.decoder_0 = UnetUpBlock(
            in_channels=256, out_channels=128, kernel_size=3, t_dim=self.t_dim
        )
        self.decoder_1 = UnetUpBlock(
            in_channels=128, out_channels=64, kernel_size=3, t_dim=self.t_dim
        )
        self.decoder_2 = UnetUpBlock(
            in_channels=64, out_channels=32, kernel_size=3, t_dim=self.t_dim
        )
        self.decoder_3 = UnetUpBlock(
            in_channels=32, out_channels=16, kernel_size=3, t_dim=self.t_dim
        )
        self.temp_enc = SinusoidalPositionEmbeddings(
            self.t_dim_enc, max_periods=100
        )
        temporal_inputs = 2 if temporal_conditioning == "relative_duration" else 3
        self.time_mlp = nn.Sequential(
            nn.Linear(temporal_inputs * self.t_dim_enc, self.t_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.t_dim, self.t_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.t_dim, self.t_dim, bias=True),
        )
        velocity_output = nn.Conv3d(reg_head_chan, 3, kernel_size=3, padding=1)
        # Start the ODE at the identity deformation with zero velocity.
        nn.init.zeros_(velocity_output.weight)
        if velocity_output.bias is not None:
            nn.init.zeros_(velocity_output.bias)
        self.reg_head = nn.Sequential(
            nn.Conv3d(reg_head_chan, reg_head_chan, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(reg_head_chan, reg_head_chan, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            velocity_output,
        )

    def forward(
        self,
        t_relative: torch.Tensor,
        age_init: torch.Tensor,
        age_final: torch.Tensor,
        phi_t: torch.Tensor,
        image_A: torch.Tensor,
        image_B: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity from relative time and fixed endpoint ages.

        ``t_relative`` runs from ``0`` to ``1``. Endpoint ages retain their
        dataset-wide normalization; they are not replaced by ``0`` and ``1``.

        Parameters
        ----------
        t_relative : torch.Tensor
            Relative integration time, scalar or shape ``(B,)``.
        age_init : torch.Tensor
            Dataset-normalized first acquisition age, scalar or shape ``(B,)``.
        age_final : torch.Tensor
            Dataset-normalized last acquisition age, scalar or shape ``(B,)``.
        phi_t : torch.Tensor
            Current deformation field of shape ``(B, 3, D, H, W)`` in
            normalised ``[-1, 1]`` coordinates.
        image_A : torch.Tensor
            Source image ``(B, 1, H, W, D)``.
        image_B : torch.Tensor
            Target image ``(B, 1, H, W, D)``.
        Returns
        -------
        v : torch.Tensor
            Predicted velocity field of shape ``(B, 3, D, H, W)``.
        """
        # Match the voxel displacement used by the external loss warping.
        scale = phi_t.new_tensor(image_A.shape[2:]).view(1, 3, 1, 1, 1)
        df = (phi_t - self.grid) * (scale - 1) / 2
        warped = registration.warp(image_A, df)
        net_input = torch.cat([image_A, warped, image_B], dim=1)
        B: int = phi_t.shape[0]

        if t_relative.dim() == 0:
            t_relative = t_relative.expand(B)
        if age_init.dim() == 0:
            age_init = age_init.expand(B)
        if age_final.dim() == 0:
            age_final = age_final.expand(B)
        temporal_values = [t_relative]
        if self.temporal_conditioning == "relative_duration":
            duration = (age_final - age_init) * self.age_span / self.duration_scale
            temporal_values.append(duration)
        else:
            temporal_values.extend([age_init, age_final])
        temporal_context = torch.cat(
            [self.temp_enc(value) for value in temporal_values],
            dim=-1,
        )
        t_all: torch.Tensor = self.time_mlp(temporal_context)

        feat_maps = self.encoder(net_input, t_all)
        v = self.decoder_0(feat_maps[4], feat_maps[3], t_all)
        v = self.decoder_1(v, feat_maps[2], t_all)
        v = self.decoder_2(v, feat_maps[1], t_all)
        v = self.decoder_3(v, feat_maps[0], t_all)
        v = self.reg_head(v)
        return v
