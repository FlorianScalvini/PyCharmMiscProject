"""
Neural ODE-based longitudinal brain MRI registration model.

Architecture overview
---------------------
The module is built around three tightly coupled classes that together
implement a continuous-time deformable registration pipeline:

1. **VelocityNet** – a time-conditioned 3-D U-Net that takes the
   concatenation of the source image, the currently warped image, and the
   target image as input and predicts a dense 3-D velocity field ``v(t)``.
   Current age, relative position between the anchors, and their age
   interval are projected by a shared MLP and injected into every encoder /
   decoder block.

2. **ODEFunction** – wraps :class:`VelocityNet` as the right-hand side
   ``f(t, φ_t)`` of the neural ODE ``dφ/dt = v(t, φ_t)``.  At each
   RK4 evaluation it also accumulates velocity and Jacobian penalties.

3. **LongitudinalODERegistration** – the top-level ``nn.Module`` consumed
   by the Lightning training loop.  It integrates :class:`ODEFunction`
   from *ages[0]* to *ages[-1]* using the fixed-step RK4 solver from
   `torchdiffeq <https://github.com/rtqichen/torchdiffeq>`_ and returns
   the full deformation trajectory (one field per acquisition age) together
   with the regularisation integrals accumulated by RK4.

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
from .unet import EncoderUnet, UnetUpBlock


# ──────────────────────────────────────────────────────────────────────────────
#  Top-level registration model
# ──────────────────────────────────────────────────────────────────────────────

class LongitudinalODERegistration(nn.Module):
    """Longitudinal registration model driven by a neural ODE.

    Given a pair of images and a sorted sequence of acquisition ages, the
    model integrates a time-varying velocity field from ``ages[0]`` to
    ``ages[-1]`` and returns the deformation trajectories together with
    the cumulative regularisation loss.

    Parameters
    ----------
    shape : list of int
        Spatial dimensions ``[H, W, D]`` of the input volumes.
    step_time : float
        Fixed RK4 step size in relative anchor time (0 to 1), used for both
        the forward and adjoint solves.
    use_absolute_age : bool
        Whether to condition VelocityNet on the current absolute age.
    velocity_smoothing_sigma : float
        Standard deviation, in voxels, of the Gaussian filter applied to the
        velocity before composition and regularisation. Set to zero to disable.
    """

    def __init__(
        self,
        shape: list[int] = [192, 224, 192],
        step_time: float = 0.05,
        use_absolute_age: bool = True,
        velocity_smoothing_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.velocity_net = VelocityNet(shape=shape, use_absolute_age=use_absolute_age)
        self.jacobian_loss = losses.NonDetJacobianPenalty()
        self.velocity_loss = monai.losses.DiffusionLoss(normalize=True)
        if step_time <= 0:
            raise ValueError("step_time must be positive")
        if velocity_smoothing_sigma < 0:
            raise ValueError("velocity_smoothing_sigma must be non-negative")
        self.step_time = step_time
        self.velocity_smoothing_sigma = velocity_smoothing_sigma

    def forward(
        self,
        imageA: torch.Tensor,
        imageB: torch.Tensor,
        ages: torch.Tensor,
        ages_target: torch.Tensor,
        grid: torch.Tensor,
        loss_v: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Integrate the velocity field over *ages* and return deformation trajectories.

        Parameters
        ----------
        imageA : torch.Tensor
            Source (baseline) image of shape ``(B, 1, H, W, D)``.
        imageB : torch.Tensor
            Target (follow-up) image of shape ``(B, 1, H, W, D)``.
        ages : torch.Tensor
            Sorted integration times of shape ``(N,)``.  ``ages[0]`` is the
            starting age *t₀* and ``ages[-1]`` is the final age.
        grid : torch.Tensor
            Identity grid of shape ``(B, 3, D, H, W)`` with coordinates in
            ``[-1, 1]``, used as the initial deformation state ``φ₀``.
        loss_v : nn.Module
            Velocity penalty evaluated at every RK4 stage
            (default: :class:`monai.losses.DiffusionLoss` with normalize=True).

        Returns
        -------
        phi_traj : torch.Tensor
            Deformation field at each integration time, shape
            ``(N, B, 3, D, H, W)``.
        loss_reg : torch.Tensor
            Velocity regularisation accumulated during integration (scalar);
            Jacobian loss is accumulated simultaneously.
        """
        if ages.ndim != 1 or ages.numel() < 2:
            raise ValueError("ages must be a one-dimensional sequence of at least two ages")
        if not torch.isfinite(ages).all() or not torch.isfinite(ages_target).all():
            raise ValueError("acquisition ages must be finite")
        interval = ages_target - ages[0]
        if interval.numel() != 1 or interval.item() == 0:
            raise ValueError("source and target anchor ages must differ")
        relative_times = (ages - ages[0]) / interval
        if not torch.all(relative_times[1:] > relative_times[:-1]):
            raise ValueError("ages must be strictly ordered in the anchor direction")
        velocity_loss = self.velocity_loss if loss_v is None else loss_v
        ode_func = ODEFunction(
            self.velocity_net,
            imageA,
            imageB,
            ages[0],
            ages_target,
            identity_grid=grid,
            loss_jac=self.jacobian_loss,
            loss_v=velocity_loss,
            velocity_smoothing_sigma=self.velocity_smoothing_sigma,
        )
        zero = imageA.new_zeros(())
        phi_traj, loss_reg_traj, loss_jac_traj = odeint(
            ode_func,
            (grid, zero, zero.clone()),
            relative_times,
            method="rk4",
            options={"step_size": self.step_time},
            adjoint_method="rk4",
            adjoint_options={"step_size": self.step_time},
        )
        # Keep a plain counter dictionary, without registering another copy of
        # the velocity network. The adjoint retains this same RHS/counter.
        self.solver_stats = ode_func.solver_stats
        self.solver_stats["forward_nfe"] = self.solver_stats["nfe"]
        return phi_traj, loss_reg_traj[-1], loss_jac_traj[-1]


# ──────────────────────────────────────────────────────────────────────────────
#  ODE right-hand side
# ──────────────────────────────────────────────────────────────────────────────

class ODEFunction(nn.Module):
    """Deformation and regularisation right-hand side in relative anchor time."""

    def __init__(
        self,
        vnet: nn.Module,
        imageA: torch.Tensor,
        imageB: torch.Tensor,
        ageA: torch.Tensor,
        ageB: torch.Tensor,
        identity_grid: torch.Tensor,
        loss_jac: nn.Module,
        loss_v: nn.Module,
        velocity_smoothing_sigma: float,
    ) -> None:
        super().__init__()
        self.vnet = vnet
        self.imageA = imageA
        self.imageB = imageB
        self.ageA = ageA
        self.ageB = ageB
        self.interval = ageB - ageA
        self.identity_grid = identity_grid
        self.loss_v = loss_v
        self.loss_jac = loss_jac
        self.velocity_smoothing_sigma = velocity_smoothing_sigma
        self.solver_stats = {"nfe": 0}

    def forward(
        self,
        t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.solver_stats["nfe"] += 1
        phi = state[0]
        current_age = self.ageA + t * self.interval
        image_t = registration.warp_with_phi(self.imageA, phi)
        raw_velocity = self.vnet(
            current_age, self.imageA, image_t, self.imageB, self.ageA, self.ageB
        )
        velocity = registration.gaussian_smooth_3d(
            raw_velocity, sigma=self.velocity_smoothing_sigma
        )
        dphi = registration.sample_vector_field(velocity, phi)
        # Penalise the unfiltered prediction so the network cannot hide
        # high-frequency variations upstream of the Gaussian filter.
        loss_reg = self.loss_v(raw_velocity)
        displacement = registration.phi_to_displacement_voxel(
            phi, self.identity_grid
        )
        loss_jac = self.loss_jac(displacement)
        duration = self.interval.abs()
        return dphi * self.interval, loss_reg * duration, loss_jac * duration


# ──────────────────────────────────────────────────────────────────────────────
#  Velocity network
# ──────────────────────────────────────────────────────────────────────────────

class VelocityNet(nn.Module):
    """Time-conditioned 3-D U-Net that predicts a dense velocity field.

    The network concatenates the source image, the current warped image
    ``image_t``, and the target image into a 3-channel input and processes
    it through a symmetric encoder–decoder with skip connections.

    Current age, relative position
    ``alpha = (t - ageA) / (ageB - ageA)``, and observed age interval
    ``ageB - ageA`` are mapped directly through a 3-layer SiLU MLP. The
    resulting vector is injected into every encoder and decoder block via
    feature-wise modulation.

    Parameters
    ----------
    reg_head_chan : int
        Number of channels in the final registration head convolutions.
    shape : list of int
        Spatial dimensions ``[H, W, D]`` of the input volumes.
    t_dim : int
        Dimensionality of the time embedding fed to the U-Net blocks.
    use_absolute_age : bool
        If true, encode current age, relative position, and observed age
        interval. Otherwise encode only the latter two values.
    """

    def __init__(
        self,
        reg_head_chan: int = 16,
        shape: list[int] = [192, 224, 192],
        t_dim: int = 48,
        use_absolute_age: bool = True,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.t_dim = t_dim
        self.use_absolute_age = use_absolute_age
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
        self.time_mlp = nn.Sequential(
            nn.Linear(
                3 if use_absolute_age else 2,
                self.t_dim,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(self.t_dim, self.t_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.t_dim, self.t_dim, bias=True),
        )
        self.reg_head = nn.Sequential(
            nn.Conv3d(reg_head_chan, reg_head_chan, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(reg_head_chan, reg_head_chan, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(reg_head_chan, 3, kernel_size=3, padding=1),
        )

    def forward(
        self,
        t: torch.Tensor,
        image_A: torch.Tensor,
        image_t: torch.Tensor,
        image_B: torch.Tensor,
        ageA: torch.Tensor,
        ageB: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the velocity field at integration time *t*.

        The ODE reconstructs globally normalised age before calling this network,
        so ``current_age = t`` uses the dataset's age coordinate. This preserves age
        information across subjects. Relative position ``alpha`` is zero
        at the first anchor, one at the second, and exceeds one during
        forward extrapolation. The observed interval preserves the time
        scale between anchors. The anchor ages must differ.

        Current age is omitted when ``use_absolute_age`` is false.

        Parameters
        ----------
        t : torch.Tensor
            Current integration time, broadcastable to ``(B,)``.
        image_t : torch.Tensor
            Current warped source image ``(B, 1, D, H, W)``.
        image_A : torch.Tensor
            Source image ``(B, 1, H, W, D)``.
        image_B : torch.Tensor
            Target image ``(B, 1, H, W, D)``.
        ageA : torch.Tensor
            Start age of the integration interval, broadcastable to ``(B,)``.
        ageB : torch.Tensor
            Age of the second anchor image, broadcastable to ``(B,)``.

        Returns
        -------
        v : torch.Tensor
            Predicted velocity field of shape ``(B, 3, D, H, W)``.
        """
        net_input = torch.cat([image_A, image_t, image_B], dim=1)
        B: int = image_t.shape[0]

        if t.dim() == 0:
            t = t.expand(B)
        if ageA.dim() == 0:
            ageA = ageA.expand(B)
        if ageB.dim() == 0:
            ageB = ageB.expand(B)

        current_age = t  # Globally normalised dataset age.
        anchor_interval = ageB - ageA
        if torch.any(anchor_interval == 0):
            raise ValueError("anchor ages must differ to compute relative position")
        alpha = (current_age - ageA) / anchor_interval
        temporal_inputs = [alpha, anchor_interval]
        if self.use_absolute_age:
            temporal_inputs.insert(0, current_age)
        t_all = torch.stack(temporal_inputs, dim=1)
        t_all = self.time_mlp(t_all)
        feat_maps = self.encoder(net_input, t_all)
        v = self.decoder_0(feat_maps[4], feat_maps[3], t_all)
        v = self.decoder_1(v, feat_maps[2], t_all)
        v = self.decoder_2(v, feat_maps[1], t_all)
        v = self.decoder_3(v, feat_maps[0], t_all)
        v = self.reg_head(v)
        return v
