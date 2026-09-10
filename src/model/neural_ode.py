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

    Given a pair of images and a sorted sequence of acquisition ages, the
    model integrates a time-varying velocity field from ``ages[0]`` to
    ``ages[-1]`` and returns the deformation trajectories together with
    the cumulative regularisation loss.

    Parameters
    ----------
    shape : list of int
        Spatial dimensions ``[H, W, D]`` of the input volumes.
    step_time : float
        Fixed step size passed to the RK4 ODE solver.  Smaller values
        increase accuracy at the cost of more :class:`VelocityNet` forward
        passes per training step.
    """

    def __init__(self, shape: list[int] = [192, 224, 192], step_time: float = 0.05) -> None:
        super().__init__()
        self.velocity_net = VelocityNet(shape=shape)
        self.jacobian_loss = losses.NonDetJacobianPenalty()
        self.step_time = step_time

    def forward(
        self,
        imageA: torch.Tensor,
        imageB: torch.Tensor,
        ages: torch.Tensor,
        ages_target: torch.Tensor,
        grid: torch.Tensor,
        loss_v: nn.Module = monai.losses.DiffusionLoss(normalize=True), # type: ignore
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
        ode_func = ODEFunction(
            self.velocity_net,
            imageA,
            imageB,
            ages[0],
            ages_target,
            identity_grid=grid,
            loss_jac=self.jacobian_loss,
            loss_v=loss_v,
        )
        zero = imageA.new_zeros(())
        phi_traj, loss_reg_traj, loss_jac_traj = odeint(
            ode_func,
            (
                grid,
                zero,
                zero.clone(),
            ),  # initial state: (phi₀, loss_reg₀, loss_jac₀)
            ages,
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
    ageA : torch.Tensor
        Scalar tensor — age at the start of the integration interval.
    ageB : torch.Tensor
        Scalar tensor — age at the end of the integration interval.
    loss_v : nn.Module
        Velocity regularisation loss module (e.g. diffusion or bending
        energy) evaluated at each ODE step.
    """

    def __init__(
        self,
        vnet: nn.Module,
        imageA: torch.Tensor,
        imageB: torch.Tensor,
        ageA: torch.Tensor,
        ageB: torch.Tensor,
        identity_grid: torch.Tensor,
        loss_jac: nn.Module,
        loss_v: nn.Module = monai.losses.DiffusionLoss(normalize=True), # type: ignore
    ) -> None:
        super().__init__()
        self.vnet = vnet
        self.imageA = imageA
        self.imageB = imageB
        self.ageA = ageA
        self.ageB = ageB
        self.identity_grid = identity_grid
        self.loss_v = loss_v
        self.loss_jac = loss_jac

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
        dphi : torch.Tensor
            Velocity sampled at the current deformation coordinates,
            ``v(t, φ_t(x))``, with shape ``(B, 3, D, H, W)``.
        loss_v : torch.Tensor
            Velocity regularisation loss at the current step (scalar),
            accumulated into the state for later retrieval.
        """
        phi_t = state[0]
        v = self.vnet(t, phi_t, self.imageA, self.imageB, self.ageA, self.ageB)
        dphi = registration.sample_vector_field(v, phi_t)
        loss_v: torch.Tensor = self.loss_v(v)
        displacement_voxel = registration.phi_to_displacement_voxel(
            phi_t, self.identity_grid
        )
        loss_jac: torch.Tensor = self.loss_jac(displacement_voxel)
        direction = torch.sign(self.ageB - self.ageA)
        return dphi, loss_v * direction, loss_jac * direction


# ──────────────────────────────────────────────────────────────────────────────
#  Velocity network
# ──────────────────────────────────────────────────────────────────────────────

class VelocityNet(nn.Module):
    """Time-conditioned 3-D U-Net that predicts a dense velocity field.

    The network concatenates three volumes — the source image *imageA*, the
    current warped image ``warp(imageA, φ_t - grid)``, and the target image
    *imageB* — into a 3-channel input and processes it through a symmetric
    encoder–decoder with skip connections.

    Temporal information is encoded by mapping the tuple
    ``(t_normalised, tA, tB)`` through sinusoidal position embeddings and
    a 3-layer SiLU MLP, producing a time embedding vector that is injected
    into every encoder and decoder block via feature-wise modulation.

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

    def __init__(
        self,
        reg_head_chan: int = 16,
        shape: list[int] = [192, 224, 192],
        t_dim: int = 48,
        t_dim_enc: int = 16,
    ) -> None:
        super().__init__()
        self.shape = shape
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
        self.time_mlp = nn.Sequential(
            nn.Linear(self.t_dim_enc, self.t_dim, bias=True),
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
        phi_t: torch.Tensor,
        image_A: torch.Tensor,
        image_B: torch.Tensor,
        ageA: torch.Tensor,
        ageB: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the velocity field at integration time *t*.

        The normalised time ``(t - ageA) / (ageB - ageA)`` is used so the
        network receives a value in ``[0, 1]`` regardless of the absolute
        age range, making it easier to learn temporal patterns across
        different developmental windows.

        Parameters
        ----------
        t : torch.Tensor
            Current integration time, broadcastable to ``(B,)``.
        phi_t : torch.Tensor
            Current deformation field of shape ``(B, 3, D, H, W)`` in
            normalised ``[-1, 1]`` coordinates.
        image_A : torch.Tensor
            Source image ``(B, 1, H, W, D)``.
        image_B : torch.Tensor
            Target image ``(B, 1, H, W, D)``.
        ageA : torch.Tensor
            Start age of the integration interval, broadcastable to ``(B,)``.
        ageB : torch.Tensor
            End age of the integration interval, broadcastable to ``(B,)``.

        Returns
        -------
        v : torch.Tensor
            Predicted velocity field of shape ``(B, 3, D, H, W)``.
        """
        warped = registration.warp_with_phi(image_A, phi_t)
        net_input = torch.cat([image_A, warped, image_B], dim=1)
        B: int = phi_t.shape[0]

        if t.dim() == 0:
            t = t.expand(B)
        if ageA.dim() == 0:
            ageA = ageA.expand(B)
        if ageB.dim() == 0:
            ageB = ageB.expand(B)

        t_enc: torch.Tensor = self.temp_enc((t - ageA) / (ageB - ageA + 1e-5))
        #ageA_enc: torch.Tensor = self.temp_enc(ageA)
        #ageB_enc: torch.Tensor = self.temp_enc(ageB)
        t_all: torch.Tensor = torch.cat([t_enc], dim=1)
        t_all = self.time_mlp(t_all)

        feat_maps = self.encoder(net_input, t_all)
        v = self.decoder_0(feat_maps[4], feat_maps[3], t_all)
        v = self.decoder_1(v, feat_maps[2], t_all)
        v = self.decoder_2(v, feat_maps[1], t_all)
        v = self.decoder_3(v, feat_maps[0], t_all)
        v = self.reg_head(v)
        return v
