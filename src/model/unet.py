"""
3-D U-Net building blocks for time-conditioned volumetric feature extraction.

Architecture overview
---------------------
The module provides a modular encoder–decoder U-Net adapted for 3-D medical
image volumes.  Two design elements differentiate it from the vanilla U-Net:

* **FiLM temporal modulation** — every encoder and decoder block optionally
  accepts a time-embedding vector and applies Feature-wise Linear Modulation
  (FiLM): ``out = γ(t) * out + β(t)``, where γ and β are learned linear
  projections of the embedding.  This allows the network to condition its
  feature responses on an arbitrary continuous time variable without
  architectural changes.

* **Sinusoidal time encoding** — the raw time value is encoded outside this
  module (see :mod:`model.time_encoding`) before being passed to the blocks,
  following the approach used in diffusion models.

References
----------
**U-Net** — base encoder–decoder architecture with skip connections:

    Ronneberger, O., Fischer, P., & Brox, T. (2015).
    *U-Net: Convolutional Networks for Biomedical Image Segmentation.*
    MICCAI 2015. https://arxiv.org/abs/1505.04597

**TimeFlow** — FiLM-based time conditioning injected into U-Net blocks:

    Bai, L. et al. (2023).
    *TimeFlow: Time-Conditioned Normalizing Flow for Longitudinal Medical
    Image Registration.*
    GitHub: https://github.com/BailiangJ/TimeFlow/

Classes
-------
Unet
    Full encoder-decoder U-Net with optional temporal modulation.
EncoderUnet
    Encoder arm: successive strided convolution blocks returning a feature pyramid.
DecoderUnet
    Decoder arm: upsampling blocks with skip connections reconstructing spatial resolution.
UnetBlock
    Single U-Net stage (two Conv-BN-ReLU layers) with optional FiLM temporal modulation.
UnetUpBlock
    Upsampling block that fuses a decoded feature map with its encoder skip connection.
Conv3dReLU
    Conv3d → InstanceNorm3d → LeakyReLU building brick.

Author : Florian Scalvini
"""

# --- Third-party ---
import torch
import torch.nn.functional as F  # noqa: F401
from torch import nn
from collections.abc import Sequence


# ──────────────────────────────────────────────────────────────────────────────
#  Full U-Net
# ──────────────────────────────────────────────────────────────────────────────

class Unet(nn.Module):
    """Full encoder-decoder U-Net with optional time conditioning.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    channels : sequence of int
        Channel widths at each encoder stage.
    out_channels : int
        Number of output channels produced by the final 1×1×1 convolution.
    t_dim : int or None
        Dimensionality of the time embedding.  Pass ``None`` to disable
        temporal modulation.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        out_channels: int,
        t_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = EncoderUnet(in_channels, channels, t_dim)
        self.decoder = DecoderUnet(channels, out_channels, t_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Run a full forward pass through encoder and decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input volume of shape ``(B, C, D, H, W)``.
        t : torch.Tensor or None
            Time embedding of shape ``(B, t_dim)``, or ``None`` if temporal
            modulation is disabled.

        Returns
        -------
        out : torch.Tensor
            Output volume of shape ``(B, out_channels, D, H, W)``.
        """
        feats = self.encoder(x, t)
        out = self.decoder(feats, t)
        return out


# ──────────────────────────────────────────────────────────────────────────────
#  Encoder / Decoder arms
# ──────────────────────────────────────────────────────────────────────────────

class EncoderUnet(nn.Module):
    """Encoder arm of the U-Net.

    Applies a series of :class:`UnetBlock` stages with progressively halved
    spatial resolution (stride-2 from the second stage onward) and returns
    the full feature pyramid.

    Based on the encoder design of the original U-Net
    (`Ronneberger et al., 2015 <https://arxiv.org/abs/1505.04597>`_).

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    channels : sequence of int
        Output channel width for each encoder stage.
    t_dim : int or None
        Dimensionality of the time embedding.  Pass ``None`` to disable
        temporal modulation.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        t_dim: int | None = None,
    ) -> None:
        super().__init__()
        ch = [in_channels] + list(channels)
        self.encoder = nn.ModuleList([
            UnetBlock(
                ch[i], ch[i + 1], ch[i + 1],
                kernel_size=3, padding=1,
                stride=1 if i == 0 else 2,
                t_dim=t_dim,
            )
            for i in range(len(channels))
        ])

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Encode an input volume into a feature pyramid.

        Parameters
        ----------
        x : torch.Tensor
            Input volume of shape ``(B, C, D, H, W)``.
        t : torch.Tensor or None
            Time embedding of shape ``(B, t_dim)``.

        Returns
        -------
        feats : list of torch.Tensor
            Feature maps at each encoder stage, ordered from shallowest to
            deepest.
        """
        feats: list[torch.Tensor] = []
        for stage in self.encoder:
            x = stage(x, t)
            feats.append(x)
        return feats


class DecoderUnet(nn.Module):
    """Decoder arm of the U-Net.

    Progressively upsamples the bottleneck feature map while fusing encoder
    skip connections, then projects to ``out_channels`` with a 1×1×1
    convolution.

    Based on the decoder design of the original U-Net
    (`Ronneberger et al., 2015 <https://arxiv.org/abs/1505.04597>`_).

    Parameters
    ----------
    channels : sequence of int
        Channel widths matching those used in :class:`EncoderUnet`.
    out_channels : int
        Number of output channels.
    t_dim : int or None
        Dimensionality of the time embedding.  Pass ``None`` to disable
        temporal modulation.
    """

    def __init__(
        self,
        channels: Sequence[int],
        out_channels: int,
        t_dim: int | None = None,
    ) -> None:
        super().__init__()
        ch = list(channels)
        self.decoder = nn.ModuleList([
            UnetUpBlock(ch[i], ch[i + 1], kernel_size=3, t_dim=t_dim)
            for i in range(len(channels) - 1)
        ])
        self.final_conv = nn.Conv3d(ch[-1], out_channels, kernel_size=1)

    def forward(
        self,
        feats: Sequence[torch.Tensor],
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode a feature pyramid into an output volume.

        Parameters
        ----------
        feats : sequence of torch.Tensor
            Feature maps from :class:`EncoderUnet`, ordered from shallowest
            to deepest (``feats[-1]`` is the bottleneck).
        t : torch.Tensor or None
            Time embedding of shape ``(B, t_dim)``.

        Returns
        -------
        out : torch.Tensor
            Decoded output of shape ``(B, out_channels, D, H, W)``.
        """
        x = feats[-1]
        for i, up_block in enumerate(self.decoder):
            x_skip = feats[-(i + 2)]
            x = up_block(x, x_skip, t)
        out = self.final_conv(x)
        return out


# ──────────────────────────────────────────────────────────────────────────────
#  Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class UnetBlock(nn.Module):
    """Single U-Net encoder stage with optional FiLM temporal modulation.

    Applies two consecutive Conv-InstanceNorm-LeakyReLU layers.  When a time
    embedding is provided, the output is modulated via learned scale (γ) and
    shift (β) parameters derived from the embedding (FiLM conditioning).

    The FiLM modulation scheme is adapted from TimeFlow
    (`Bai et al., 2023 <https://github.com/BailiangJ/TimeFlow/>`_).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of intermediate channels between the two convolutions.
    out_channels : int
        Number of output channels.
    t_dim : int or None
        Dimensionality of the time embedding.  Pass ``None`` to disable
        temporal modulation.
    kernel_size : int or sequence of int
        Convolution kernel size.
    stride : int
        Stride of the first convolution (use 2 for downsampling).
    padding : int
        Padding applied to both convolutions.
    bias : bool
        Whether to add a learnable bias to the convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        t_dim: int | None,
        kernel_size: Sequence[int] | int,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.cvbact_1 = Conv3dReLU(in_channels, mid_channels, kernel_size, padding=padding, stride=stride)
        self.cvbact_2 = Conv3dReLU(mid_channels, out_channels, kernel_size, padding=padding, stride=1)
        self.spatial_dims = 3
        if t_dim is not None:
            self.time_mlp = nn.Linear(t_dim, out_channels * 2, bias=True)
            self.temporal_modulation = True
        else:
            self.temporal_modulation = False

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the block to *x* with optional temporal modulation.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map ``(B, in_channels, D, H, W)``.
        t : torch.Tensor or None
            Time embedding of shape ``(B, t_dim)``.

        Returns
        -------
        out : torch.Tensor
            Output feature map ``(B, out_channels, D, H, W)``.
        """
        out = self.cvbact_1(x)
        out = self.cvbact_2(out)
        if self.temporal_modulation:
            t_embed = self.time_mlp(t)
            spatial_shape = [1] * self.spatial_dims
            gamma, beta = t_embed.chunk(2, dim=-1)
            gamma = gamma.view(*gamma.shape, *spatial_shape)
            beta = beta.view(*beta.shape, *spatial_shape)
            out = out * gamma + beta
        return out


class UnetUpBlock(nn.Module):
    """Upsampling block that fuses a decoded map with its encoder skip connection.

    Upsamples *x* by 2× (trilinear), concatenates the result with the skip
    feature map, and applies a Conv-InstanceNorm-LeakyReLU fusion layer.
    Optionally applies FiLM temporal modulation after fusion.

    The FiLM modulation scheme is adapted from TimeFlow
    (`Bai et al., 2023 <https://github.com/BailiangJ/TimeFlow/>`_).

    Parameters
    ----------
    in_channels : int
        Number of input channels (from the deeper decoder stage).
    out_channels : int
        Number of output channels.
    kernel_size : int or sequence of int
        Kernel size used in the upsampling convolution.
    t_dim : int or None
        Dimensionality of the time embedding.  Pass ``None`` to disable
        temporal modulation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        t_dim: int | None,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]
        self.upsample = nn.Sequential(
            Conv3dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),  # type: ignore
            nn.Upsample(scale_factor=2.0, mode="trilinear", align_corners=True),
        )
        self.conv_block = Conv3dReLU(out_channels * 2, out_channels, kernel_size=1)
        self.spatial_dims = 3
        if t_dim is not None:
            self.time_mlp = nn.Linear(t_dim, out_channels * 2, bias=True)
            self.temporal_modulation = True
        else:
            self.temporal_modulation = False

    def forward(
        self,
        x: torch.Tensor,
        x_skip: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Upsample *x*, fuse with *x_skip*, and apply temporal modulation.

        Parameters
        ----------
        x : torch.Tensor
            Feature map from the deeper decoder stage
            ``(B, in_channels, D, H, W)``.
        x_skip : torch.Tensor
            Encoder skip-connection feature map
            ``(B, out_channels, 2D, 2H, 2W)``.
        t : torch.Tensor or None
            Time embedding of shape ``(B, t_dim)``.

        Returns
        -------
        out : torch.Tensor
            Fused output feature map ``(B, out_channels, 2D, 2H, 2W)``.
        """
        out = self.upsample(x)
        out = torch.cat((out, x_skip), dim=1)
        out = self.conv_block(out)
        if self.temporal_modulation:
            t_embed = self.time_mlp(t)
            spatial_shape = [1] * self.spatial_dims
            gamma, beta = t_embed.chunk(2, dim=-1)
            gamma = gamma.view(*gamma.shape, *spatial_shape)
            beta = beta.view(*beta.shape, *spatial_shape)
            out = out * gamma + beta
        return out


class Conv3dReLU(nn.Sequential):
    """Conv3d → InstanceNorm3d → LeakyReLU building brick.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or sequence of int
        Convolution kernel size.
    padding : int
        Zero-padding added to all spatial sides.
    stride : int
        Convolution stride.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False) # type: ignore
        self.nm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalisation, and activation.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map ``(B, in_channels, D, H, W)``.

        Returns
        -------
        out : torch.Tensor
            Activated output ``(B, out_channels, D', H', W')``.
        """
        out = self.conv(x)
        out = self.nm(out)
        out = self.relu(out)
        return out
