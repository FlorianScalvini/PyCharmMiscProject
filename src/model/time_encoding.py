"""Sinusoidal position embeddings for scalar time or age values."""

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for scalar time or age values.

    Encodes a batch of scalar inputs into a fixed-length embedding by
    projecting onto a bank of sinusoids with geometrically spaced
    frequencies, following the scheme introduced in *Attention Is All You
    Need* (Vaswani et al., 2017).

    Parameters
    ----------
    embed_dim : int
        Output embedding dimensionality.  Must be even.
    max_periods : int
        Base period used to space the frequency bank.  Larger values
        capture lower-frequency (longer-range) variation.
    """

    def __init__(self, embed_dim: int, max_periods: int = 10000) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_periods = max_periods

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode scalar inputs as sinusoidal embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Scalar time or age values of shape ``(B,)``.

        Returns
        -------
        embeddings : torch.Tensor
            Sinusoidal embeddings of shape ``(B, embed_dim)``.
        """
        indices = torch.arange(0, self.embed_dim // 2, dtype=torch.float32, device=x.device)
        freqs = torch.pow(self.max_periods, -2 * indices / self.embed_dim)  # (embed_dim // 2,)
        angles = torch.einsum("b,d->bd", x, freqs)
        embeddings = torch.cat((angles.sin(), angles.cos()), dim=-1)  # (B, embed_dim)
        return embeddings
    
