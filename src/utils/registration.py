"""
Spatial registration utilities for 3-D brain MRI.

Provides the low-level building blocks used throughout the longitudinal
registration pipeline:

* **Grid generation** – create identity coordinate grids in voxel space
  (``get_reference_grid``, ``generate_grid3d_tensor``) or in normalised
  [-1, 1] space (``generate_grid3d_tensor``).
* **Coordinate conversion** – map a voxel-space displacement field to a
  normalised sampling grid compatible with ``torch.nn.functional.grid_sample``
  (``displacement2grid``).
* **Image warping** – apply a voxel-space displacement field to deform an
  image volume (``warp_with_phi``).

Coordinate convention
---------------------
All displacement / flow tensors follow the shape convention
``(B, 3, D, H, W)``, where the three channels correspond to the x, y, z
(or i, j, k) spatial axes.  Grids produced by ``generate_grid3d_tensor``
are ordered ``(z, y, x)`` to match PyTorch's ``grid_sample`` expectation.

Author : Florian Scalvini
"""

# --- Standard library ---
from collections.abc import Sequence

# --- Third-party ---
import torch
from torch import Tensor
from torch.nn import functional as F


# ──────────────────────────────────────────────────────────────────────────────
#  Grid generation
# ──────────────────────────────────────────────────────────────────────────────

def meshgrid_ij(*tensors: Tensor) -> tuple[Tensor, ...]:
    """Thin compatibility wrapper around ``torch.meshgrid`` that always uses IJ indexing.

    ``indexing='ij'`` was introduced in PyTorch 1.10. This helper detects the
    API version at runtime and falls back to the legacy call on older builds,
    ensuring consistent row-major (matrix) index ordering regardless of the
    installed version.
    """
    if torch.meshgrid.__kwdefaults__ is not None and "indexing" in torch.meshgrid.__kwdefaults__:
        return torch.meshgrid(*tensors, indexing="ij")  # new api pytorch after 1.10
    return torch.meshgrid(*tensors)


def get_reference_grid(ddf: Tensor) -> Tensor:
    """Build an identity voxel-space coordinate grid matching the spatial extent of *ddf*.

    Each voxel position ``(i, j, k)`` is filled with its own integer index,
    so the grid represents the identity deformation (no displacement).

    Args:
        ddf: Displacement / flow tensor of shape ``(B, 3, D, H, W)``.
             Only the batch size and spatial dimensions are used; the channel
             values are ignored.

    Returns:
        Tensor of shape ``(B, 3, D, H, W)`` in voxel coordinates, on the
        same device and dtype as *ddf*.
    """
    mesh_points = [torch.arange(0, dim) for dim in ddf.shape[2:]]
    grid = torch.stack(meshgrid_ij(*mesh_points), dim=0)  # (spatial_dims, ...)
    grid = torch.stack([grid] * ddf.shape[0], dim=0)  # (batch, spatial_dims, ...)
    ref_grid = grid.to(ddf)
    return ref_grid


def generate_grid3d_tensor(shape: Sequence[int]) -> Tensor:
    """Create a 3-D identity grid normalised to [-1, 1] for a given spatial shape.

    Produces the canonical sampling grid used as the initial state of the
    deformation field: every position maps to itself in normalised coordinates.
    The channel order is ``(z, y, x)`` to match ``torch.nn.functional.grid_sample``.

    Args:
        shape: Spatial dimensions ``(D, H, W)`` of the target volume
               (tuple or list of three ints).

    Returns:
        Tensor of shape ``(3, D, H, W)`` with values in ``[-1, 1]``.
    """
    x = torch.linspace(-1., 1., shape[0])
    y = torch.linspace(-1., 1., shape[1])
    z = torch.linspace(-1., 1., shape[2])
    x, y, z = torch.meshgrid(x, y, z, indexing='ij')
    return torch.stack([z, y, x], dim=0)   # (3, D, H, W)


def sample_vector_field(vector_field: Tensor, phi: Tensor) -> Tensor:
    """Sample a vector field at the coordinates of a deformation map.

    Both tensors use shape ``(B, 3, D, H, W)``. ``phi`` contains absolute
    coordinates normalized to ``[-1, 1]`` and ordered as expected by
    :func:`torch.nn.functional.grid_sample`.
    """
    if vector_field.shape != phi.shape:
        raise ValueError(
            "vector_field and phi must have the same shape, got "
            f"{tuple(vector_field.shape)} and {tuple(phi.shape)}"
        )
    return warp_with_phi(vector_field, phi)


def warp_with_phi(image: Tensor, phi: Tensor, mode: str = "bilinear") -> Tensor:
    """Warp an image with an absolute deformation in normalized coordinates."""
    if image.ndim != 5 or phi.ndim != 5 or phi.shape[1] != 3:
        raise ValueError(
            "expected image (B,C,D,H,W) and phi (B,3,D,H,W), got "
            f"{tuple(image.shape)} and {tuple(phi.shape)}"
        )
    sampling_grid = phi.permute(0, 2, 3, 4, 1)
    return F.grid_sample(
        image,
        sampling_grid,
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Coordinate conversion
# ──────────────────────────────────────────────────────────────────────────────

def displacement2grid(flow: Tensor) -> Tensor:
    """Convert a voxel-space displacement field to a normalised sampling grid.

    Adds the identity reference grid to the displacement field to obtain the
    absolute voxel positions ``φ = id + u``, then normalises each spatial axis
    to ``[-1, 1]`` so the result can be passed directly to
    ``torch.nn.functional.grid_sample``.

    Args:
        flow: Displacement field of shape ``(B, 2, H, W)`` (2-D) or
              ``(B, 3, D, H, W)`` (3-D) in voxel units.

    Returns:
        Normalised sampling grid of shape ``(B, H, W, 2)`` or
        ``(B, D, H, W, 3)`` with values in ``[-1, 1]``, compatible with
        ``F.grid_sample(..., align_corners=True)``.

    Raises:
        NotImplementedError: If *flow* has a spatial dimensionality other than
            2 or 3.
    """
    spatial_dims = len(flow.shape) - 2
    if spatial_dims not in (2, 3):
        raise NotImplementedError(f"got unsupported spatial_dims={spatial_dims}, currently support 2 or 3.")
    grid = get_reference_grid(flow).to(flow.device) + flow

    grid = grid.permute([0] + list(range(2, 2 + spatial_dims)) + [1])
    normalized_grid = grid.clone()
    for i, dim in enumerate(normalized_grid.shape[1:-1]):
        normalized_grid[..., i] = normalized_grid[..., i] * 2 / (dim - 1) - 1
    return normalized_grid.flip(-1)



def phi_to_displacement_voxel(phi: Tensor, identity: Tensor | None = None) -> Tensor:
    """Convert normalized XYZ maps to voxel displacements ordered D,H,W.

    This conversion is reserved for Jacobian calculations and flow exports.
    Warping uses the normalized absolute map directly.
    """
    if phi.ndim != 5 or phi.shape[1] != 3:
        raise ValueError("phi must have shape (B,3,D,H,W)")
    if identity is None:
        identity = generate_grid3d_tensor(phi.shape[2:]).to(phi).unsqueeze(0)
    scale = phi.new_tensor([size - 1 for size in phi.shape[2:]]).view(1, 3, 1, 1, 1) / 2
    return (phi - identity).flip(1) * scale
