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
  image volume (``warp``).

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
from monai.networks.blocks.warp import Warp


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
    return normalized_grid


# ──────────────────────────────────────────────────────────────────────────────
#  Image warping
# ──────────────────────────────────────────────────────────────────────────────

def warp(image: Tensor, flow: Tensor, mode: str = 'bilinear') -> Tensor:
    """Deform *image* using the voxel-space displacement field *flow*.

    Internally delegates to ``monai.networks.blocks.Warp``, which calls
    ``F.grid_sample`` after converting *flow* to a normalised grid.
    Out-of-bounds coordinates are handled with reflection padding to avoid
    black border artefacts.

    Args:
        image: Moving image of shape ``(B, C, D, H, W)``.
        flow:  Displacement field of shape ``(B, 3, D, H, W)`` in voxel units,
               with the same spatial extent as *image*.
        mode:  Interpolation mode forwarded to ``grid_sample``
               (``'bilinear'`` or ``'nearest'``).  Defaults to ``'bilinear'``.

    Returns:
        Warped image of shape ``(B, C, D, H, W)`` on the same device as
        *image*.
    """
    warper = Warp(mode=mode, padding_mode='reflection')
    return warper(image, flow)
