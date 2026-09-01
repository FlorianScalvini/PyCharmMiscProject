from typing import Any, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import pytorch_lightning as pl
import numpy as np
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
import torch.utils.checkpoint as checkpoint
import monai
from torch.nn.utils import spectral_norm as sn
from torch import Tensor
import matplotlib.pyplot as plt
from PIL import Image
import io




def compute_jacobian_determinant_3d(displacement, spacing=(1.0, 1.0, 1.0)):
    """
    Compute the Jacobian determinant of a 3D displacement field.

    Parameters:
    - displacement: torch.Tensor of shape (3, D, H, W), representing the displacement field.
    - spacing: tuple of floats (dz, dy, dx), representing the physical spacing between voxels.

    Returns:
    - jacobian_determinant: torch.Tensor of shape (D, H, W), the Jacobian determinant at each point.
    """
    displacement = displacement.squeeze(0)

    dz, dy, dx = spacing
    grads = []
    for i in range(3):  # u_x, u_y, u_z
        grad_i = torch.gradient(displacement[i], spacing=(dz, dy, dx), dim=(0, 1, 2))
        grads.append(grad_i)

    jacobian = torch.stack([torch.stack(grad_k, dim=-1) for grad_k in grads], dim=-1)
    # Add identity to convert ∂φ/∂x = I + ∂u/∂x
    identity = torch.eye(3).to(displacement.device)
    jacobian = jacobian + identity
    det_j = torch.linalg.det(jacobian)
    return det_j.unsqueeze(0)


def normalize_to_0_1(volume):
    '''
        Normalize volume to 0-1 range
    '''
    max_val = volume.max()
    min_val = volume.min()
    return (volume - min_val) / (max_val - min_val)

