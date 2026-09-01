import torch
import matplotlib.pyplot as plt
from PIL import Image
import io


def plt_grid(xy: torch.Tensor, ratio=0.8):
    """
    Plots the 2D grid
    Args:
        xy (torch.Tensor): generated grids [h, w]
    """

    # Figure size in inches = pixels / DPI
    dpi = 100
    width_px = int(1000 * ratio)
    height_px = 1000

    # Create figure with exact size
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Fill entire canvas

    # Set black background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw grid lines in white
    for i in range(xy.shape[0]):
        ax.plot(xy[i, :, 0], xy[i, :, 1], '-', lw=1.3, color='k')
    for j in range(xy.shape[1]):
        ax.plot(xy[:, j, 0], xy[:, j, 1], '-', lw=1.3, color='k')

    ax.set_xlim(xy[..., 0].min(), xy[..., 0].max())
    ax.set_ylim(xy[..., 1].min(), xy[..., 1].max())
    ax.invert_yaxis()  # Optional, depends on how your grid is laid out

    # Hide all axis decorations
    ax.axis('off')

    # Save to a PIL Image using in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches=None, pad_inches=0, transparent=False)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    buf.close()

    return image, fig
