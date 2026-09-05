# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Visualization utility functions for colorization and color bars.
"""

import dataclasses
from typing import Optional, Tuple

import numpy as np
import torch
import cv2
import matplotlib
import matplotlib.cm as cm


@dataclasses.dataclass
class CameraState:
    """Camera state for rendering."""
    fov: float
    aspect: float
    c2w: np.ndarray

    def get_K(self, img_wh: Tuple[int, int]) -> np.ndarray:
        """Get camera intrinsic matrix from FOV and image size."""
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array([
            [focal_length, 0.0, W / 2.0],
            [0.0, focal_length, H / 2.0],
            [0.0, 0.0, 1.0],
        ])
        return K


def get_vertical_colorbar(
    h: int,
    vmin: float,
    vmax: float,
    cmap_name: str = "jet",
    label: Optional[str] = None,
    cbar_precision: int = 2
) -> np.ndarray:
    """
    Create a vertical colorbar image.

    Args:
        h: Height in pixels
        vmin: Minimum value
        vmax: Maximum value
        cmap_name: Colormap name
        label: Optional label for the colorbar
        cbar_precision: Decimal precision for tick labels

    Returns:
        Colorbar image as numpy array (H, W, 3)
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib as mpl

    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    ax = fig.add_subplot(111)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, ticks=tick_loc, orientation="vertical"
    )

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)
    cb1.ax.tick_params(labelsize=18, rotation=0)
    if label is not None:
        cb1.set_label(label)

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    im = im[:, :, :3].astype(np.float32) / 255.0

    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(
    x: np.ndarray,
    cmap_name: str = "jet",
    mask: Optional[np.ndarray] = None,
    range: Optional[Tuple[float, float]] = None,
    append_cbar: bool = False,
    cbar_in_image: bool = False,
    cbar_precision: int = 2,
) -> np.ndarray:
    """
    Turn a grayscale image into a color image.

    Args:
        x: Input grayscale image [H, W]
        cmap_name: Colormap name
        mask: Optional mask image [H, W]
        range: Value range for scaling [min, max], automatic if None
        append_cbar: Whether to append colorbar
        cbar_in_image: Put colorbar inside image
        cbar_precision: Colorbar tick precision

    Returns:
        Colorized image [H, W, 3]
    """
    if range is not None:
        vmin, vmax = range
    elif mask is not None:
        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])
        x[np.logical_not(mask)] = vmin
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += 1e-6

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)

    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1.0 - mask)

    cbar = get_vertical_colorbar(
        h=x.shape[0],
        vmin=vmin,
        vmax=vmax,
        cmap_name=cmap_name,
        cbar_precision=cbar_precision,
    )

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1]:, :] = cbar
        else:
            x_new = np.concatenate(
                (x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1
            )
        return x_new
    else:
        return x_new


def colorize(
    x: torch.Tensor,
    cmap_name: str = "jet",
    mask: Optional[torch.Tensor] = None,
    range: Optional[Tuple[float, float]] = None,
    append_cbar: bool = False,
    cbar_in_image: bool = False
) -> torch.Tensor:
    """
    Turn a grayscale image into a color image (PyTorch tensor version).

    Args:
        x: Grayscale image tensor [H, W] or [B, H, W]
        cmap_name: Colormap name
        mask: Optional mask tensor [H, W] or [B, H, W]
        range: Value range for scaling
        append_cbar: Whether to append colorbar
        cbar_in_image: Put colorbar inside image

    Returns:
        Colorized tensor
    """
    device = x.device
    x = x.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy() > 0.99
        kernel = np.ones((3, 3), np.uint8)

    if x.ndim == 2:
        x = x[None]
        if mask is not None:
            mask = mask[None]

    out = []
    for x_ in x:
        if mask is not None:
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        x_ = colorize_np(x_, cmap_name, mask, range, append_cbar, cbar_in_image)
        out.append(torch.from_numpy(x_).to(device).float())
    out = torch.stack(out).squeeze(0)
    return out
