"""Visualization utilities for the benchmark framework.

Provides functions for depth map visualization using color maps.
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def visualize_depth_colormap(
    depth: np.ndarray,
    min_depth: float,
    max_depth: float,
    colormap: str = 'turbo',
    invalid_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """Convert depth map to pseudo-color visualization.

    Args:
        depth: HxW depth map (meters)
        min_depth: Minimum depth value for normalization
        max_depth: Maximum depth value for normalization
        colormap: Colormap name ('turbo', 'viridis', 'jet', etc.)
        invalid_color: RGB color for invalid depth values (≤0 or non-finite)

    Returns:
        HxWx3 RGB image (uint8, 0-255 range)

    Note:
        Depth values are normalized to [0, 1] range using min_depth and
        max_depth, then mapped to colors using the specified colormap.
        Values outside the range are clipped.
    """
    # Create valid mask
    valid_mask = (depth > 0) & np.isfinite(depth)

    # Normalize depth to [0, 1]
    depth_normalized = np.zeros_like(depth)
    depth_range = max_depth - min_depth
    if depth_range > 1e-6:
        depth_normalized[valid_mask] = np.clip(
            (depth[valid_mask] - min_depth) / depth_range,
            0, 1
        )

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(depth_normalized)[:, :, :3]  # RGB, drop alpha channel
    colored = (colored * 255).astype(np.uint8)

    # Handle invalid pixels
    colored[~valid_mask] = invalid_color

    return colored


def save_depth_visualization(
    depth: np.ndarray,
    output_file: Path,
    min_depth: float,
    max_depth: float,
    colormap: str = 'turbo'
) -> None:
    """Save depth map as pseudo-color visualization.

    Args:
        depth: HxW depth map (meters)
        output_file: Output JPEG file path
        min_depth: Minimum depth value for normalization
        max_depth: Maximum depth value for normalization
        colormap: Colormap name

    Note:
        Output is saved as JPEG with 95% quality.
        Parent directories are created automatically if needed.
    """
    from PIL import Image

    colored = visualize_depth_colormap(depth, min_depth, max_depth, colormap)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(colored)
    img.save(output_file, quality=95)
