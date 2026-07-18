"""Point cloud I/O utilities for binary PLY format.

Provides functions for reading and writing point clouds in binary PLY format
using the plyfile library.
"""

from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement


def save_point_cloud_ply(points: np.ndarray, output_file: Path) -> None:
    """Save point cloud to binary PLY format.

    Args:
        points: Numpy array with shape [..., 3/4/6/7]
            - [..., 3]: xyz coordinates only
            - [..., 4]: xyz coordinates + distance
            - [..., 6]: xyz coordinates + rgb colors in [0, 1] range
            - [..., 7]: xyz coordinates + rgb colors + distance
        output_file: Output PLY file path

    Note:
        - xyz stored as float64 (Open3D format)
        - RGB values converted from [0, 1] float to [0, 255] uint8
        - RGB field names: 'red', 'green', 'blue' (Open3D format)
        - Distance values stored as float32 (meters)
    """
    if not isinstance(points, np.ndarray):
        raise ValueError("Points must be a numpy array")

    if points.ndim < 2 or points.shape[-1] not in [3, 4, 6, 7]:
        raise ValueError(f"Points must have shape [..., 3/4/6/7], got {points.shape}")

    points = points.reshape(-1, points.shape[-1])
    n_dims = points.shape[1]

    if n_dims == 3:
        # xyz only
        dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
        xyz = points.astype(np.float64)
        arr = np.array([tuple(row) for row in xyz], dtype=dtype)
    elif n_dims == 4:
        # xyz + distance
        dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('distance', 'f4')]
        data = points.copy()
        data[:, :3] = data[:, :3].astype(np.float64)
        data[:, 3] = data[:, 3].astype(np.float32)
        arr = np.array([tuple(row) for row in data], dtype=dtype)
    elif n_dims == 6:
        # xyz + rgb
        dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        xyz = points[:, :3].astype(np.float64)
        rgb = (points[:, 3:6] * 255).clip(0, 255).astype(np.uint8)
        arr = np.array([tuple(list(xyz[i]) + list(rgb[i])) for i in range(len(points))], dtype=dtype)
    else:  # n_dims == 7
        # xyz + rgb + distance
        dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('distance', 'f4')]
        xyz = points[:, :3].astype(np.float64)
        rgb = (points[:, 3:6] * 255).clip(0, 255).astype(np.uint8)
        dist = points[:, 6:7].astype(np.float32)
        arr = np.array([tuple(list(xyz[i]) + list(rgb[i]) + list(dist[i])) for i in range(len(points))], dtype=dtype)

    el = PlyElement.describe(arr, 'vertex')
    PlyData([el], text=False, comments=['Created by Open3D']).write(output_file)


def load_point_cloud_ply(input_file: Path) -> np.ndarray:
    """Load point cloud from binary PLY file (Open3D compatible).

    Args:
        input_file: Input PLY file path

    Returns:
        Numpy array with shape [N, 3/4/6/7]
        - [N, 3]: xyz coordinates only
        - [N, 4]: xyz coordinates + distance
        - [N, 6]: xyz coordinates + rgb colors in [0, 1] range
        - [N, 7]: xyz coordinates + rgb colors + distance
        
    Note:
        - Supports Open3D format: 'red', 'green', 'blue' (uint8 0-255)
        - Also supports legacy format: 'r', 'g', 'b' (float32 0-1)
    """
    ply = PlyData.read(input_file)
    data = ply['vertex'].data
    props = data.dtype.names

    xyz = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)
    
    # Check for Open3D format (red, green, blue) or legacy format (r, g, b)
    has_color_open3d = 'red' in props and 'green' in props and 'blue' in props
    has_color_legacy = 'r' in props and 'g' in props and 'b' in props
    has_dist = 'distance' in props

    parts = [xyz]
    if has_color_open3d:
        # Open3D format: uint8 [0, 255] -> float32 [0, 1]
        rgb = np.stack([data['red'], data['green'], data['blue']], axis=1).astype(np.float32) / 255.0
        parts.append(rgb)
    elif has_color_legacy:
        # Legacy format: float32 [0, 1]
        parts.append(np.stack([data['r'], data['g'], data['b']], axis=1).astype(np.float32))
    
    if has_dist:
        parts.append(data['distance'].astype(np.float32).reshape(-1, 1))

    return np.hstack(parts)
