"""Depth projection and point cloud utilities.

Provides functions for converting depth maps to point clouds and
merging multiple point clouds.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def depth_to_point_cloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    c2w: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    confidence_percentile: float = 0.0
) -> Dict[str, np.ndarray]:
    """Convert depth map to point cloud using camera intrinsics.

    This function backprojects a depth map to 3D space using pinhole camera
    model. Points can optionally be transformed to world coordinates and
    colored using RGB values.

    Args:
        depth: HxW depth map (meters, Z-plane depth)
        intrinsics: 3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                   or 1D array [fx, fy, cx, cy]
        c2w: Optional 4x4 C2W matrix to transform points to world coordinates.
            If None, points remain in camera coordinates.
        mask: Optional HxW boolean mask to select valid pixels (True for valid)
        rgb: Optional HxWx3 RGB image (uint8, 0-255 range)
        confidence: Optional HxW confidence map (any value range)
        confidence_percentile: Percentile threshold (0.0-1.0). E.g., 0.3 filters lowest 30% confidence points

    Returns:
        Point cloud dictionary with structure:
            - 'xyz': Nx3 float32 array - 3D coordinates (REQUIRED)
            - 'rgb': Nx3 float32 array - colors in [0,1] range (if rgb provided)
            - 'confidence': N float32 array - confidence scores (if confidence provided)

    Note:
        - Invalid depth values (≤0 or non-finite) are automatically filtered
        - Z-plane depth is assumed (distance along camera Z-axis, not planar)
        - Percentile is computed only on points with valid depth to avoid bias from invalid regions
    """
    h, w = depth.shape

    # Parse intrinsics
    if intrinsics.shape == (3, 3):
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    else:
        fx, fy, cx, cy = intrinsics

    # Create coordinate grids
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    # Backproject to 3D (camera coordinates)
    # Pinhole camera model:
    #   X = (u - cx) * Z / fx
    #   Y = (v - cy) * Z / fy
    #   Z = depth
    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    # Stack into Nx3 point cloud
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Prepare RGB colors if provided
    colors = None
    if rgb is not None:
        # Normalize RGB from [0, 255] to [0, 1]
        if rgb.dtype == np.uint8:
            rgb_normalized = rgb.astype(np.float32) / 255.0
        else:
            rgb_normalized = np.clip(rgb, 0, 1).astype(np.float32)
        colors = rgb_normalized.reshape(-1, 3)

    # Prepare confidence if provided
    conf_values = None
    if confidence is not None:
        conf_values = confidence.reshape(-1).astype(np.float32)

    # Build valid mask from depth
    if mask is not None:
        valid = mask.reshape(-1) > 0
    else:
        valid = (z.reshape(-1) > 0) & np.isfinite(z.reshape(-1))

    # Apply confidence percentile filtering if provided
    if confidence is not None and confidence_percentile > 0.0:
        # Compute actual threshold from percentile (e.g., 0.3 -> 30th percentile)
        # Only compute percentile on valid depth points
        threshold_value = np.percentile(conf_values[valid], confidence_percentile * 100)
        conf_valid = conf_values >= threshold_value
        valid = valid & conf_valid

    # Filter points, colors, and confidence
    points = points[valid]
    if colors is not None:
        colors = colors[valid]
    if conf_values is not None:
        conf_values = conf_values[valid]

    # Transform to world coordinates if c2w is provided
    if c2w is not None:
        rotation = c2w[:3, :3]
        translation = c2w[:3, 3]
        points = points @ rotation.T + translation[np.newaxis, :]

    # Build result dictionary
    result = {'xyz': points}
    if colors is not None:
        result['rgb'] = colors
    if conf_values is not None:
        result['confidence'] = conf_values

    return result


def merge_point_clouds(
    point_clouds: List[Dict[str, np.ndarray]],
    max_points: int = -1
) -> Dict[str, np.ndarray]:
    """Merge multiple point clouds and optionally limit to max_points.

    Args:
        point_clouds: List of point cloud dictionaries with 'xyz' (required)
                     and optional 'rgb', 'confidence', etc.
        max_points: Maximum number of points to keep (-1 = unlimited)

    Returns:
        Merged point cloud dictionary with all common attributes

    Note:
        - When max_points is exceeded, points are randomly sampled
        - Only merges attributes present in ALL input clouds
    """
    if not point_clouds:
        return {'xyz': np.empty((0, 3), dtype=np.float32)}

    # Find common keys across all point clouds
    common_keys = set(point_clouds[0].keys())
    for pc in point_clouds[1:]:
        common_keys &= set(pc.keys())

    # Merge each attribute
    result = {}
    for key in common_keys:
        arrays = [pc[key] for pc in point_clouds]
        result[key] = np.vstack(arrays) if arrays[0].ndim == 2 else np.concatenate(arrays)

    # Limit to max_points if specified
    if max_points > 0 and len(result['xyz']) > max_points:
        indices = np.random.choice(len(result['xyz']), size=max_points, replace=False)
        for key in result:
            result[key] = result[key][indices]

    return result


def compute_depth_range(depths: List[np.ndarray]) -> Tuple[float, float]:
    """Compute depth range across multiple depth maps.

    Uses 1st and 99th percentiles to exclude outliers.

    Args:
        depths: List of HxW depth maps

    Returns:
        (min_depth, max_depth) tuple based on 1st and 99th percentiles

    Note:
        Only valid depth values (>0 and finite) are considered.
    """
    all_depths = []
    for depth in depths:
        valid = (depth > 0) & np.isfinite(depth)
        all_depths.extend(depth[valid].flatten())

    all_depths = np.array(all_depths)
    return float(np.percentile(all_depths, 1)), float(np.percentile(all_depths, 99))
