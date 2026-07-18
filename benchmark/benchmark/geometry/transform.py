"""Transformation matrix operations."""

import numpy as np


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert 4x4 SE3 transformation matrix using R^T (assumes orthogonal R).

    Uses the efficient SE3 inverse: [R^T | -R^T @ t] which is valid when R
    is a proper rotation matrix. This matches DA3's closed_form_inverse_se3.

    For non-orthogonal R (e.g., raw KinectFusion poses), use np.linalg.inv()
    directly instead of this function.

    Args:
        T: (..., 4, 4) transformation matrix

    Returns:
        Inverted matrix of same shape
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3:]

    R_inv = np.swapaxes(R, -2, -1)  # R^T
    t_inv = -R_inv @ t

    result = np.zeros_like(T)
    result[..., :3, :3] = R_inv
    result[..., :3, 3:] = t_inv
    result[..., 3, 3] = 1.0
    return result
