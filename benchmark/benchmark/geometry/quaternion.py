"""Quaternion and rotation matrix conversions."""

import numpy as np


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix or batch of matrices to quaternion(s).

    Uses Shepperd's method for numerical stability.

    Args:
        R: (3, 3) or (..., 3, 3) rotation matrix

    Returns:
        (..., 4) array [qx, qy, qz, qw]
    """
    R = np.asarray(R, dtype=float)
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    N = len(R_flat)

    q = np.zeros((N, 4), dtype=float)
    trace = R_flat[..., 0, 0] + R_flat[..., 1, 1] + R_flat[..., 2, 2]

    # Case 1: trace > 0
    mask1 = trace > 0
    if np.any(mask1):
        s = 0.5 / np.sqrt(trace[mask1] + 1.0)
        q[mask1, 3] = 0.25 / s
        q[mask1, 0] = (R_flat[mask1][..., 2, 1] - R_flat[mask1][..., 1, 2]) * s
        q[mask1, 1] = (R_flat[mask1][..., 0, 2] - R_flat[mask1][..., 2, 0]) * s
        q[mask1, 2] = (R_flat[mask1][..., 1, 0] - R_flat[mask1][..., 0, 1]) * s

    # Case 2: R[0,0] is max diagonal
    mask2 = ~mask1 & (R_flat[..., 0, 0] > R_flat[..., 1, 1]) & (R_flat[..., 0, 0] > R_flat[..., 2, 2])
    if np.any(mask2):
        s = 2.0 * np.sqrt(1.0 + R_flat[mask2][..., 0, 0] - R_flat[mask2][..., 1, 1] - R_flat[mask2][..., 2, 2])
        q[mask2, 3] = (R_flat[mask2][..., 2, 1] - R_flat[mask2][..., 1, 2]) / s
        q[mask2, 0] = 0.25 * s
        q[mask2, 1] = (R_flat[mask2][..., 0, 1] + R_flat[mask2][..., 1, 0]) / s
        q[mask2, 2] = (R_flat[mask2][..., 0, 2] + R_flat[mask2][..., 2, 0]) / s

    # Case 3: R[1,1] is max diagonal
    mask3 = ~mask1 & ~mask2 & (R_flat[..., 1, 1] > R_flat[..., 2, 2])
    if np.any(mask3):
        s = 2.0 * np.sqrt(1.0 + R_flat[mask3][..., 1, 1] - R_flat[mask3][..., 0, 0] - R_flat[mask3][..., 2, 2])
        q[mask3, 3] = (R_flat[mask3][..., 0, 2] - R_flat[mask3][..., 2, 0]) / s
        q[mask3, 0] = (R_flat[mask3][..., 0, 1] + R_flat[mask3][..., 1, 0]) / s
        q[mask3, 1] = 0.25 * s
        q[mask3, 2] = (R_flat[mask3][..., 1, 2] + R_flat[mask3][..., 2, 1]) / s

    # Case 4: R[2,2] is max diagonal
    mask4 = ~mask1 & ~mask2 & ~mask3
    if np.any(mask4):
        s = 2.0 * np.sqrt(1.0 + R_flat[mask4][..., 2, 2] - R_flat[mask4][..., 0, 0] - R_flat[mask4][..., 1, 1])
        q[mask4, 3] = (R_flat[mask4][..., 1, 0] - R_flat[mask4][..., 0, 1]) / s
        q[mask4, 0] = (R_flat[mask4][..., 0, 2] + R_flat[mask4][..., 2, 0]) / s
        q[mask4, 1] = (R_flat[mask4][..., 1, 2] + R_flat[mask4][..., 2, 1]) / s
        q[mask4, 2] = 0.25 * s

    return q.reshape(batch_shape + (4,))
