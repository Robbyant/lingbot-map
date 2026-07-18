
"""Registration utilities for point clouds."""

import numpy as np
import open3d as o3d


def apply_transform(
    points: np.ndarray,
    transformation: np.ndarray
) -> np.ndarray:
    """Apply a 4x4 transformation matrix to a set of 3D points.

    This function transforms 3D points using a homogeneous transformation matrix.
    Only the first 3 dimensions (xyz coordinates) are transformed, while any
    additional dimensions (e.g., RGB colors, normals) are preserved unchanged.

    Args:
        points: [..., N] array of 3D points where N >= 3. The first three columns
                are xyz coordinates, additional columns (if any) are preserved.
        transformation: 4x4 homogeneous transformation matrix.

    Returns:
        [..., N] array of transformed points with the same shape as input.
              Only xyz coordinates are transformed.

    Example:
        >>> points = np.array([[1, 2, 3, 255], [4, 5, 6, 128]])  # xyz + color
        >>> T = np.eye(4)
        >>> T[:3, 3] = [1, 1, 1]  # Translation
        >>> transformed = transform(points, T)
        >>> # xyz translated, color preserved
    """
    # Store original shape for later restoration
    original_shape = points.shape

    # Extract xyz coordinates (first 3 dimensions)
    xyz = points[..., :3]

    # Flatten to (N, 3) for batch processing
    xyz_flat = xyz.reshape(-1, 3)

    # Convert to homogeneous coordinates (N, 4)
    ones = np.ones((xyz_flat.shape[0], 1))
    xyz_homogeneous = np.hstack([xyz_flat, ones])

    # Apply transformation: (N, 4) @ (4, 4).T = (N, 4)
    xyz_transformed = xyz_homogeneous @ transformation.T

    # Convert back from homogeneous to Euclidean coordinates
    xyz_transformed = xyz_transformed[:, :3] / xyz_transformed[:, 3:4]

    # Reshape back to original shape
    xyz_transformed = xyz_transformed.reshape(original_shape[:-1] + (3,))

    # If points have additional dimensions beyond xyz, preserve them
    if original_shape[-1] > 3:
        additional_dims = points[..., 3:]
        result = np.concatenate([xyz_transformed, additional_dims], axis=-1)
    else:
        result = xyz_transformed

    return result

def umeyama_registration(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """Perform Umeyama registration to align source points to target points.

    This function implements the Umeyama algorithm for estimating a similarity
    transformation (rotation, translation, and uniform scaling) that best aligns
    corresponding point pairs from source to target.

    The algorithm computes optimal scale (c), rotation (R), and translation (t)
    such that: target ≈ c * R @ source + t

    Important: The input point clouds must have one-to-one correspondence, meaning
    source_points[i] should correspond to target_points[i] for all i.

    Args:
        source_points: [..., N] array of source point cloud where N >= 3.
                       The first three columns are xyz coordinates. Points must
                       correspond one-to-one with target_points.
        target_points: [..., N] array of target point cloud where N >= 3.
                       The first three columns are xyz coordinates. Must have
                       the same shape as source_points.

    Returns:
        4x4 homogeneous transformation matrix that transforms source to target.
        The matrix includes rotation, translation, and uniform scaling.

    Raises:
        ValueError: If source and target point clouds have different shapes.

    Example:
        >>> source = np.random.rand(100, 3)
        >>> T_gt = np.eye(4)
        >>> T_gt[:3, :3] = rotation_matrix  # Some rotation
        >>> T_gt[:3, 3] = [1, 2, 3]  # Translation
        >>> target = transform(source, T_gt)
        >>> T_estimated = umeyama_registration(source, target)
        >>> # T_estimated should be close to T_gt

    Reference:
        Umeyama, S. (1991). Least-squares estimation of transformation parameters
        between two point patterns. IEEE Transactions on Pattern Analysis and
        Machine Intelligence, 13(4), 376-380.
    """
    # Validate input shapes
    if source_points.shape != target_points.shape:
        raise ValueError(
            f"Source and target must have same shape, got {source_points.shape} "
            f"and {target_points.shape}"
        )

    # Extract xyz coordinates (first 3 dimensions) and flatten.
    # Cast to float64 to avoid numerical precision loss with large point counts
    # (float32 cross-covariance accumulation fails at ~10M+ points).
    source_xyz = source_points[..., :3].reshape(-1, 3).astype(np.float64)
    target_xyz = target_points[..., :3].reshape(-1, 3).astype(np.float64)

    # Transpose for computation: (3, N)
    X = source_xyz.T
    Y = target_xyz.T

    # Compute centroids
    mu_x = X.mean(axis=1, keepdims=True)  # (3, 1)
    mu_y = Y.mean(axis=1, keepdims=True)  # (3, 1)

    # Center the point clouds
    X_centered = X - mu_x
    Y_centered = Y - mu_y

    # Compute variance of source point cloud
    var_x = np.square(X_centered).sum() / X.shape[1]

    # Compute cross-covariance matrix
    cov_xy = (Y_centered @ X_centered.T) / X.shape[1]  # (3, 3)

    # Singular Value Decomposition
    U, D, VH = np.linalg.svd(cov_xy)

    # Construct the S matrix to handle reflections
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        # Ensure a proper rotation (det = +1) by flipping sign if needed
        S[2, 2] = -1

    # Compute optimal rotation
    R = U @ S @ VH  # (3, 3)

    # Compute optimal scale
    c = np.trace(np.diag(D) @ S) / var_x

    # Compute optimal translation
    t = mu_y - c * R @ mu_x  # (3, 1)

    # Construct 4x4 transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = c * R  # Scale and rotation
    transformation[:3, 3:4] = t     # Translation

    return transformation

def umeyama_registration_ransac(
    source_points: np.ndarray,
    target_points: np.ndarray,
    inlier_threshold: float = 0.1,
    ransac_n: int = 3,
    num_iterations: int = 100000,
    confidence: float = 0.999,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform RANSAC-based similarity registration robust to outliers.

    Uses Open3D's RANSAC engine (`registration_ransac_based_on_correspondence`) with
    `TransformationEstimationPointToPoint(with_scaling=True)`, which is equivalent to
    Umeyama similarity registration but runs in optimized C++.

    One-to-one correspondences (i ↔ i) are constructed automatically from the input arrays.
    After RANSAC, the inlier mask is recomputed from the returned transform for convenience.

    Args:
        source_points: (N, D) array, D >= 3.  xyz in first 3 columns.
                       Must correspond one-to-one with target_points.
        target_points: (N, D) array, same shape as source_points.
        inlier_threshold: Maximum Euclidean distance (after transform) for a
                          correspondence to be an inlier. Default: 0.1.
        ransac_n: Minimum number of correspondences sampled per iteration. Default: 3.
        num_iterations: Maximum RANSAC iterations. Default: 100000.
        confidence: RANSAC confidence level used for early stopping. Default: 0.999.
        verbose: If True, enables Open3D Debug verbosity to print RANSAC progress
                 (iteration count, best fitness, etc.). Default: False.

    Returns:
        transformation: 4x4 homogeneous similarity transform (source → target).
        inlier_mask: Boolean array of shape (N,) marking inliers of the final fit.

    Raises:
        ValueError: If source and target have different shapes or fewer than
                    `ransac_n` points.

    Example:
        >>> source = np.random.rand(200, 3)
        >>> T_gt = np.eye(4); T_gt[:3, 3] = [1, 2, 3]
        >>> target = (T_gt[:3, :3] @ source.T + T_gt[:3, 3:]).T
        >>> # Corrupt 30 % of correspondences
        >>> noise_idx = np.random.choice(200, 60, replace=False)
        >>> target[noise_idx] += np.random.randn(60, 3) * 5
        >>> T_est, mask = umeyama_registration_ransac(source, target)
    """
    if source_points.shape != target_points.shape:
        raise ValueError(
            f"Source and target must have same shape, got {source_points.shape} "
            f"and {target_points.shape}"
        )

    source_xyz = source_points[..., :3].reshape(-1, 3)
    target_xyz = target_points[..., :3].reshape(-1, 3)
    n = source_xyz.shape[0]

    if n < ransac_n:
        raise ValueError(f"Need at least {ransac_n} points, got {n}.")

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(source_xyz)

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(target_xyz)

    # One-to-one correspondences: (i, i) for i in 0..N-1
    corres = o3d.utility.Vector2iVector(np.stack([np.arange(n), np.arange(n)], axis=1))

    prev_level = o3d.utility.get_verbosity_level()
    if verbose:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=src_pcd,
        target=tgt_pcd,
        corres=corres,
        max_correspondence_distance=inlier_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True
        ),
        ransac_n=ransac_n,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            num_iterations, confidence
        ),
    )

    if verbose:
        o3d.utility.set_verbosity_level(prev_level)

    transformation = np.asarray(result.transformation)

    src_h = np.hstack([source_xyz, np.ones((n, 1))])
    transformed = (transformation @ src_h.T).T[:, :3]
    residuals = np.linalg.norm(transformed - target_xyz, axis=1)
    inlier_mask = residuals < inlier_threshold

    return transformation, inlier_mask


def icp_registration(
    source_points: np.ndarray,
    target_points: np.ndarray,
    icp_threshold: float = 0.1,
    max_iterations: int = 20,
    tolerance: float = 1e-6
) -> np.ndarray:
    """Perform ICP registration to align source points to target points.

    This function implements Iterative Closest Point (ICP) algorithm using Open3D.
    Unlike Umeyama registration, ICP does not require point correspondence and
    iteratively finds the best alignment by matching nearest neighbors.

    The algorithm:
    1. Finds closest point pairs between source and target
    2. Computes optimal transformation for current correspondences
    3. Applies transformation and repeats until convergence

    Args:
        source_points: [..., N] array of source point cloud where N >= 3.
                       The first three columns are xyz coordinates. Points do NOT
                       need to correspond with target_points.
        target_points: [..., N] array of target point cloud where N >= 3.
                       The first three columns are xyz coordinates.
        icp_threshold: Maximum correspondence distance for point pairs (in meters).
                       Point pairs with distance > threshold are rejected.
                       Default: 0.1 meters.
        max_iterations: Maximum number of ICP iterations. Default: 20.
        tolerance: Convergence criterion - stops when fitness change < tolerance.
                   Default: 1e-6.

    Returns:
        4x4 homogeneous transformation matrix that aligns source to target.

    Raises:
        ValueError: If point clouds have insufficient points (< 3).

    Example:
        >>> source = np.random.rand(1000, 3)
        >>> target = np.random.rand(1000, 3)
        >>> T = icp_registration(source, target, icp_threshold=0.05)
        >>> aligned_source = transform(source, T)

    Note:
        This function uses Open3D's RegistrationICP with Point-to-Point metric.
        For better results with planar surfaces, consider Point-to-Plane ICP.

    Reference:
        Besl, P. J., & McKay, N. D. (1992). A method for registration of 3-D shapes.
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 14(2), 239-256.
    """
    # Extract xyz coordinates (first 3 dimensions) and flatten
    source_xyz = source_points[..., :3].reshape(-1, 3)
    target_xyz = target_points[..., :3].reshape(-1, 3)

    # Validate point cloud sizes
    if source_xyz.shape[0] < 3 or target_xyz.shape[0] < 3:
        raise ValueError(
            f"Point clouds must have at least 3 points, got source: {source_xyz.shape[0]}, "
            f"target: {target_xyz.shape[0]}"
        )

    # Convert numpy arrays to Open3D point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_xyz)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_xyz)

    # Set up ICP convergence criteria
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=tolerance,
        relative_rmse=tolerance,
        max_iteration=max_iterations
    )

    # Perform Point-to-Point ICP registration
    # Initial transformation is identity (no initial guess)
    init_transformation = np.eye(4)

    reg_result = o3d.pipelines.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=icp_threshold,
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=criteria
    )

    # Extract and return the transformation matrix
    transformation = np.asarray(reg_result.transformation)

    return transformation


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample a point cloud using a voxel grid filter.

    Args:
        points: (N, 3) or (N, 6) xyz or xyzrgb point cloud.
        voxel_size: Voxel size in meters.

    Returns:
        (M, D) downsampled point cloud preserving color channels if present.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if points.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    pcd_down = pcd.voxel_down_sample(voxel_size)
    xyz = np.asarray(pcd_down.points)
    if points.shape[1] >= 6:
        rgb = np.asarray(pcd_down.colors)
        return np.hstack([xyz, rgb]).astype(np.float32)
    return xyz.astype(np.float32)