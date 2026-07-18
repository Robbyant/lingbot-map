"""Benchmark trajectory format I/O utilities."""

import numpy as np
from pathlib import Path
from typing import List, Optional, Union


def read_trajectory(traj_file: Path, num_frames: Optional[int] = None) -> np.ndarray:
    """Read trajectory file in BSS v2 format.

    BSS Trajectory Format v2 (one line per frame):
        frame_idx r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz

    Where the 12 values represent the 3x4 camera-to-world (C2W) matrix:
        [r00 r01 r02 tx]
        [r10 r11 r12 ty]
        [r20 r21 r22 tz]
        [ 0   0   0   1]

    Missing poses are represented by absent lines (sparse format).
    Legacy files may also contain explicit NaN rows; these are handled correctly.

    Args:
        traj_file:  Trajectory file path
        num_frames: If provided, the returned array is guaranteed to have exactly
                    this many rows (padded with NaN if the last keyframe index is
                    smaller than num_frames - 1).  If None, the array size is
                    max(frame_idx) + 1.

    Returns:
        np.ndarray of shape (N, 4, 4), NaN 4x4 matrices where pose is missing.

    Note:
        Lines starting with '#' are treated as comments and ignored.
    """
    entries = []  # list of (frame_idx, pose_or_none)

    with open(traj_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) != 13:
                raise ValueError(f"Invalid trajectory line (expected 13 values): {line}")

            frame_idx = int(parts[0])

            # Check if pose is NaN (legacy explicit-NaN rows)
            val = float(parts[1])
            if np.isnan(val):
                entries.append((frame_idx, None))
            else:
                r00, r01, r02, tx = map(float, parts[1:5])
                r10, r11, r12, ty = map(float, parts[5:9])
                r20, r21, r22, tz = map(float, parts[9:13])

                c2w = np.eye(4, dtype=np.float64)
                c2w[0, :] = [r00, r01, r02, tx]
                c2w[1, :] = [r10, r11, r12, ty]
                c2w[2, :] = [r20, r21, r22, tz]

                entries.append((frame_idx, c2w))

    if not entries:
        N = num_frames if num_frames is not None else 0
        return np.full((N, 4, 4), np.nan, dtype=np.float64)

    max_idx = max(idx for idx, _ in entries) + 1
    N = max(num_frames if num_frames is not None else 0, max_idx)
    result = np.full((N, 4, 4), np.nan, dtype=np.float64)

    for frame_idx, pose in entries:
        if pose is not None:
            result[frame_idx] = pose

    return result


def write_trajectory(
    traj_file: Path,
    poses: Union[np.ndarray, List[Optional[np.ndarray]]],
) -> None:
    """Write trajectory in BSS v2 format.

    Args:
        traj_file: Output file path
        poses:     Either np.ndarray of shape (N, 4, 4) or List[Optional[np.ndarray]].
                   NaN rows or None entries represent missing/unknown poses.

    Note:
        Writes 13 values per line: frame_idx + 12 matrix elements (3x4 row-major).
        Frame indices are always sequential (0, 1, 2, ...) regardless of the original
        GT frame positions. For sparse SLAM outputs, the mapping from sequential index
        to original GT frame index is stored in .complete.json as 'frame_index_map'.
        Precision: .10f for matrix values.
        Missing poses (None or NaN) are silently omitted — readers infer NaN for
        absent frame indices.
    """
    if isinstance(poses, np.ndarray):
        pose_list = [poses[i] for i in range(len(poses))]
    else:
        pose_list = list(poses)

    with open(traj_file, 'w') as f:
        f.write("# BSS Trajectory Format v2\n")
        f.write("# frame_idx r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz\n")
        f.write("# Represents the 3x4 camera-to-world (C2W) transformation matrix:\n")
        f.write("#   [r00 r01 r02 tx]\n")
        f.write("#   [r10 r11 r12 ty]\n")
        f.write("#   [r20 r21 r22 tz]\n")
        f.write("#   [ 0   0   0   1]\n")

        for i, pose in enumerate(pose_list):
            if pose is None or np.any(np.isnan(np.asarray(pose))):
                continue  # absent frames are inferred as NaN by the reader
            T = np.asarray(pose)
            r00, r01, r02, tx = T[0, :]
            r10, r11, r12, ty = T[1, :]
            r20, r21, r22, tz = T[2, :]
            f.write(
                f"{i} "
                f"{r00:.10f} {r01:.10f} {r02:.10f} {tx:.10f} "
                f"{r10:.10f} {r11:.10f} {r12:.10f} {ty:.10f} "
                f"{r20:.10f} {r21:.10f} {r22:.10f} {tz:.10f}\n"
            )
