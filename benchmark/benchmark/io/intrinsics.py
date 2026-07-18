"""Camera intrinsics I/O utilities."""

import numpy as np
from pathlib import Path
from typing import List, Optional, Union


def read_intrinsics(intr_file: Path) -> np.ndarray:
    """Read camera intrinsics file in BSS v2 format.

    Format (one line per frame):
        frame_idx fx fy cx cy width height

    Args:
        intr_file: Intrinsics file path

    Returns:
        np.ndarray of shape (N, 4) with [fx, fy, cx, cy] per row.
        NaN rows where intrinsics are missing.
        N = max frame_idx + 1.

    Note:
        Lines starting with '#' are treated as comments and ignored.
    """
    entries = []  # list of (frame_idx, intr_or_none)

    with open(intr_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            frame_idx = int(parts[0])

            # Check if intrinsics are NaN
            val = float(parts[1])
            if np.isnan(val):
                entries.append((frame_idx, None))
            else:
                fx, fy, cx, cy = map(float, parts[1:5])
                entries.append((frame_idx, np.array([fx, fy, cx, cy], dtype=np.float64)))

    if not entries:
        return np.full((0, 4), np.nan, dtype=np.float64)

    N = max(idx for idx, _ in entries) + 1
    result = np.full((N, 4), np.nan, dtype=np.float64)

    for frame_idx, intr in entries:
        if intr is not None:
            result[frame_idx] = intr

    return result


def write_intrinsics(
    intr_file: Path,
    intrinsics: Union[np.ndarray, List[Optional[np.ndarray]]],
    width: int,
    height: int,
) -> None:
    """Write camera intrinsics file in BSS v2 format.

    Args:
        intr_file:  Output file path
        intrinsics: np.ndarray of shape (N, 4) or List[Optional[array-like]].
                    Each entry is [fx, fy, cx, cy]. None or NaN → missing row.
        width:      Image width (pixels)
        height:     Image height (pixels)

    Note:
        Format: frame_idx fx fy cx cy width height
    """
    if isinstance(intrinsics, np.ndarray):
        intr_list = [intrinsics[i] for i in range(len(intrinsics))]
    else:
        intr_list = list(intrinsics)

    with open(intr_file, 'w') as f:
        f.write("# BSS Intrinsics Format v2\n")
        f.write("# frame_idx fx fy cx cy width height\n")
        f.write("# fx, fy: focal lengths (pixels)\n")
        f.write("# cx, cy: principal point (pixels)\n")
        f.write("# width, height: image dimensions (pixels)\n")

        for i, intr in enumerate(intr_list):
            if intr is None or np.any(np.isnan(np.asarray(intr))):
                f.write(f"{i} nan nan nan nan {width} {height}\n")
            else:
                arr = np.asarray(intr)
                fx, fy, cx, cy = arr[0], arr[1], arr[2], arr[3]
                f.write(f"{i} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} {width} {height}\n")
