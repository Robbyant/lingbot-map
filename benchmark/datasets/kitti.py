"""KITTI Odometry dataset loader.

Dataset format (the standard KITTI Odometry release):
  {raw_data_root}/
    poses/
      {seq:02d}.txt          # one line per frame: 12 floats = 3x4 row-major C2W
                             # (left grayscale camera, cam_0 frame). Provided
                             # only for sequences 00-10.
    sequences/
      {seq:02d}/
        calib.txt            # P0/P1/P2/P3 lines (3x4 each), projection matrices
        times.txt            # one timestamp per frame, seconds (starts at 0)
        image_2/000000.png   # left color camera (cam_2). 1226x370 etc.
        image_3/             # right color camera (cam_3). Not used here.

Conventions:
  - Poses are C2W (KITTI doc: "aligns camera coordinate system with world")
  - cam_0 (left grayscale) is the reference frame for poses; cam_2 (left color)
    is a rectified rigid offset of cam_0. We feed image_2 RGB and use cam_0
    poses unchanged - the constant baseline offset is absorbed by Sim(3)
    trajectory alignment in evaluation.
  - Intrinsics: fx, fy, cx, cy come from P2[:3, :3] (rectified, R = I).
  - Sequences 11-21 have no ground-truth poses; default whitelist = 00..10.

Reference:
  https://www.cvlibs.net/datasets/kitti/eval_odometry.php
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple

from benchmark.dataset.base import BaseDataset


# Sequences that ship with ground-truth poses
GT_SEQUENCES = [f"{i:02d}" for i in range(11)]

# Patch alignment for downstream ViT-style methods (lingbot-map, vggt, ...).
_PATCH_ALIGN = 14


class KittiDataset(BaseDataset):
    """KITTI Odometry dataset loader (color, image_2)."""

    def __init__(self, raw_data_root: str, sequences: Optional[List[str]] = None,
                 target_size: Optional[List[int]] = None,
                 logger=None):
        """Initialize KITTI Odometry loader.

        Args:
            raw_data_root: Dataset root containing `poses/` and `sequences/`.
            sequences: Optional whitelist of sequence IDs (e.g. ['00', '05']).
                       If None, all sequences with GT poses (00..10) are used.
            target_size: Optional [W, H] target. When set, frames are
                cover-fit resized (max scale, preserving aspect ratio) and
                center-cropped to exactly W×H, with intrinsics updated. Both
                dims must be multiples of 14. Resize uses cv2.INTER_AREA.
            logger: Optional logger.
        """
        super().__init__(raw_data_root, logger=logger)
        self._sequences_whitelist = sequences
        self._scene_cache: Dict[str, Dict] = {}

        if target_size is None:
            self.target_size: Optional[Tuple[int, int]] = None
        else:
            tw, th = int(target_size[0]), int(target_size[1])
            if tw % _PATCH_ALIGN or th % _PATCH_ALIGN:
                raise ValueError(
                    f"target_size {target_size} must be multiples of {_PATCH_ALIGN}"
                )
            self.target_size = (tw, th)

    def get_scenes(self) -> List[str]:
        """Return list of sequence IDs as scene identifiers."""
        if self._sequences_whitelist is not None:
            return sorted(str(s).zfill(2) for s in self._sequences_whitelist)
        sequences_dir = self.raw_data_root / 'sequences'
        poses_dir = self.raw_data_root / 'poses'
        return sorted(
            d.name for d in sequences_dir.iterdir()
            if d.is_dir() and (poses_dir / f"{d.name}.txt").exists()
        )

    def get_frame_list(self, scene: str) -> List[int]:
        """Return sequential frame indices [0, 1, ..., N-1] for the sequence."""
        data = self._load_scene_data(scene)
        return list(range(len(data['timestamps'])))

    def load_frame_data(self, scene: str, frame_id: int) -> Dict[str, Any]:
        """Load RGB + GT pose + intrinsics for a single frame.

        Returns:
            Dict with:
              - 'rgb':        HxWx3 uint8 RGB image (from image_2/)
              - 'pose':       4x4 float32 C2W matrix (cam_0 frame), or None
                              if the sequence has no GT
              - 'intrinsics': [fx, fy, cx, cy] float32 (from P2)

        Note:
            The framework re-indexes frames as 0..K-1 in BSS and does not
            preserve the original KITTI per-frame timestamps from times.txt.
            We therefore intentionally omit the 'timestamp' key — including
            it would trip the saver's custom-key check.
        """
        data = self._load_scene_data(scene)

        seq_dir = self.raw_data_root / 'sequences' / scene
        rgb_path = seq_dir / 'image_2' / f"{frame_id:06d}.png"
        rgb = np.array(Image.open(rgb_path).convert('RGB'), dtype=np.uint8)

        pose = data['poses'][frame_id] if data['poses'] is not None else None
        intrinsics = data['intrinsics'].copy()

        if self.target_size is not None:
            rgb, intrinsics = self._cover_fit_center_crop(rgb, intrinsics)

        return {
            'rgb': rgb,
            'pose': pose,
            'intrinsics': intrinsics,
        }

    def _cover_fit_center_crop(
        self, rgb: np.ndarray, intrinsics: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cover-fit resize + center-crop to ``self.target_size``.

        Picks the smallest scale s such that the resized image still covers
        the target box, then center-crops the excess. Intrinsics are scaled
        by s and shifted by the crop offset.
        """
        H, W = rgb.shape[:2]
        Tw, Th = self.target_size
        scale = max(Tw / W, Th / H)
        rW, rH = int(round(W * scale)), int(round(H * scale))
        resized = cv2.resize(rgb, (rW, rH), interpolation=cv2.INTER_AREA)

        x0 = (rW - Tw) // 2
        y0 = (rH - Th) // 2
        cropped = resized[y0:y0 + Th, x0:x0 + Tw]

        fx, fy, cx, cy = (float(v) for v in intrinsics)
        fx *= scale
        fy *= scale
        cx = cx * scale - x0
        cy = cy * scale - y0
        return cropped, np.array([fx, fy, cx, cy], dtype=np.float32)

    def load_global_data(self, scene: str) -> Dict[str, Any]:
        """KITTI Odometry does not provide a GT global point cloud."""
        return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_scene_data(self, scene: str) -> Dict:
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        seq_dir = self.raw_data_root / 'sequences' / scene
        if not seq_dir.is_dir():
            raise FileNotFoundError(f"KITTI sequence not found: {seq_dir}")

        timestamps = np.loadtxt(seq_dir / 'times.txt', dtype=np.float64)
        if timestamps.ndim == 0:
            timestamps = timestamps.reshape(1)

        intrinsics = _parse_intrinsics_p2(seq_dir / 'calib.txt')

        poses_path = self.raw_data_root / 'poses' / f"{scene}.txt"
        if poses_path.exists():
            poses = _parse_poses(poses_path)
            if len(poses) != len(timestamps):
                if self.logger is not None:
                    self.logger.warning(
                        f"KITTI seq {scene}: pose count {len(poses)} != "
                        f"frame count {len(timestamps)}; truncating to min."
                    )
                n = min(len(poses), len(timestamps))
                poses = poses[:n]
                timestamps = timestamps[:n]
        else:
            poses = None
            if self.logger is not None:
                self.logger.warning(
                    f"KITTI seq {scene}: no GT poses at {poses_path}; "
                    f"trajectory will be all-NaN."
                )

        self._scene_cache[scene] = {
            'timestamps': timestamps,
            'poses': poses,
            'intrinsics': intrinsics,
        }
        return self._scene_cache[scene]


def _parse_intrinsics_p2(calib_path: Path) -> np.ndarray:
    """Extract [fx, fy, cx, cy] from the P2 row of KITTI calib.txt.

    P2 is a 3x4 projection matrix for the left color camera (cam_2),
    in the rectified frame. For rectified cameras P2 = K [I | t / fx],
    so K = P2[:, :3].
    """
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('P2'):
                continue
            vals = [float(v) for v in line.split()[1:]]
            if len(vals) != 12:
                raise ValueError(f"Malformed P2 line in {calib_path}: {line}")
            P2 = np.array(vals, dtype=np.float64).reshape(3, 4)
            fx, fy = float(P2[0, 0]), float(P2[1, 1])
            cx, cy = float(P2[0, 2]), float(P2[1, 2])
            return np.array([fx, fy, cx, cy], dtype=np.float32)
    raise ValueError(f"P2 line not found in {calib_path}")


def _parse_poses(poses_path: Path) -> List[np.ndarray]:
    """Parse KITTI poses file: one line per frame, 12 floats = 3x4 row-major C2W."""
    arr = np.loadtxt(poses_path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 12:
        raise ValueError(
            f"KITTI poses file {poses_path} has {arr.shape[1]} cols, expected 12"
        )
    poses: List[np.ndarray] = []
    for row in arr:
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :4] = row.reshape(3, 4).astype(np.float32)
        poses.append(c2w)
    return poses
