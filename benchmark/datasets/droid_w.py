"""DROID-W dataset loader.

Dataset format:
  {raw_data_root}/
    {scene_name}/                         # e.g., downtown1, ..., downtown7
      images_anonymized/                  # JPEG frames, filename stem = Unix timestamp (seconds)
      images_anonymized_sky_masks/        # sparse sky masks (not used by this loader)
      traj_gt.txt OR traj_gt_fastlivo.txt # TUM-format trajectory (see below)
    videos_mp4/                           # raw mp4 reference (not used by this loader)

Trajectory format (per non-comment line):
    timestamp tx ty tz qx qy qz qw

Pose convention:
    Poses are CAMERA-TO-WORLD (C2W). This matches the TUM convention
    (tx,ty,tz = camera position in world frame; quaternion = camera
    orientation in world frame), and is confirmed by the LingBot-Map
    pipeline that produces these files (lingbot_streaming/base.py:
    "txt file: Each line contains a 4x4 C2W matrix"). DO NOT invert.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from scipy.spatial.transform import Rotation

from benchmark.dataset.base import BaseDataset


ASSOC_TOLERANCE = 0.05  # seconds; nearest-neighbor match between image and GT timestamp
PATCH_MULTIPLE = 14     # resize height is floored to a multiple of this (patch-based models)


class DroidWDataset(BaseDataset):
    """DROID-W dataset loader (RGB + C2W TUM trajectory)."""

    def __init__(
        self,
        raw_data_root: str,
        scenes: Optional[List[str]] = None,
        load_img_size: Optional[int] = 518,
        assoc_tolerance: float = ASSOC_TOLERANCE,
        logger=None,
    ):
        """Initialize DROID-W dataset loader.

        Args:
            raw_data_root:   Root directory containing downtown*/ subdirs.
            scenes:          Optional whitelist of scene names; if None, auto-discover.
            load_img_size:   Target image width for resize; None keeps native 1600x1200.
                             Height is scaled proportionally and floored to a multiple of 14.
            assoc_tolerance: Max |image_ts - pose_ts| (seconds) for nearest-neighbor match.
                             Exceeding this -> pose=None for that frame.
            logger:          Optional logger.
        """
        super().__init__(raw_data_root, logger=logger)
        self._scenes_whitelist = scenes
        self._load_img_size = load_img_size
        self._assoc_tolerance = assoc_tolerance
        self._scene_cache: Dict[str, Dict[str, Any]] = {}

    def get_scenes(self) -> List[str]:
        if self._scenes_whitelist is not None:
            return sorted(self._scenes_whitelist)
        found = []
        for d in sorted(self.raw_data_root.iterdir()):
            if not d.is_dir() or d.name.startswith('.'):
                continue
            if d.name == 'videos_mp4':
                continue
            if not (d / 'images_anonymized').is_dir():
                continue
            if _find_pose_file(d) is None:
                continue
            found.append(d.name)
        return found

    def get_frame_list(self, scene: str) -> List[int]:
        data = self._load_scene_data(scene)
        return list(range(len(data['rgb_files'])))

    def load_frame_data(self, scene: str, frame_id: int) -> Dict[str, Any]:
        data = self._load_scene_data(scene)
        rgb_ts, rgb_path = data['rgb_files'][frame_id]

        img = Image.open(rgb_path).convert('RGB')
        if self._load_img_size is not None:
            w, h = img.size
            target_w = self._load_img_size
            target_h = (int(round(target_w * h / w)) // PATCH_MULTIPLE) * PATCH_MULTIPLE
            if target_h <= 0:
                target_h = PATCH_MULTIPLE
            img = img.resize((target_w, target_h), Image.LANCZOS)
        rgb = np.array(img, dtype=np.uint8)

        pose: Optional[np.ndarray] = None
        gt_ts = data['gt_ts']
        if len(gt_ts) > 0:
            idx = int(np.argmin(np.abs(gt_ts - rgb_ts)))
            if abs(gt_ts[idx] - rgb_ts) <= self._assoc_tolerance:
                pose = data['gt_poses'][idx]

        return {'rgb': rgb, 'pose': pose}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_scene_data(self, scene: str) -> Dict[str, Any]:
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        scene_dir = self.raw_data_root / scene
        image_dir = scene_dir / 'images_anonymized'

        rgb_files: List[Tuple[float, Path]] = sorted(
            (float(p.stem), p)
            for p in image_dir.iterdir()
            if p.suffix.lower() in ('.jpg', '.jpeg')
        )

        pose_file = _find_pose_file(scene_dir)
        if pose_file is None:
            gt_ts = np.empty((0,), dtype=np.float64)
            gt_poses: List[np.ndarray] = []
        else:
            gt_ts, gt_poses = _parse_tum_trajectory(pose_file)

        self._scene_cache[scene] = {
            'rgb_files': rgb_files,
            'gt_ts': gt_ts,
            'gt_poses': gt_poses,
        }
        return self._scene_cache[scene]


# ------------------------------------------------------------------ #
#  Module-level helpers                                                #
# ------------------------------------------------------------------ #

def _find_pose_file(scene_dir: Path) -> Optional[Path]:
    """Return the first existing pose file among known candidates."""
    for name in ('traj_gt.txt', 'traj_gt_fastlivo.txt'):
        candidate = scene_dir / name
        if candidate.exists():
            return candidate
    return None


def _parse_tum_trajectory(path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Parse a TUM trajectory file into (timestamps, C2W 4x4 matrices).

    Each data line: timestamp tx ty tz qx qy qz qw   (scalar-last quaternion)
    The result is C2W: [R|t] with R from the quaternion and t = [tx,ty,tz].
    """
    timestamps: List[float] = []
    poses: List[np.ndarray] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = line.split()
            if len(vals) < 8:
                continue
            ts = float(vals[0])
            tx, ty, tz = float(vals[1]), float(vals[2]), float(vals[3])
            qx, qy, qz, qw = float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R
            c2w[:3, 3] = [tx, ty, tz]
            timestamps.append(ts)
            poses.append(c2w)
    return np.array(timestamps, dtype=np.float64), poses
