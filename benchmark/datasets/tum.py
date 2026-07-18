"""TUM RGB-D dataset loader.

Dataset format:
  Source format:
    {raw_data_root}/
      {scene_name}/                 # e.g., rgbd_dataset_freiburg1_desk
        rgb/                        # RGB images (640x480 PNG); filenames are timestamps
        rgb.txt                     # timestamp filename
        depth/                      # Depth maps (16-bit PNG, 1/5000 m); not used here
        groundtruth.txt             # timestamp tx ty tz qx qy qz qw  (C2W, TUM convention)

Notes:
  - RGB and ground-truth poses are sampled asynchronously, so each RGB frame is
    associated with the temporally nearest GT pose within ASSOC_TOLERANCE seconds.
    Frames without a GT pose inside the tolerance get pose=None (the framework
    writes a NaN row into traj.txt).
  - Intrinsics are the official TUM factory calibration, selected per Freiburg
    camera (fr1/fr2/fr3) from the scene name. Images are resized to load_img_size
    width and the intrinsics are scaled to match.

Reference:
  https://cvg.cit.tum.de/data/datasets/rgbd-dataset
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple

from scipy.spatial.transform import Rotation

from benchmark.dataset.base import BaseDataset


# Maximum |rgb_ts - gt_ts| (seconds) for a frame to be associated with a GT pose.
# Matches the TUM RGB-D associate.py default.
ASSOC_TOLERANCE = 0.02

# Official TUM RGB-D factory intrinsics [fx, fy, cx, cy] for the 640x480 color
# camera, keyed by Freiburg camera id. See:
# https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
TUM_INTRINSICS = {
    'freiburg1': (517.306408, 516.469215, 318.643040, 255.313989),
    'freiburg2': (520.908620, 521.007327, 325.141442, 249.701764),
    'freiburg3': (535.4, 539.2, 320.1, 247.6),
}
DEFAULT_CAMERA = 'freiburg1'

ORIG_WIDTH = 640
ORIG_HEIGHT = 480


def _parse_groundtruth(path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Parse a TUM groundtruth.txt into timestamps and C2W matrices.

    Format per line: timestamp tx ty tz qx qy qz qw  (comment lines start with '#')

    Args:
        path: Path to groundtruth.txt

    Returns:
        (timestamps, poses) where timestamps is a float64 array of length N and
        poses is a list of N 4x4 float64 C2W matrices.
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
            timestamps.append(float(vals[0]))
            tx, ty, tz = (float(v) for v in vals[1:4])
            qx, qy, qz, qw = (float(v) for v in vals[4:8])
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            c2w[:3, 3] = (tx, ty, tz)
            poses.append(c2w)
    return np.asarray(timestamps, dtype=np.float64), poses


def _nearest_index(array: np.ndarray, query: float) -> Optional[int]:
    """Return the index of the nearest timestamp, or None if the array is empty."""
    if array.size == 0:
        return None
    return int(np.argmin(np.abs(array - query)))


def _camera_of(scene: str) -> str:
    """Infer the Freiburg camera id (freiburg1/2/3) from a scene directory name."""
    for cam in TUM_INTRINSICS:
        if cam in scene:
            return cam
    return DEFAULT_CAMERA


class TumDataset(BaseDataset):
    """TUM RGB-D dataset loader (trajectory evaluation)."""

    def __init__(self, raw_data_root: str, load_img_size: int = 518,
                 scenes: Optional[List[str]] = None, logger=None):
        """Initialize the TUM RGB-D dataset loader.

        Args:
            raw_data_root: Dataset root directory (e.g., /data0/clz/TUM-RGBD)
            load_img_size: Resize images so the width equals this value
            scenes: Optional whitelist of scene directory names; if None, all
                    rgbd_dataset_freiburg* subdirectories are discovered
            logger: Optional logger instance
        """
        super().__init__(raw_data_root, logger=logger)
        self.load_img_size = load_img_size
        self._scenes_whitelist = scenes
        self._scene_cache: Dict[str, Dict[str, Any]] = {}

    def get_scenes(self) -> List[str]:
        """Discover all scene directories or return the configured whitelist."""
        if self._scenes_whitelist is not None:
            return list(self._scenes_whitelist)
        return sorted(
            p.name for p in self.raw_data_root.iterdir()
            if p.is_dir() and p.name.startswith('rgbd_dataset_freiburg')
        )

    def get_frame_list(self, scene: str) -> List[int]:
        """Return sequential frame indices [0, 1, ..., N-1]."""
        data = self._load_scene_data(scene)
        return list(range(len(data['rgb_files'])))

    def load_frame_data(self, scene: str, frame_id: int) -> Dict[str, Any]:
        """Load the RGB image and associated GT pose for a single frame.

        Args:
            scene:    Scene directory name
            frame_id: Sequential frame index

        Returns:
            Dictionary with:
              - 'rgb':        HxWx3 uint8 RGB image (resized to load_img_size width)
              - 'intrinsics': [fx, fy, cx, cy] float32 (scaled to the resized image)
              - 'pose':       4x4 float32 C2W matrix, or None if no GT within tolerance
        """
        data = self._load_scene_data(scene)
        rgb_ts, rgb_path = data['rgb_files'][frame_id]

        rgb = np.array(Image.open(rgb_path).convert('RGB'), dtype=np.uint8)
        intrinsics = data['intrinsics'].copy()

        # Resize so width == load_img_size, scaling intrinsics to match.
        orig_h, orig_w = rgb.shape[:2]
        if self.load_img_size > 0 and orig_w != self.load_img_size:
            target_width = self.load_img_size
            aspect_ratio = float(orig_h) / orig_w
            target_height = int(target_width * aspect_ratio)
            # Round to a multiple of 14 for patch-based models.
            target_height = (target_height // 14) * 14

            rgb = cv2.resize(rgb, (target_width, target_height),
                             interpolation=cv2.INTER_LINEAR)
            intrinsics[0] *= target_width / orig_w   # fx
            intrinsics[2] *= target_width / orig_w   # cx
            intrinsics[1] *= target_height / orig_h  # fy
            intrinsics[3] *= target_height / orig_h  # cy

        # Associate the temporally nearest GT pose.
        gt_ts = data['gt_ts']
        gt_idx = _nearest_index(gt_ts, rgb_ts)
        pose = None
        if gt_idx is not None and abs(gt_ts[gt_idx] - rgb_ts) <= ASSOC_TOLERANCE:
            pose = data['gt_poses'][gt_idx].astype(np.float32)
        elif self.logger is not None:
            self.logger.warning(
                f"No GT pose within {ASSOC_TOLERANCE}s for scene '{scene}' "
                f"frame {frame_id} (rgb_ts={rgb_ts:.4f})"
            )

        return {
            'rgb': rgb,
            'intrinsics': intrinsics,
            'pose': pose,
        }

    def _load_scene_data(self, scene: str) -> Dict[str, Any]:
        """Lazy-load and cache per-scene RGB file list and GT trajectory."""
        if scene not in self._scene_cache:
            scene_dir = self.raw_data_root / scene

            # RGB frames sorted by their timestamp filename (e.g. rgb/1305031452.791720.png)
            rgb_files = sorted(
                ((float(p.stem), p) for p in (scene_dir / 'rgb').glob('*.png')),
                key=lambda x: x[0],
            )

            gt_ts, gt_poses = _parse_groundtruth(scene_dir / 'groundtruth.txt')

            fx, fy, cx, cy = TUM_INTRINSICS[_camera_of(scene)]
            intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)

            self._scene_cache[scene] = {
                'rgb_files': rgb_files,
                'gt_ts': gt_ts,
                'gt_poses': gt_poses,
                'intrinsics': intrinsics,
            }
        return self._scene_cache[scene]
