"""Tanks and Temples dataset loader.

Dataset format:
  Source format:
    {raw_data_root}/
      {scene_name}/
        {NNNNNN}.jpg              # RGB images (1-indexed, e.g., 000001.jpg)
        {scene}_COLMAP_SfM.log   # COLMAP SfM camera poses (5 lines per entry)
        {scene}.ply               # Ground truth point cloud
        {scene}.json              # Scene crop volume (SelectionPolygonVolume)
        {scene}_trans.txt         # Alignment transform (4x4 matrix)

Reference:
  https://www.tanksandtemples.org/
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List

from benchmark.dataset.base import BaseDataset

SCENES_TAU_DICT = {
    'Barn': 0.01,
    'Caterpillar': 0.005,
    'Church': 0.025,
    'Courthouse': 0.025,
    'Ignatius': 0.003,
    'Meetingroom': 0.01,
    'Truck': 0.005,
}

TAT_SCENES = ['Barn', 'Caterpillar', 'Church', 'Ignatius', 'Meetingroom', 'Truck']

def _read_tat_trajectory(log_file: Path) -> List[np.ndarray]:
    """Read a COLMAP SfM .log file and return list of 4x4 C2W matrices.

    Log format: 5 lines per entry
      Line 1: metadata (idx idx 0)
      Lines 2-5: 4 rows of the 4x4 camera-to-world matrix

    Args:
        log_file: Path to the .log file

    Returns:
        List of 4x4 C2W transformation matrices (float64), one per frame
    """
    poses = []
    with open(log_file, 'r') as f:
        while True:
            meta_line = f.readline()
            if not meta_line:
                break
            meta_line = meta_line.strip()
            if not meta_line:
                break
            mat = np.zeros((4, 4), dtype=np.float64)
            for i in range(4):
                row_str = f.readline()
                mat[i, :] = [float(x) for x in row_str.split()]
            poses.append(mat)
    return poses


class TntDataset(BaseDataset):
    """Tanks and Temples dataset loader."""

    def __init__(self, raw_data_root: str, load_img_size: int = 518, logger=None):
        """Initialize Tanks and Temples dataset loader.

        Args:
            raw_data_root: Dataset root directory (containing scene subdirectories)
            load_img_size: Resize images so the width equals this value
            logger: Optional logger instance
        """
        super().__init__(raw_data_root, logger=logger)
        self.load_img_size = load_img_size
        self._pose_cache: Dict[str, List[np.ndarray]] = {}

    def get_scenes(self) -> List[str]:
        """Return TAT scenes whose directories exist under raw_data_root."""
        return [s for s in TAT_SCENES if (self.raw_data_root / s).is_dir()]

    def get_frame_list(self, scene: str) -> List[int]:
        """Get sorted list of frame IDs (parsed from JPG filenames).

        TAT images are named 000001.jpg, 000002.jpg, etc. (1-indexed).

        Args:
            scene: Scene name (e.g., 'Barn')

        Returns:
            Sorted list of integer frame IDs
        """
        scene_dir = self.raw_data_root / scene
        frame_ids = sorted(int(p.stem) for p in scene_dir.glob('*.jpg'))
        return frame_ids

    def load_frame_data(self, scene: str, frame_id: int) -> Dict[str, Any]:
        """Load a single frame: RGB image and COLMAP C2W pose.

        Args:
            scene: Scene name (e.g., 'Barn')
            frame_id: 1-indexed frame ID (matches filename 000001.jpg → frame_id=1)

        Returns:
            Dict with keys:
              - 'rgb': HxWx3 uint8 RGB array (resized to load_img_size width)
              - 'pose': 4x4 float64 C2W matrix from COLMAP SfM log
        """
        scene_dir = self.raw_data_root / scene
        img_file = scene_dir / f"{frame_id:06d}.jpg"

        # Load and resize RGB
        img = Image.open(img_file).convert('RGB')
        w, h = img.size

        target_width = self.load_img_size
        aspect_ratio = float(h) / w
        target_height = int(target_width * aspect_ratio)
        # Round to nearest multiple of 14 for patch-based models
        target_height = (target_height // 14) * 14

        img = img.resize((target_width, target_height), Image.LANCZOS)
        rgb = np.array(img, dtype=np.uint8)

        # COLMAP log is 0-indexed; images are 1-indexed
        poses = self._get_colmap_poses(scene)
        pose = poses[frame_id - 1]

        # Estimate intrinsics (assuming standard camera with principal point at center)
        # Use a reasonable default focal length (approximately 1.2 * width)
        fx = fy = 1.2 * target_width
        cx, cy = target_width / 2, target_height / 2

        return {
            'rgb': rgb,
            'pose': pose,
            'intrinsics': np.array([fx, fy, cx, cy], dtype=np.float32)
        }

    def _get_colmap_poses(self, scene: str) -> List[np.ndarray]:
        """Lazy-load and cache all COLMAP poses for a scene."""
        if scene not in self._pose_cache:
            log_file = self.raw_data_root / scene / f"{scene}_COLMAP_SfM.log"
            self._pose_cache[scene] = _read_tat_trajectory(log_file)
        return self._pose_cache[scene]
