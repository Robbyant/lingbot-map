"""Oxford Spires dataset loader.

Dataset format:
  {raw_data_root}/
    {scene_name}/                        # e.g., keble-college-02
      images/
        000000.png, 000001.png, ...      # RGB images (1440x1080, uint8)
      depth/
        000000.npy, 000001.npy, ...      # GT depth maps (float32, meters, 0=invalid) (NOT used)
      depth_vis/                         # Visualization (ignored)  (NOT used)
      poses_c2w.txt                      # N lines, each: 16 floats (4x4 C2W, row-major)
      intrinsics.txt                     # "fx fy cx cy width height"
      ground_truth.ply                   # Per-scene LiDAR reference cloud (xyzrgb)  (NOT used)

Reference:
  https://ori-drs.github.io/oxford-spires-dataset/
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional

from benchmark.dataset.base import BaseDataset


# Voxel size for downsampling reference LiDAR clouds during prepare.
# Per-scene ground_truth.ply files are 287 MB–1.1 GB; 5 cm voxels reduce them
# to ~50–200 MB while preserving sufficient spatial detail for outdoor evaluation.
VOXEL_SIZE_PREPARE = 0.05  # meters

# Distance thresholds suitable for large-scale outdoor scenes (meters)
# EVAL_THRESHOLDS = [0.1, 0.25, 0.5, 1.0]

# The gt points is not aligned with the gt pose, 
# and we exclude these scenes from evaluation.
EXCLUDED_SCENES = [
    'blenheim-palace-01',
    "blenheim-palace-02",
    "blenheim-palace-05",
    "christ-church-01",
]


class OxfordSpiresDataset(BaseDataset):
    """Oxford Spires large-scale outdoor dataset."""

    def __init__(self, raw_data_root: str, load_img_size: int = 518, logger=None):
        super().__init__(raw_data_root, logger=logger)

        self.load_img_size = load_img_size

        # Per-scene caches to avoid repeated file I/O
        self._poses: Dict[str, np.ndarray] = {}       # scene -> (N, 4, 4) float32
        self._intrinsics: Dict[str, np.ndarray] = {}  # scene -> [fx, fy, cx, cy]

    def get_scenes(self) -> List[str]:
        """Return all scene directory names that have images/ and poses_c2w.txt."""
        scenes = []
        for d in sorted(self.raw_data_root.iterdir()):
            if d.is_dir() and (d / 'images').is_dir() and (d / 'poses_c2w.txt').exists():
                if d.name not in EXCLUDED_SCENES:
                    scenes.append(d.name)
        return scenes

    def get_frame_list(self, scene: str) -> List[int]:
        """Return 0-based frame IDs (one per pose line)."""
        return list(range(len(self._load_poses(scene))))

    def load_frame_data(self, scene: str, frame_id: int) -> Dict[str, Any]:
        """Load RGB image, C2W pose, and intrinsics for one frame."""
        scene_dir = self.raw_data_root / scene

        rgb_file = scene_dir / 'images' / f'{frame_id:06d}.png'

        # Load and resize RGB
        img = np.array(Image.open(rgb_file).convert('RGB'), dtype=np.uint8)
        w, h = img.shape[1], img.shape[0]
        
        target_width = self.load_img_size
        aspect_ratio = float(h) / w
        target_height = int(target_width * aspect_ratio)
        # Round to nearest multiple of 14 for patch-based models
        target_height = (target_height // 14) * 14

        img = Image.fromarray(img)
        img = img.resize((target_width, target_height), Image.LANCZOS)
        img = np.array(img, dtype=np.uint8)

        # resize intrinsics accordingly (fx, fy scaled by width ratio; cx, cy scaled by width and height ratios)
        intrinsics = self._load_intrinsics(scene)
        fx, fy, cx, cy = intrinsics
        scale_x = target_width / w
        scale_y = target_height / h
        intrinsics_resized = np.array([
            fx * scale_x,
            fy * scale_y,
            cx * scale_x,
            cy * scale_y,
        ], dtype=np.float32)

        return {
            'rgb': img,
            'pose': self._load_poses(scene)[frame_id],
            'intrinsics': intrinsics_resized,
        }

    # ------------------------------------------------------------------ helpers

    def _load_poses(self, scene: str) -> np.ndarray:
        """Load and cache all C2W poses from poses_c2w.txt.

        Each line contains 16 space-separated floats (4x4 matrix, row-major).
        Returns shape (N, 4, 4), dtype float32.
        """
        if scene not in self._poses:
            poses_file = self.raw_data_root / scene / 'poses_c2w.txt'
            rows = []
            with open(poses_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        vals = [float(v) for v in line.split()]
                        rows.append(np.array(vals, dtype=np.float32).reshape(4, 4))
            self._poses[scene] = np.stack(rows)
        return self._poses[scene]

    def _load_intrinsics(self, scene: str) -> np.ndarray:
        """Load and cache intrinsics from intrinsics.txt.

        File format: "fx fy cx cy width height" (single line).
        Returns [fx, fy, cx, cy], dtype float32.
        """
        if scene not in self._intrinsics:
            intr_file = self.raw_data_root / scene / 'intrinsics.txt'
            with open(intr_file, 'r') as f:
                vals = [float(v) for v in f.read().split()]
            self._intrinsics[scene] = np.array(vals[:4], dtype=np.float32)
        return self._intrinsics[scene]
