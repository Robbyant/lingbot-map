"""VBR (Vision Benchmark in Rome) dataset loader.

Dataset format:
  {raw_data_root}/
    {scene}_processed_aligned/
      rgb/000000.png ... 000NNN.png   # one frame per file, sequential
      camera_pose.txt                 # TUM-format trajectory (aligned)
      intrinsics.txt                  # 3x3 K matrix
    processed_gt/
      {scene}_gt.txt                  # TUM-format ground-truth poses

Pose convention:
    Poses are CAMERA-TO-WORLD (C2W), TUM scalar-last quaternion:
        timestamp tx ty tz qx qy qz qw
    The 'timestamp' column is the integer frame index (0, 1, 2, ...),
    i.e. it directly identifies the rgb file 000000.png, 000001.png, ...
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from scipy.spatial.transform import Rotation

from benchmark.dataset.base import BaseDataset


SCENE_DIR_SUFFIX = '_processed_aligned'
GT_DIR_NAME = 'processed_gt'

# Patch alignment for downstream ViT-style methods (lingbot-map, vggt, ...).
_PATCH_ALIGN = 14


class VbrDataset(BaseDataset):
    """VBR dataset loader (RGB + C2W TUM trajectory + 3x3 intrinsics)."""

    def __init__(
        self,
        raw_data_root: str,
        scenes: Optional[List[str]] = None,
        target_size: Optional[List[int]] = None,
        logger=None,
    ):
        """Initialize VBR loader.

        Args:
            raw_data_root: Root containing `{scene}_processed_aligned/` dirs and
                           a sibling `processed_gt/` directory.
            scenes:        Optional whitelist of scene names (without the
                           '_processed_aligned' suffix). If None, auto-discover.
            target_size:   Optional [W, H] target. When set, frames are
                cover-fit resized (max scale, preserving aspect ratio) and
                center-cropped to exactly W×H, with intrinsics updated. Both
                dims must be multiples of 14. Resize uses cv2.INTER_AREA.
            logger:        Optional logger.
        """
        super().__init__(raw_data_root, logger=logger)
        self._scenes_whitelist = scenes
        self._scene_cache: Dict[str, Dict[str, Any]] = {}

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
        if self._scenes_whitelist is not None:
            return sorted(self._scenes_whitelist)
        found: List[str] = []
        for d in sorted(self.raw_data_root.iterdir()):
            if not d.is_dir() or not d.name.endswith(SCENE_DIR_SUFFIX):
                continue
            scene = d.name[:-len(SCENE_DIR_SUFFIX)]
            if (d / 'rgb').is_dir() and (d / 'intrinsics.txt').exists():
                found.append(scene)
        return found

    def get_frame_list(self, scene: str) -> List[int]:
        data = self._load_scene_data(scene)
        return list(range(len(data['rgb_files'])))

    def load_frame_data(self, scene: str, frame_id: int) -> Dict[str, Any]:
        data = self._load_scene_data(scene)
        rgb_path = data['rgb_files'][frame_id]
        rgb = np.array(Image.open(rgb_path).convert('RGB'), dtype=np.uint8)

        pose = data['poses_by_frame'].get(frame_id)
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

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _load_scene_data(self, scene: str) -> Dict[str, Any]:
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        scene_dir = self.raw_data_root / f"{scene}{SCENE_DIR_SUFFIX}"
        rgb_dir = scene_dir / 'rgb'
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f"VBR rgb dir not found: {rgb_dir}")

        rgb_files = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() == '.png')
        if not rgb_files:
            raise FileNotFoundError(f"No PNG frames in {rgb_dir}")

        intrinsics = _parse_intrinsics(scene_dir / 'intrinsics.txt')

        gt_path = self.raw_data_root / GT_DIR_NAME / f"{scene}_gt.txt"
        if gt_path.exists():
            poses_by_frame = _parse_tum_trajectory_indexed(gt_path)
        else:
            poses_by_frame = {}
            if self.logger is not None:
                self.logger.warning(
                    f"VBR scene {scene}: no GT at {gt_path}; "
                    f"trajectory will be all-NaN."
                )

        self._scene_cache[scene] = {
            'rgb_files': rgb_files,
            'poses_by_frame': poses_by_frame,
            'intrinsics': intrinsics,
        }
        return self._scene_cache[scene]


# ------------------------------------------------------------------ #
#  Module-level helpers                                                #
# ------------------------------------------------------------------ #

def _parse_intrinsics(path: Path) -> np.ndarray:
    """Read a 3x3 intrinsics matrix and return [fx, fy, cx, cy]."""
    K = np.loadtxt(path, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics in {path}, got shape {K.shape}")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.array([fx, fy, cx, cy], dtype=np.float32)


def _parse_tum_trajectory_indexed(path: Path) -> Dict[int, np.ndarray]:
    """Parse a TUM-style trajectory; key each pose by its integer timestamp.

    Each non-comment line: timestamp tx ty tz qx qy qz qw
    The timestamp is interpreted as an integer frame index.
    """
    poses_by_frame: Dict[int, np.ndarray] = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = line.split()
            if len(vals) < 8:
                continue
            frame_idx = int(float(vals[0]))
            tx, ty, tz = float(vals[1]), float(vals[2]), float(vals[3])
            qx, qy, qz, qw = float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R
            c2w[:3, 3] = [tx, ty, tz]
            poses_by_frame[frame_idx] = c2w
    return poses_by_frame
