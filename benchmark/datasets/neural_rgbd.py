"""Neural RGB-D dataset loader.

Dataset format:
  {raw_data_root}/
    {scene_name}/
      images/img{N}.png       - RGB images (640x480, uint8)
      depth/depth{N}.png      - 16-bit PNG depth maps (millimeters)
      poses.txt               - Camera poses (4x4 C2W in OpenGL convention, 4 lines per matrix)
      focal.txt               - Focal length scalar (fx = fy)

Coordinate convention in poses.txt:
  - Each 4x4 matrix is a Camera-to-World (C2W) transform in OpenGL convention
    (camera looks down -Z, Y up).
  - Converted to OpenCV convention (camera looks down +Z, Y down) by negating
    the y and z columns: pose[:, 1:3] *= -1.0

Camera intrinsics:
  - Fixed per scene, read from focal.txt: fx = fy = focal, cx = 320, cy = 240
  - Native resolution: 640x480

Depth filtering:
  - Values in millimeters; converted to meters by dividing by 1000
  - Depths > 10 m or < 1 mm are set to 0 (invalid)

Reference:
  https://github.com/dazinovic/neural-rgbd-surface-reconstruction
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional

from benchmark.core.loader import BSSLoader
from benchmark.dataset.base import BaseDataset

# Native camera resolution
NRGBD_WIDTH = 640
NRGBD_HEIGHT = 480

# Fixed principal point (image center at native resolution)
NRGBD_CX = 320.0
NRGBD_CY = 240.0

# Depth validity range (meters)
NRGBD_DEPTH_MAX = 10.0
NRGBD_DEPTH_MIN = 1e-3

# Point cloud evaluation ICP threshold (meters), matching Pi3 evaluation setting
NRGBD_ICP_THRESHOLD = 0.1


class NeuralRgbdDataset(BaseDataset):
    """Neural RGB-D dataset loader.

    Supports all 9 indoor scenes:
      breakfast_room, complete_kitchen, green_room, grey_white_room, kitchen,
      morning_apartment, staircase, thin_geometry, whiteroom

    Evaluation protocol (depth backprojection + Umeyama + ICP, same as 7Scenes):
      - GT point cloud built from depth backprojection via load_point_cloud_grid().
      - Umeyama coarse alignment uses GT resized to pred resolution for pixel-level
        correspondences.
      - ICP fine registration against full-resolution GT point cloud.
    """

    def __init__(self, raw_data_root: str, logger=None):
        """Initialize Neural RGB-D dataset loader.

        Args:
            raw_data_root: Path to dataset root (contains per-scene subdirectories).
            logger: Optional logger instance.
        """
        super().__init__(raw_data_root, logger=logger)
        # Lazy-loaded caches to avoid re-reading large poses files
        self._pose_cache: Dict[str, np.ndarray] = {}   # scene -> (N, 4, 4) C2W
        self._focal_cache: Dict[str, float] = {}        # scene -> focal length

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_poses(self, scene: str) -> np.ndarray:
        """Read poses.txt and return all C2W matrices in OpenCV convention.

        poses.txt contains 4×4 matrices stored 4 lines each (row-major),
        in OpenGL convention (camera looks -Z, Y up). This method applies
        the GL→CV conversion by negating the y and z column vectors.

        Args:
            scene: Scene name (e.g., 'breakfast_room').

        Returns:
            (N, 4, 4) float32 array of C2W matrices in OpenCV convention.
        """
        if scene in self._pose_cache:
            return self._pose_cache[scene]

        pose_file = self.raw_data_root / scene / 'poses.txt'
        with open(pose_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        poses = []
        for i in range(0, len(lines), 4):
            mat = np.array(
                [[float(x) for x in lines[i + r].split()] for r in range(4)],
                dtype=np.float32,
            )  # (4, 4) OpenGL C2W
            # GL to CV: negate y-axis column (col 1) and z-axis column (col 2)
            mat[:, 1:3] *= -1.0
            poses.append(mat)

        self._pose_cache[scene] = np.stack(poses, axis=0)  # (N, 4, 4)
        return self._pose_cache[scene]

    def _load_focal(self, scene: str) -> float:
        """Read focal length from focal.txt.

        Args:
            scene: Scene name.

        Returns:
            Focal length in pixels (fx = fy).
        """
        if scene not in self._focal_cache:
            focal_file = self.raw_data_root / scene / 'focal.txt'
            self._focal_cache[scene] = float(focal_file.read_text().strip())
        return self._focal_cache[scene]

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def get_scenes(self) -> List[str]:
        """Return sorted list of scene names.

        Only directories containing a valid poses.txt are included,
        which correctly excludes any archives (e.g., neural_rgbd_data.zip).

        Returns:
            List of scene name strings (e.g., ['breakfast_room', 'kitchen', ...]).
        """
        scenes = []
        for d in sorted(self.raw_data_root.iterdir()):
            if d.is_dir() and (d / 'poses.txt').exists():
                scenes.append(d.name)
        return scenes

    def get_frame_list(self, scene: str) -> List[int]:
        """Return frame indices for a scene.

        Indices are 0-based integers matching the img{N}.png / depth{N}.png
        file naming convention.

        Args:
            scene: Scene name.

        Returns:
            List of integer frame IDs.
        """
        poses = self._load_poses(scene)
        return list(range(len(poses)))

    def load_frame_data(self, scene: str, frame_id: int) -> Dict[str, Any]:
        """Load a single frame.

        Args:
            scene:    Scene name (e.g., 'breakfast_room').
            frame_id: Integer frame index.

        Returns:
            Dictionary with:
            - 'rgb':        (H, W, 3) uint8 RGB image.
            - 'depth':      (H, W) float32 depth map in meters (0 = invalid).
            - 'pose':       (4, 4) float32 C2W matrix (OpenCV convention).
            - 'intrinsics': [fx, fy, cx, cy] float32 array.
        """
        scene_dir = self.raw_data_root / scene

        # RGB
        rgb_path = scene_dir / 'images' / f'img{frame_id}.png'
        rgb = np.array(Image.open(rgb_path).convert('RGB'), dtype=np.uint8)

        # Depth: 16-bit PNG in millimeters → float32 meters
        depth_path = scene_dir / 'depth' / f'depth{frame_id}.png'
        depth = np.array(Image.open(depth_path), dtype=np.float32)
        depth = np.nan_to_num(depth, nan=0.0) / 1000.0  # mm to meters
        depth[depth > NRGBD_DEPTH_MAX] = 0.0             # too far
        depth[depth < NRGBD_DEPTH_MIN] = 0.0             # too near

        # Pose: C2W in OpenCV convention (converted during _load_poses)
        c2w = self._load_poses(scene)[frame_id]

        # Intrinsics: fx = fy = focal (from focal.txt), cx/cy fixed at image center
        focal = self._load_focal(scene)
        intrinsics = np.array([focal, focal, NRGBD_CX, NRGBD_CY], dtype=np.float32)

        return {
            'rgb':        rgb,
            'depth':      depth,
            'pose':       c2w,
            'intrinsics': intrinsics,
        }

    def load_global_data(self, scene: str) -> Dict[str, Any]:
        """Return empty dict; GT point cloud is built from depth during evaluation."""
        return {}

    # ------------------------------------------------------------------
    # Point cloud evaluation (same pipeline as 7Scenes / ETH3D)
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_pointcloud(
        gt_loader: BSSLoader,
        pred_loader: BSSLoader,
        logger,
        options: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate 3D reconstruction against Neural RGB-D GT depth.

        Pipeline:
          1. GT resized to pred resolution for pixel-level Umeyama coarse alignment.
          2. ICP fine registration of aligned pred against full-resolution GT cloud.
          3. Precision / recall / F1 at icp_threshold (default 0.1 m).

        Args:
            gt_loader:   BSSLoader for the ground-truth directory.
            pred_loader: BSSLoader for the method output directory.
            logger:      Logger instance.
            options:     Optional dict of evaluation options:
                           icp_threshold (float, default 0.1)

        Returns:
            Dict with chamfer, accuracy, completeness, precision, recall, f1,
            pred_points, gt_points, thresholds. None on failure.
        """
        from benchmark.geometry.registration import (
            apply_transform,
            icp_registration,
            umeyama_registration,
            voxel_downsample,
        )
        from benchmark.evaluation.points import evaluate_pointcloud as eval_pc

        icp_threshold = (options or {}).get('icp_threshold', NRGBD_ICP_THRESHOLD)
        voxel_size = (options or {}).get('voxel_size', 4.0 / 512.0)

        # --- Step 1: GT point cloud (full, for ICP/eval; subset for Umeyama) ---
        gt_xyzrgb_full, gt_mask_full = gt_loader.load_point_cloud_grid()

        # --- Step 2: Pred point cloud ---
        try:
            pred_xyzrgb, pred_mask = pred_loader.load_point_cloud_grid()
        except Exception as e:
            logger.warning(f"NRGBD eval: cannot build pred cloud ({e})")
            return None

        # --- Step 3: Match GT frames to pred keyframes (sparse SLAM support) ---
        pred_frame_indices = pred_loader.get_frame_indices()
        gt_xyzrgb_u = gt_xyzrgb_full[pred_frame_indices]
        gt_mask_u = gt_mask_full[pred_frame_indices]

        # --- Step 4: Pixel-level Umeyama coarse alignment ---
        common_mask = gt_mask_u & pred_mask
        gt_pts_u = gt_xyzrgb_u[common_mask][:, :3]
        pred_pts_u = pred_xyzrgb[common_mask][:, :3]

        if len(gt_pts_u) < 6:
            logger.warning(
                f"NRGBD eval: insufficient correspondences ({len(gt_pts_u)}) for Umeyama"
            )
            return None

        logger.info(f"NRGBD eval: Umeyama with {len(gt_pts_u):,} correspondences")
        T_umeyama = umeyama_registration(
            source_points=pred_pts_u,
            target_points=gt_pts_u,
        )
        logger.info(f"Umeyama transform:\n{T_umeyama}")

        # --- Step 5: Full-resolution GT point cloud for ICP and evaluation ---
        gt_pts = gt_xyzrgb_full[gt_mask_full][:, :3]

        # --- Step 6: ICP fine registration ---
        all_pred_pts = pred_xyzrgb[common_mask][:, :3]
        all_pred_after_umeyama = apply_transform(all_pred_pts, T_umeyama)

        # Voxel downsample once — used for both ICP and eval_pc (same as DA3 bench)
        if voxel_size > 0:
            logger.info(
                f"NRGBD eval: voxel downsampling at {voxel_size:.6f}m "
                f"(pred: {len(all_pred_after_umeyama):,}, gt: {len(gt_pts):,})"
            )
            pred_ds = voxel_downsample(all_pred_after_umeyama, voxel_size)
            gt_ds   = voxel_downsample(gt_pts, voxel_size)
            logger.info(
                f"NRGBD eval: after downsampling: pred={len(pred_ds):,}, gt={len(gt_ds):,}"
            )
        else:
            pred_ds = all_pred_after_umeyama
            gt_ds   = gt_pts

        logger.info(f"NRGBD eval: ICP with threshold {icp_threshold:.3f} m")
        T_icp = icp_registration(
            source_points=pred_ds,
            target_points=gt_ds,
            icp_threshold=icp_threshold,
        )
        logger.info(f"ICP transform:\n{T_icp}")

        pred_pts_eval = apply_transform(pred_ds, T_icp)
        gt_pts_eval   = gt_ds

        # DEBUG: save intermediate clouds to method dir for alignment inspection
        def _save_ply(path, pts):
            import numpy as _np
            pts = _np.asarray(pts, dtype=_np.float32)
            header = (
                f"ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(pts)}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"end_header\n"
            )
            with open(path, 'wb') as f:
                f.write(header.encode())
                f.write(pts.tobytes())

        # _debug_dir = pred_loader.artifact.root
        # _save_ply(_debug_dir / 'debug_gt.ply',                 gt_ds)
        # _save_ply(_debug_dir / 'debug_pred_raw.ply',           pred_xyzrgb[common_mask][:, :3])
        # _save_ply(_debug_dir / 'debug_pred_after_umeyama.ply', pred_ds)
        # _save_ply(_debug_dir / 'debug_pred_after_icp.ply',     pred_pts_eval)
        # logger.info(f"NRGBD debug: saved 4 intermediate clouds to {_debug_dir}")

        logger.info(
            f"NRGBD final eval: {len(pred_pts_eval):,} pred pts "
            f"vs {len(gt_pts_eval):,} gt pts"
        )
        return eval_pc(
            source_points=pred_pts_eval,
            target_points=gt_pts_eval,
            thresholds=[0.05],
        )
