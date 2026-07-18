"""ETH3D high-resolution multi-view stereo dataset (pi3 / custom_undistorted variant).

Dataset structure (per scene, eth3d_pi3 format):
  {raw_data_root}/{scene}/
    images/custom_undistorted/*.JPG              - Already-undistorted RGB images (6048x4032)
    ground_truth_depth/custom_undistorted/*.JPG  - Raw float32 depth (binary, not JPEG)
    custom_undistorted_cam/*.npz                 - Per-frame calibration (intrinsics + extrinsics)
    combined_mesh.ply                            - GT mesh (not used; depth backprojection instead)

NPZ calibration format (per frame):
  intrinsics: (3, 3) float32  - standard pinhole K matrix, for original 6048x4032 resolution
  extrinsics: (4, 4) float64  - W2C transformation matrix (COLMAP convention; convert to C2W)

Notes:
  - Frame set is determined by *.npz files in custom_undistorted_cam/ (not a COLMAP images.txt).
  - Images are already undistorted; no fisheye correction needed.
  - No masks_for_images/ directory in eth3d_pi3; depth>0 is used as the sole validity criterion.
  - Images are loaded and stored at load_img_size width (default 518) to save disk space;
    the original 6048x4032 DSLR resolution is impractical for BSS storage.
  - COLMAP uses W2C convention; extrinsics are converted to C2W for BSS trajectory format.
  - Depth files have .JPG extension but are raw float32 binary (not JPEG images).

Evaluation protocol (depth backprojection + pixel-level Umeyama + ICP, same as 7scenes):
  - GT point cloud built from GT depth maps via load_point_cloud_grid() (no mesh sampling needed).
  - Umeyama coarse alignment uses GT resized to pred resolution for pixel-level correspondences.
  - ICP fine registration against full-resolution GT point cloud.

Reference: https://www.eth3d.net/
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional

from benchmark.core.loader import BSSLoader
from benchmark.dataset.base import BaseDataset


SCENES = [
    "courtyard", 
    "delivery_area", 
    "electro", 
    "facade", 
    "kicker",
    "meadow",
    "office", 
    "pipes", 
    "playground", 
    "relief", 
    "relief_2",
    "terrace", 
    "terrains",
]

# ALIGN WITH DA#
DA3_FILTER_SCENES = [
    "meadow", "terrace"
]

# DA3_FILTER_SCENES = None  # No scene-level filtering; use all scenes

# Images to filter out (known problematic views per scene)
DA3_FILTER_KEYS = {
    "delivery_area": ["711.JPG", "712.JPG", "713.JPG", "714.JPG"],
    "electro": ["9289.JPG", "9290.JPG", "9291.JPG", "9292.JPG", "9293.JPG", "9298.JPG"],
    "playground": ["587.JPG", "588.JPG", "589.JPG", "590.JPG", "591.JPG", "592.JPG"],
    "relief": [
        "427.JPG", "428.JPG", "429.JPG", "430.JPG", "431.JPG", "432.JPG",
        "433.JPG", "434.JPG", "435.JPG", "436.JPG", "437.JPG", "438.JPG",
    ],
    "relief_2": [
        "458.JPG", "459.JPG", "460.JPG", "461.JPG", "462.JPG", "463.JPG",
        "464.JPG", "465.JPG", "466.JPG", "467.JPG", "468.JPG",
    ],
}
# DA3_FILTER_KEYS = {}

# Evaluation parameters
# 0.25 m threshold for both ICP convergence and precision/recall.
# ETH3D is an outdoor dataset at larger scale than 7scenes (which uses 0.1 m).
EVAL_THRESHOLD = 0.05 * 5         # 0.25 m

def _w2c_to_c2w(R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    """Convert W2C (rotation + translation) to 4x4 C2W matrix."""
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_cw
    c2w[:3, 3] = t_cw
    return c2w


class Eth3dDataset(BaseDataset):
    """ETH3D multi-view stereo dataset (pi3 / custom_undistorted variant)."""

    def __init__(self, raw_data_root: str, load_img_size: int = 518, logger=None):
        super().__init__(raw_data_root, logger=logger)
        self.load_img_size = load_img_size
        self._cache: Dict[str, List[str]] = {}  # scene -> sorted list of frame base names

    def _get_frame_names(self, scene: str) -> List[str]:
        """Return sorted list of frame base names (without extension) from custom_undistorted_cam/.

        Frames listed in DA3_FILTER_KEYS[scene] are excluded when DA3_FILTER_KEYS is not None.
        Filter keys are matched as exact filenames (e.g. '711.JPG' excludes stem '711').
        """
        if scene not in self._cache:
            cam_dir = self.raw_data_root / scene / 'custom_undistorted_cam'
            names = sorted(p.stem for p in cam_dir.glob('*.npz'))
            if DA3_FILTER_KEYS is not None:
                bad = set(DA3_FILTER_KEYS.get(scene, []))
                if bad:
                    before = len(names)
                    names = [n for n in names if not any(f'{n}.JPG'.endswith(k) for k in bad)]
                    self.logger.info(
                        f"Scene '{scene}': filtered out {before - len(names)} frames "
                        f"[Total: {before}] based on DA3_FILTER_KEYS"
                    )
                    
            self._cache[scene] = names
        return self._cache[scene]

    def get_scenes(self) -> List[str]:
        full_scenes = [s for s in SCENES if (self.raw_data_root / s).exists()]

        scenes = []
        for s in full_scenes:
            if DA3_FILTER_SCENES is not None and s in DA3_FILTER_SCENES:
                self.logger.info(f"Scene '{s}' is filtered out based on DA3_FILTER_SCENES")
            else:
                scenes.append(s)

        return scenes

    def get_frame_list(self, scene: str) -> List[int]:
        return list(range(len(self._get_frame_names(scene))))

    def load_frame_data(self, scene: str, frame_id: int) -> Dict[str, Any]:
        scene_dir = self.raw_data_root / scene
        name = self._get_frame_names(scene)[frame_id]  # e.g. 'DSC_0286'

        # Load per-frame calibration from NPZ
        npz = np.load(scene_dir / 'custom_undistorted_cam' / f'{name}.npz')
        K = npz['intrinsics']   # (3, 3) float32, for original 6048x4032 resolution
        E = npz['extrinsics']   # (4, 4) float64, W2C matrix

        intrinsics = np.array(
            [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32
        )  # [fx, fy, cx, cy]
        c2w = _w2c_to_c2w(E[:3, :3].astype(np.float32), E[:3, 3].astype(np.float32))

        # RGB (already undistorted)
        rgb = np.array(
            Image.open(
                scene_dir / 'images' / 'custom_undistorted' / f'{name}.JPG'
            ).convert('RGB'),
            dtype=np.uint8,
        )

        # Depth: same raw float32 binary format as the original eth3d dataset.
        # File extension is .JPG but content is raw float32 (not a JPEG image).
        depth_path = scene_dir / 'ground_truth_depth' / 'custom_undistorted' / f'{name}.JPG'
        orig_h, orig_w = rgb.shape[:2]
        if depth_path.exists():
            depth = np.fromfile(str(depth_path), dtype=np.float32).reshape(orig_h, orig_w)
            depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
        else:
            depth = np.zeros((orig_h, orig_w), dtype=np.float32)

        # No masks_for_images/ in eth3d_pi3; depth>0 is the sole validity criterion.

        # Resize to load_img_size width (maintaining aspect ratio) to reduce BSS storage.
        # The original 6048x4032 DSLR resolution is impractical for BSS storage.
        if self.load_img_size > 0 and orig_w != self.load_img_size:
            target_width = self.load_img_size
            h, w = rgb.shape[0], rgb.shape[1]

            aspect_ratio = float(h) / w
            target_height = int(target_width * aspect_ratio)
            # Round to nearest multiple of 14 for patch-based models
            target_height = (target_height // 14) * 14

            rgb   = cv2.resize(rgb,   (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            intrinsics[0] *= target_width / orig_w  # fx
            intrinsics[2] *= target_width / orig_w  # cx
            intrinsics[1] *= target_height / orig_h  # fy
            intrinsics[3] *= target_height / orig_h  # cy

        return {
            'rgb':        rgb,
            'depth':      depth,
            'pose':       c2w,
            'intrinsics': intrinsics,
        }

    def load_global_data(self, scene: str) -> Dict[str, Any]:
        """No global data for ETH3D; GT point cloud is built from depth during evaluation."""
        return {}

    @staticmethod
    def evaluate_pointcloud(
        gt_loader: BSSLoader,
        pred_loader: BSSLoader,
        logger,
        options: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate 3D reconstruction against ETH3D GT depth (7scenes-style pipeline).

        Pipeline:
          1. GT resized to pred resolution -> pixel-level Umeyama coarse alignment.
          2. ICP fine registration: aligned pred vs. full-resolution GT point cloud.
          3. Precision/recall/F1 evaluation at EVAL_THRESHOLD (0.25 m).

        This mirrors SevenScenesDataset.evaluate_pointcloud() exactly, with a larger
        ICP threshold (0.25 m vs 0.1 m) to match ETH3D's outdoor metric scale.

        Args:
            gt_loader:   BSSLoader for ground truth.
            pred_loader: BSSLoader for method output.
            logger:      Logger instance.
            options:     Optional dict; supported keys:
                           icp_threshold (float, default EVAL_THRESHOLD = 0.25)

        Returns:
            Dict with chamfer, accuracy, completeness, precision, recall, f1,
            pred_points, gt_points, thresholds. None on failure.
        """
        from benchmark.geometry.registration import umeyama_registration, icp_registration, apply_transform
        from benchmark.evaluation.points import evaluate_pointcloud as eval_pc

        icp_threshold = (options or {}).get('icp_threshold', 0.1)

        # --- Step 1: GT point cloud for pixel-level Umeyama alignment ---
        gt_xyzrgb_u, gt_mask_u = gt_loader.load_point_cloud_grid()

        # --- Step 2: Pred point cloud (auto-selects points or depth backprojection) ---
        try:
            pred_xyzrgb, pred_mask = pred_loader.load_point_cloud_grid()
        except Exception as e:
            logger.warning(f"ETH3D eval: cannot build pred cloud ({e})")
            return None

        # --- Step 3: Align GT frames to pred keyframes (sparse SLAM support) ---
        pred_frame_indices = pred_loader.get_frame_indices()
        gt_xyzrgb_u = gt_xyzrgb_u[pred_frame_indices]
        gt_mask_u   = gt_mask_u[pred_frame_indices]

        # --- Step 4: Pixel-level Umeyama alignment ---
        common_mask = gt_mask_u & pred_mask
        gt_pts_u    = gt_xyzrgb_u[common_mask][:, :3]
        pred_pts_u  = pred_xyzrgb[common_mask][:, :3]

        if len(gt_pts_u) < 6:
            logger.warning(
                f"ETH3D eval: insufficient correspondences ({len(gt_pts_u)}) for Umeyama"
            )
            return None

        logger.info(f"ETH3D eval: Umeyama with {len(gt_pts_u):,} correspondences")
        T_umeyama = umeyama_registration(
            source_points=pred_pts_u,
            target_points=gt_pts_u,
        )
        logger.info(f"Umeyama transform:\n{T_umeyama}")

        # --- Step 5: Full-resolution GT point cloud for ICP and evaluation ---
        gt_xyzrgb_full, gt_mask_full = gt_loader.load_point_cloud_grid()
        gt_pts = gt_xyzrgb_full[gt_mask_full][:, :3]

        # --- Step 6: ICP fine registration ---
        all_pred_pts = pred_xyzrgb[common_mask][:, :3]
        all_pred_after_umeyama = apply_transform(all_pred_pts, T_umeyama)

        logger.info(f"ETH3D eval: ICP with threshold {icp_threshold:.3f} m")
        T_icp = icp_registration(
            source_points=all_pred_after_umeyama,
            target_points=gt_pts,
            icp_threshold=icp_threshold,
        )
        logger.info(f"ICP transform:\n{T_icp}")

        T_total = T_icp @ T_umeyama
        all_pred_aligned = apply_transform(all_pred_pts, T_total)

        logger.info(
            f"ETH3D final eval: {len(all_pred_aligned):,} pred pts "
            f"vs {len(gt_pts):,} gt pts"
        )
        return eval_pc(
            source_points=all_pred_aligned,
            target_points=gt_pts,
            thresholds=[EVAL_THRESHOLD],
        )