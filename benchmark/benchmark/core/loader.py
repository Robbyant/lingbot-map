"""BSS data loader.

Provides utilities to load data from standardized BSS format directories.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2

from benchmark.io.image import load_rgb, load_exr
from benchmark.io.trajectory import read_trajectory
from benchmark.io.intrinsics import read_intrinsics
from benchmark.io.pointcloud import load_point_cloud_ply
from benchmark.core.storage import BSSArtifact
from benchmark.geometry.resize import (
    ResizeContext,
    adjust_intrinsics,
)

DEFAULT_CONFIDENCE_VALUE = -1.0


class BSSLoader:
    """Loads data from a BSS artifact directory."""

    def __init__(
        self,
        artifact: BSSArtifact,
        resize_context: Optional[ResizeContext] = None,
        context: object = None,
    ):
        """Initialize BSS loader.

        Args:
            artifact:       BSSArtifact describing the directory to load from
            resize_context: Optional ResizeContext controlling how images are resized.
                            If None, images are loaded at native resolution.
            context:        Optional dataset or method instance; used to dispatch
                            __load_{key}_file__ for custom key loading.
        """
        self.artifact = artifact
        if not artifact.exists():
            raise FileNotFoundError(f"BSS directory not found: {artifact}")
        self.resize_context = resize_context
        self.context = context

    def copy(self) -> 'BSSLoader':
        """Return a new BSSLoader with the same artifact, resize_context, and context."""
        return BSSLoader(self.artifact, self.resize_context, self.context)

    # ------------------------------------------------------------------
    # Frame count / metadata helpers
    # ------------------------------------------------------------------

    def get_num_frames(self) -> int:
        """Get the number of frames from .complete.json metadata.

        Returns:
            Number of frames in this scene

        Raises:
            ValueError: If .complete.json is missing or num_frames not found
        """
        metadata = self.artifact.read_metadata()
        if metadata is None:
            raise ValueError(f"Scene not complete or metadata missing: {self.artifact}")

        num_frames = metadata.get('num_frames')
        if num_frames is None:
            raise ValueError(f"No num_frames in metadata: {self.artifact}")

        return int(num_frames)

    def get_frame_indices(self) -> List[int]:
        """Get the original GT frame indices for stored frames.

        For sparse SLAM outputs (K < N), returns the 'frame_index_map' list stored
        in .complete.json, which maps each stored frame position to its original
        GT frame index.

        For dense methods (K == N), returns identity [0, 1, ..., N-1] as the map
        is absent from .complete.json.

        Returns:
            List of length K with original GT frame indices
        """
        metadata = self.artifact.read_metadata()
        if metadata is not None and 'frame_index_map' in metadata:
            return list(metadata['frame_index_map'])
        return list(range(self.get_num_frames()))

    def get_image_dimensions(self) -> Tuple[int, int]:
        """Get image dimensions from metadata.

        Returns:
            (width, height) tuple
        """
        metadata = self.artifact.read_metadata()
        if metadata is None:
            raise ValueError(f"Scene not complete or metadata missing: {self.artifact}")

        width = metadata.get('image_width')
        height = metadata.get('image_height')

        if width is None or height is None:
            raise ValueError(f"Image dimensions not in metadata: {self.artifact}")

        return width, height

    def get_processing_dimensions(self) -> Tuple[int, int]:
        """Get processing dimensions (what methods see after resize).

        Returns:
            (width, height) tuple
        """
        orig_w, orig_h = self.get_image_dimensions()
        if self.resize_context and self.resize_context.enabled:
            transform = self.resize_context.get_transform(orig_w, orig_h)
            return transform.final_size
        return orig_w, orig_h

    # ------------------------------------------------------------------
    # Frame key presence
    # ------------------------------------------------------------------

    def has_frame_key(self, key: str) -> bool:
        """Check whether a frame-level data key is present."""
        return self.artifact.has_frame_key(key)

    # ------------------------------------------------------------------
    # Custom key loading
    # ------------------------------------------------------------------

    def load_custom_global_data(self, key: str) -> Optional[Any]:
        """Load global-level custom data via context's __load_{key}_file__ method."""
        if self.context is None:
            return None
        loader_fn = getattr(self.context, f'__load_{key}_file__', None)
        if loader_fn is None or not callable(loader_fn):
            return None
        return loader_fn(self.artifact.root)

    def load_custom_frame_data(self, key: str) -> Optional[List[Any]]:
        """Load per-frame custom data via context's __load_{key}_file__ method."""
        if self.context is None:
            return None
        loader_fn = getattr(self.context, f'__load_{key}_file__', None)
        if loader_fn is None or not callable(loader_fn):
            return None
        N = self.get_num_frames()
        return [loader_fn(self.artifact.root, f"{idx:06d}") for idx in range(N)]

    def has_custom_frame_data(self, key: str) -> bool:
        """Check if a frame key was saved (from .complete.json frame_keys)."""
        metadata = self.artifact.read_metadata()
        if metadata is None:
            return False
        return key in metadata.get('frame_keys', [])

    def has_custom_global_data(self, key: str) -> bool:
        """Check if a global key was saved (from .complete.json global_keys)."""
        metadata = self.artifact.read_metadata()
        if metadata is None:
            return False
        return key in metadata.get('global_keys', [])

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_rgb_list(self) -> List[Optional[np.ndarray]]:
        """Load all RGB images in frame index order (000000.png, 000001.png, ...).

        For sparse outputs (e.g. SLAM keyframe-only), frames without a saved file
        produce a None entry rather than raising FileNotFoundError.

        Returns:
            List of length N; each entry is either a HxWx3 RGB image (uint8) resized
            per resize_context, or None if the file is absent for that frame.

        Raises:
            FileNotFoundError: If the RGB directory itself does not exist.
        """
        N = self.get_num_frames()
        rgb_dir = self.artifact.rgb_dir

        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

        rgb_list = []
        for idx in range(N):
            rgb_file = rgb_dir / f"{idx:06d}.png"
            if not rgb_file.exists():
                rgb_list.append(None)
                continue
            rgb = load_rgb(rgb_file)

            if self.resize_context and self.resize_context.enabled:
                rgb = self.resize_context.apply(rgb, cv2.INTER_LINEAR)

            rgb_list.append(rgb)

        return rgb_list

    def load_depth_list(self) -> Optional[List[Optional[np.ndarray]]]:
        """Load all depth maps in frame index order.

        Returns:
            List of HxW depth maps (float32, meters), resized if resize_context is set.
            Individual entries may be None if the file is missing for that frame.
            Returns None if depth directory does not exist.
        """
        depth_dir = self.artifact.depth_dir
        if not depth_dir.exists():
            return None

        N = self.get_num_frames()
        depth_list = []

        for idx in range(N):
            depth_file = depth_dir / f"{idx:06d}.exr"
            if depth_file.exists():
                depth = load_exr(depth_file)

                if self.resize_context and self.resize_context.enabled:
                    depth = self.resize_context.apply_nearest(depth)

                depth_list.append(depth)
            else:
                depth_list.append(None)

        return depth_list

    def load_points_list(self) -> Optional[List[Optional[np.ndarray]]]:
        """Load all point cloud maps in frame index order.

        Returns:
            List of HxWx3 or HxWx6 point cloud maps (float32), or None if not available
        """
        point_dir = self.artifact.points_dir
        if not point_dir.exists():
            return None

        N = self.get_num_frames()
        point_list = []

        for idx in range(N):
            point_file = point_dir / f"{idx:06d}.exr"
            if point_file.exists():
                points = load_exr(point_file)

                if self.resize_context and self.resize_context.enabled:
                    points = self.resize_context.apply_nearest(points)

                point_list.append(points)
            else:
                point_list.append(None)

        return point_list

    def load_confidence_list(self) -> Optional[List[Optional[np.ndarray]]]:
        """Load all confidence maps in frame index order.

        Returns:
            List of HxW confidence maps (float32), resized if resize_context is set.
            Individual entries may be None if the file is missing for that frame.
            Returns None if confidence directory does not exist.
        """
        conf_dir = self.artifact.confidence_dir
        if not conf_dir.exists():
            return None

        N = self.get_num_frames()
        conf_list = []

        for idx in range(N):
            conf_file = conf_dir / f"{idx:06d}.exr"
            if conf_file.exists():
                conf = load_exr(conf_file)

                if self.resize_context and self.resize_context.enabled:
                    conf = self.resize_context.apply(conf, cv2.INTER_LINEAR)

                conf_list.append(conf)
            else:
                conf_list.append(None)

        return conf_list

    def load_mask_list(self) -> Optional[List[Optional[np.ndarray]]]:
        """Load all mask images in frame index order.

        Returns:
            List of HxW mask arrays (bool), or None if not available
        """
        mask_dir = self.artifact.mask_dir
        if not mask_dir.exists():
            return None

        N = self.get_num_frames()
        mask_list = []

        for idx in range(N):
            mask_file = mask_dir / f"{idx:06d}.png"
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                mask = mask.astype(bool)

                if self.resize_context and self.resize_context.enabled:
                    mask_uint8 = mask.astype(np.uint8) * 255
                    mask_uint8 = self.resize_context.apply_nearest(mask_uint8)
                    mask = mask_uint8.astype(bool)

                mask_list.append(mask)
            else:
                mask_list.append(None)

        return mask_list

    # ------------------------------------------------------------------
    # Trajectory / intrinsics
    # ------------------------------------------------------------------

    def load_trajectory(self) -> Optional[np.ndarray]:
        """Load camera trajectory.

        Returns:
            np.ndarray of shape (N, 4, 4), with NaN 4x4 matrices for missing poses.
            N is always equal to get_num_frames() so that sparse keyframe-only files
            still produce a correctly-sized array.
            Returns None if traj.txt does not exist.
        """
        if not self.artifact.traj_file.exists():
            return None
        return read_trajectory(self.artifact.traj_file, self.get_num_frames())

    def load_intrinsics(self) -> Optional[np.ndarray]:
        """Load camera intrinsics.

        Returns:
            np.ndarray of shape (N, 4) with [fx, fy, cx, cy] per row.
            NaN rows where intrinsics are missing.
            Intrinsics are adjusted for the resize transform when enabled.
            Returns None if intrinsics.txt does not exist.
        """
        if not self.artifact.intrinsics_file.exists():
            return None

        intrinsics_array = read_intrinsics(self.artifact.intrinsics_file)

        if self.resize_context and self.resize_context.enabled:
            orig_w, orig_h = self.get_image_dimensions()
            transform = self.resize_context.get_transform(orig_w, orig_h)
            for i in range(len(intrinsics_array)):
                if not np.any(np.isnan(intrinsics_array[i])):
                    intrinsics_array[i] = adjust_intrinsics(intrinsics_array[i], transform)

        return intrinsics_array

    def load_traj_transform(self) -> Optional[np.ndarray]:
        """Load trajectory alignment transform from traj_transform.txt.

        Returns:
            4x4 Sim(3) transformation matrix, or None if file does not exist
            or has an unexpected shape.
        """
        f = self.artifact.traj_transform_file
        if not f.exists():
            return None
        try:
            T = np.loadtxt(f)
            return T if T.shape == (4, 4) else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Point cloud
    # ------------------------------------------------------------------

    def load_global_points(self) -> Optional[np.ndarray]:
        """Load global point cloud from points.ply.

        Returns:
            Nx3 or Nx6 array of global points, or None if not available
        """
        if not self.artifact.global_points_file.exists():
            return None
        return load_point_cloud_ply(self.artifact.global_points_file)

    def _resolve_aoi_mask_dir(self) -> Optional[Path]:
        """Find mask/ directory: check self.artifact.root first, then sibling gt/."""
        local = self.artifact.mask_dir
        if local.exists():
            return local
        gt_sibling = self.artifact.root.parent / 'gt' / 'mask'
        if gt_sibling.exists():
            return gt_sibling
        return None

    def load_point_cloud_grid(
        self,
        confidence_threshold: float = 0.0,
        use_aoi_mask: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load colored point cloud in world coordinates.

        This method loads colored 3D point clouds in world coordinate system from two sources:
        (1) If load_point_list() returns point cloud data, use the point cloud xyz directly
        (2) If load_point_list() returns no point cloud, use load_depth_list() to get depth maps,
            combine with load_intrinsics() camera intrinsics to backproject depth into xyz points,
            and use load_trajectory() camera poses to transform points from camera to world coords.
        Then combine with load_rgb_list() RGB images to get point cloud color information.

        Args:
            confidence_threshold: Percentile threshold (0-1) to filter points based on confidence maps.
                                If > 0, only points with confidence above this percentile are kept.
                                Default 0.0 means no filtering.
            use_aoi_mask: If True, apply area-of-interest masks from mask/ directory
                         (checks data_dir/mask/ first, then sibling gt/mask/).
                         Only pixels where the mask is non-zero are kept.
                         Default False. Typically only enabled during evaluation.

        Returns:
            xyzrgb: DxHxWx6 float32 array (xyz in meters, rgb in [0,1])
            mask: DxHxW boolean mask where valid points exist (filtered by confidence if threshold > 0)
        """
        N = self.get_num_frames()
        if N == 0:
            raise ValueError("No frames found")

        point_list = self.load_points_list()
        use_point_cloud = (point_list is not None and all(p is not None for p in point_list))

        rgb_list = self.load_rgb_list()

        traj_array = self.load_trajectory()
        if traj_array is None:
            raise FileNotFoundError(f"Trajectory file not found in {self.artifact}")

        if not use_point_cloud:
            depth_list = self.load_depth_list()
            intrinsics_array = self.load_intrinsics()

            if depth_list is None:
                raise FileNotFoundError(f"Depth directory not found in {self.artifact}")
            if intrinsics_array is None:
                raise FileNotFoundError(f"Intrinsics file not found in {self.artifact}")
            if any(d is None for d in depth_list):
                raise ValueError("Some depth maps are missing")

        H, W = rgb_list[0].shape[:2]

        xyzrgb = np.zeros((N, H, W, 6), dtype=np.float32)
        mask = np.zeros((N, H, W), dtype=bool)

        if use_point_cloud:
            for i in range(N):
                points = point_list[i]
                rgb = rgb_list[i]

                if rgb.shape[:2] != (H, W):
                    raise ValueError(
                        f"RGB shape {rgb.shape[:2]} doesn't match expected {(H, W)}"
                    )

                if points.ndim == 3 and points.shape[:2] == (H, W):
                    xyz_grid = points[:, :, :3]
                elif points.ndim == 2 and points.shape[0] == H * W:
                    xyz_grid = points[:, :3].reshape(H, W, 3)
                else:
                    raise ValueError(
                        f"Point cloud shape {points.shape} doesn't match expected "
                        f"[{H}, {W}, 3/6] or [{H*W}, 3/6]"
                    )

                valid_mask = np.any(xyz_grid != 0, axis=-1)
                mask[i] = valid_mask

                xyzrgb[i, :, :, :3] = xyz_grid
                xyzrgb[i, :, :, 3:] = rgb.astype(np.float32) / 255.0
        else:
            u = np.arange(W, dtype=np.float32)
            v = np.arange(H, dtype=np.float32)
            uu, vv = np.meshgrid(u, v)

            for i in range(N):
                depth = depth_list[i]
                intr = intrinsics_array[i]

                # Skip frames with missing intrinsics or pose
                if np.any(np.isnan(intr)):
                    continue

                c2w = traj_array[i]
                if np.any(np.isnan(c2w)):
                    continue

                fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]

                z = depth
                x = (uu - cx) * z / fx
                y = (vv - cy) * z / fy
                points_cam = np.stack([x, y, z], axis=-1)

                rotation = c2w[:3, :3]
                translation = c2w[:3, 3]
                points_flat = points_cam.reshape(-1, 3)
                points_world = points_flat @ rotation.T + translation[np.newaxis, :]
                xyz_grid = points_world.reshape(H, W, 3)

                valid_mask = (depth > 1e-4) & np.isfinite(depth)
                mask[i] = valid_mask

                xyzrgb[i, :, :, :3] = xyz_grid
                xyzrgb[i, ~valid_mask, :3] = 0.0

                rgb = rgb_list[i]
                if rgb.shape[:2] != (H, W):
                    raise ValueError(
                        f"RGB shape {rgb.shape[:2]} doesn't match depth shape {(H, W)}"
                    )
                xyzrgb[i, :, :, 3:] = rgb.astype(np.float32) / 255.0
        
        # Apply global thresholding if confidence_threshold > 0.0
        if confidence_threshold > 0.0:
            conf_list = self.load_confidence_list()

            if conf_list is not None:
                all_conf = np.concatenate([conf[mask[i]] for i, conf in enumerate(conf_list) if conf is not None and np.any(mask[i])])
                if len(all_conf) > 0:
                    global_threshold_value = np.percentile(all_conf, confidence_threshold * 100)
                    for i in range(N):
                        if conf_list[i] is not None:
                            conf_mask = conf_list[i] >= global_threshold_value
                            mask[i] = mask[i] & conf_mask

        if use_aoi_mask:
            aoi_mask_dir_path = self._resolve_aoi_mask_dir()
            if aoi_mask_dir_path is not None:
                for i in range(N):
                    mask_file = aoi_mask_dir_path / f"{i:06d}.png"
                    if mask_file.exists():
                        aoi_u8 = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                        if aoi_u8.shape[0] != H or aoi_u8.shape[1] != W:
                            aoi_u8 = cv2.resize(aoi_u8, (W, H), interpolation=cv2.INTER_NEAREST)
                        mask[i] &= (aoi_u8 > 0)

        return xyzrgb, mask
