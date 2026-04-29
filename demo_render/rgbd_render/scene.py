"""Scene: immutable point cloud data container.

The Scene is a pure data object. It does NOT know about cameras or
perform frustum culling — that's the renderer's job.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class Scene:
    """Immutable scene data: point cloud + octree + metadata.

    The Scene does NOT know about cameras or perform culling.
    Rendering goes through octree.lod_select().
    """

    def __init__(self, sorted_xyz, sorted_rgb, sorted_frames, ptrs,
                 c2w_poses, scene_scale,
                 images=None, intrinsics=None, depths=None, depth_range=None,
                 octree=None):
        # CPU arrays
        self.sorted_xyz = sorted_xyz          # (M, 3) float32
        self.sorted_rgb = sorted_rgb          # (M, 3) float32
        self.sorted_frames = sorted_frames    # (M,) int32
        self.ptrs = ptrs                      # (S,) int32
        self.c2w_poses = c2w_poses            # (S, 4, 4) float32
        self.scene_scale = scene_scale
        self.images = images                  # (S, H, W, 3) uint8, optional
        self.intrinsics = intrinsics          # (S, 3, 3) float32, optional
        self.depths = depths                  # (S, H, W) float, optional
        self.depth_range = depth_range        # (min, max) float, optional

        # Octree for LOD rendering
        self.octree = octree

    @property
    def num_frames(self) -> int:
        return len(self.ptrs)

    def active_points(self, frame_idx: int):
        """Return (xyz, rgb, frame_indices) up to frame_idx. No culling."""
        ptr = int(self.ptrs[frame_idx])
        frames = self.sorted_frames[:ptr] if self.sorted_frames is not None else None
        return self.sorted_xyz[:ptr], self.sorted_rgb[:ptr], frames

    def thumbnail_images(self, long_edge: int = 240) -> Optional[np.ndarray]:
        """Return images downsampled so the long edge = long_edge pixels."""
        if self.images is None:
            return None
        h0, w0 = self.images[0].shape[:2]
        if max(h0, w0) <= long_edge:
            return self.images
        if h0 >= w0:
            new_h = long_edge
            new_w = max(1, int(w0 * long_edge / h0))
        else:
            new_w = long_edge
            new_h = max(1, int(h0 * long_edge / w0))
        out = np.empty((len(self.images), new_h, new_w, 3), dtype=np.uint8)
        for i in range(len(self.images)):
            out[i] = cv2.resize(self.images[i], (new_w, new_h),
                                interpolation=cv2.INTER_AREA)
        return out

    def destroy(self):
        """Release octree references."""
        self.octree = None

