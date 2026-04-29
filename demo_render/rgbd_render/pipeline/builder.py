"""SceneBuilder: chain-style scene construction from NPZ.

Usage:
    scene = (SceneBuilder(config)
             .load(input_npz)
             .preprocess()
             .voxelize()
             .build())
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from tqdm import tqdm

from ..camera import compute_scene_scale
from ..config import PipelineConfig
from ..data.loader import load_npz_data
from ..data.sky import load_or_create_sky_masks, _SKYSEG_SOFT_THRESHOLD
from ..geometry.unproject import unproject_depth_batch_gpu
from ..geometry.octree import OctreeSPC
from ..scene import Scene
from .gpu_mem import GpuMemoryManager


class SceneBuilder:
    """Builds a Scene from NPZ data, step by step.

    Each step stores intermediate state. Call .build() to produce the
    final immutable Scene.
    """

    def __init__(self, config: PipelineConfig, log=None):
        self.cfg = config
        self._log = log or (lambda msg: None)
        self._gpu_mem = GpuMemoryManager(config.gpu.memory_limit_gb)

        # Intermediate state (populated by load/preprocess/voxelize)
        self.images = None
        self.depths = None
        self.c2w = None
        self.Ks = None
        self.confs = None
        self.is_keyframe = None  # Optional (S,) bool mask from NPZ meta
        self.S = 0
        self.scene_scale = 0.0
        self.depth_range = None
        self.sky_masks = None
        self.octree = None

    def load(self, npz_path: str = '') -> 'SceneBuilder':
        """Load NPZ data."""
        path = npz_path or self.cfg.input
        if not path:
            raise ValueError("No input NPZ path specified")

        data = load_npz_data(path)
        self.images = data['images']
        self.depths = data['depth']
        self.c2w = data['c2w']
        self.Ks = data['K']
        self.confs = data['confidence']
        self.is_keyframe = data.get('is_keyframe')
        if self.is_keyframe is not None and len(self.is_keyframe) != len(self.images):
            self._log(f"[warn] is_keyframe length mismatch "
                      f"({len(self.is_keyframe)} vs {len(self.images)}), ignoring")
            self.is_keyframe = None
        self.S = len(self.images)

        skip = getattr(self.cfg, 'skip_first', 0) or 0
        if skip > 0:
            self.images = self.images[skip:]
            self.depths = self.depths[skip:]
            self.c2w = self.c2w[skip:]
            self.Ks = self.Ks[skip:]
            if self.confs is not None:
                self.confs = self.confs[skip:]
            if self.is_keyframe is not None:
                self.is_keyframe = self.is_keyframe[skip:]
            self.S = len(self.images)
            self._log(f"[skip_first] Dropped first {skip} frames -> {self.S} remaining")

        if self.cfg.fast_review > 0:
            self.S = min(self.cfg.fast_review, self.S)
            self.images = self.images[:self.S]
            self.depths = self.depths[:self.S]
            self.c2w = self.c2w[:self.S]
            self.Ks = self.Ks[:self.S]
            if self.confs is not None:
                self.confs = self.confs[:self.S]
            if self.is_keyframe is not None:
                self.is_keyframe = self.is_keyframe[:self.S]
            self._log(f"[fast_review] Using first {self.S} frames")

        stride = getattr(self.cfg, 'frame_stride', 1) or 1
        if stride > 1:
            self.images = self.images[::stride]
            self.depths = self.depths[::stride]
            self.c2w = self.c2w[::stride]
            self.Ks = self.Ks[::stride]
            if self.confs is not None:
                self.confs = self.confs[::stride]
            if self.is_keyframe is not None:
                self.is_keyframe = self.is_keyframe[::stride]
            self.S = len(self.images)
            self._log(f"[stride] Every {stride}-th frame -> {self.S} frames")

        self.scene_scale = compute_scene_scale(self.c2w)

        # Compute global depth range for depth visualization
        rc = self.cfg.render
        valid = self.depths[self.depths > 0]
        if len(valid) > 0:
            d_lo = float(np.percentile(valid, rc.depth_percentile_lo))
            d_hi = float(np.percentile(valid, rc.depth_percentile_hi))
            self.depth_range = (d_lo, d_hi)
            self._log(f"[depth] range = [{d_lo:.3f}, {d_hi:.3f}] m "
                      f"(p{rc.depth_percentile_lo}-p{rc.depth_percentile_hi})")

        self._log(f"[scene] scale = {self.scene_scale:.3f} m, {self.S} frames")
        return self

    def preprocess(self) -> 'SceneBuilder':
        """Apply sky masking (if configured)."""
        pp = self.cfg.preprocess
        if pp.mask_sky:
            self._log(f"[sky] Running sky segmentation (model={pp.sky_model})...")
            sky_mask_dir = getattr(pp, 'sky_mask_dir', None)
            sky_mask_visualization_dir = getattr(pp, 'sky_mask_visualization_dir', None)
            # images is (S, H, W, 3) uint8; load_or_create_sky_masks accepts (S, H, W, 3)
            soft_masks = load_or_create_sky_masks(
                images=self.images,
                skyseg_model_path=pp.sky_model,
                sky_mask_dir=sky_mask_dir,
                sky_mask_visualization_dir=sky_mask_visualization_dir,
                target_shape=self.images.shape[1:3],
                num_frames=self.S,
                batch_size=pp.sky_batch_size,
            )
            if soft_masks is not None:
                self.sky_masks = (soft_masks <= _SKYSEG_SOFT_THRESHOLD)
                self._log(f"[sky] Generated {self.sky_masks.shape[0]} sky masks")
            else:
                self._log("[sky] Sky segmentation skipped (no masks generated)")
        return self

    def voxelize(self) -> 'SceneBuilder':
        """Unproject depth and build octree."""
        sc = self.cfg.scene
        self._log(f"[octree] Unprojecting + building level-{sc.octree_level} octree...")
        t0 = time.time()

        all_xyz, all_rgb, all_frames = self._collect_points()

        self.octree = OctreeSPC(max_level=sc.octree_level)
        self.octree.build(all_xyz, all_rgb, all_frames, log=self._log)
        del all_xyz, all_rgb, all_frames

        self._log(f"[octree] {len(self.octree.sorted_xyz):,} cells, "
                  f"built in {time.time()-t0:.1f}s")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def build(self) -> Scene:
        """Produce the final immutable Scene."""
        if self.octree is None:
            raise RuntimeError("Must call .voxelize() before .build()")

        ptrs = self.octree.compute_ptrs(self.S)
        keep_depths = self.cfg.render.depth_video and self.depths is not None

        scene = Scene(
            sorted_xyz=self.octree.sorted_xyz,
            sorted_rgb=self.octree.sorted_rgb,
            sorted_frames=self.octree.sorted_frames,
            ptrs=ptrs,
            c2w_poses=self.c2w,
            scene_scale=self.scene_scale,
            images=self.images,
            intrinsics=self.Ks,
            depths=self.depths if keep_depths else None,
            depth_range=self.depth_range,
            octree=self.octree,
        )

        # Clear builder state
        self.octree = None
        self.depths = self.confs = self.sky_masks = None
        return scene

    # ------------------------------------------------------------------
    # GPU unproject → collect raw points
    # ------------------------------------------------------------------

    def _collect_points(self):
        """Unproject all frames and return (all_xyz, all_rgb, all_frames) on CPU."""
        sc = self.cfg.scene
        pp = self.cfg.preprocess
        H, W = self.depths[0].shape
        batch_size = self._gpu_mem.build_batch_size(
            H, W, sc.downsample, self.cfg.gpu.build_batch_size)
        self._log(f"[gpu] build batch_size={batch_size}")

        # Optional keyframe-only point filtering.  When enabled, non-keyframe
        # frames still go through the loop (their poses are needed for the
        # camera trajectory) but their depth is zeroed out so no points are
        # unprojected.
        kf_only = bool(getattr(sc, 'keyframes_only_points', False))
        if kf_only and self.is_keyframe is None:
            self._log("[warn] --keyframes_only_points requested but NPZ has no "
                      "is_keyframe / frame_type mask; falling back to all frames")
            kf_only = False
        elif kf_only:
            kept = int(self.is_keyframe.sum())
            self._log(f"[keyframes-only] unprojecting {kept}/{self.S} frames "
                      f"(non-keyframes kept in camera trajectory only)")

        xyz_parts, rgb_parts, frame_parts = [], [], []

        with tqdm(total=self.S, unit='frame', dynamic_ncols=True,
                  desc='Unprojecting') as pbar:
            for bs in range(0, self.S, batch_size):
                be = min(bs + batch_size, self.S)
                B = be - bs

                d_batch = torch.from_numpy(
                    self.depths[bs:be].astype(np.float32)).cuda()
                img_batch = torch.from_numpy(self.images[bs:be].copy()).cuda()
                Ks_batch = torch.from_numpy(
                    self.Ks[bs:be].astype(np.float32)).cuda()
                c2w_batch = torch.from_numpy(
                    self.c2w[bs:be].astype(np.float32)).cuda()

                # Apply sky mask
                if self.sky_masks is not None:
                    sky_batch = torch.from_numpy(
                        self.sky_masks[bs:be].copy()).cuda()
                    d_batch[sky_batch] = 0.0
                    del sky_batch

                # Drop non-keyframe depth if requested.  Done after sky mask so
                # the zeros also get picked up by downstream "no valid pixel"
                # short-circuits.
                if kf_only:
                    kf_slice = self.is_keyframe[bs:be]
                    for j in range(B):
                        if not kf_slice[j]:
                            d_batch[j] = 0.0

                # Apply confidence threshold
                if pp.conf_threshold > 0 and self.confs is not None:
                    c_batch = torch.from_numpy(
                        self.confs[bs:be].astype(np.float32)).cuda()
                    for j in range(B):
                        valid_j = d_batch[j] > 0
                        if valid_j.any():
                            thr = torch.quantile(
                                c_batch[j][valid_j],
                                pp.conf_threshold / 100.0)
                            d_batch[j][valid_j & (c_batch[j] < thr)] = 0.0
                    del c_batch

                # Apply visibility threshold
                if pp.vis_threshold > 0 and self.confs is not None:
                    conf_batch = torch.from_numpy(
                        self.confs[bs:be].astype(np.float32)).cuda()
                    d_batch[conf_batch < pp.vis_threshold] = 0.0
                    del conf_batch

                pts_xyz, pts_rgb, vcounts = unproject_depth_batch_gpu(
                    d_batch, img_batch, Ks_batch, c2w_batch,
                    sc.max_depth, sc.downsample, sc.jitter)
                del d_batch, img_batch, Ks_batch, c2w_batch

                offset = 0
                for j in range(B):
                    cnt = vcounts[j].item()
                    if cnt > 0:
                        xyz_parts.append(pts_xyz[offset:offset + cnt].cpu())
                        rgb_parts.append(pts_rgb[offset:offset + cnt].cpu())
                        frame_parts.append(
                            torch.full((cnt,), bs + j, dtype=torch.int32))
                    offset += cnt
                del pts_xyz, pts_rgb
                pbar.update(B)

        torch.cuda.empty_cache()
        return (torch.cat(xyz_parts), torch.cat(rgb_parts),
                torch.cat(frame_parts))
