"""Octree-based multi-level point cloud using Kaolin SPC.

Replaces VoxelGridCUDA with a proper spatial hierarchy.  Each octree level
stores centroid coordinates (not grid-snapped centers) and mean colors,
enabling LOD selection based on viewing distance.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import kaolin
import numpy as np
import torch


# 8 child offsets for octree expansion (constant, created once)
_CHILD_OFFSETS = None


def _get_child_offsets():
    global _CHILD_OFFSETS
    if _CHILD_OFFSETS is None or _CHILD_OFFSETS.device.type != 'cuda':
        _CHILD_OFFSETS = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
             [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
            dtype=torch.long, device='cuda')
    return _CHILD_OFFSETS


class OctreeSPC:
    """Multi-level octree point cloud backed by Kaolin SPC.

    Stores per-level centroid xyz, mean rgb, and earliest frame index.
    The finest level (max_level) replaces the old VoxelGridCUDA output;
    coarser levels are used by lod_select() for distance-adaptive rendering.
    """

    def __init__(self, max_level: int = 10):
        if max_level > 21:
            raise ValueError(
                f"octree_level={max_level} exceeds int64 cell_id limit (max 21)")
        self.max_level = max_level
        self.spc = None
        # Per-level GPU tensors (indexed 0 .. max_level)
        self.level_xyz: List[Optional[torch.Tensor]] = []      # (N_lv, 3) float32
        self.level_rgb: List[Optional[torch.Tensor]] = []      # (N_lv, 3) float32
        self.level_frames: List[Optional[torch.Tensor]] = []   # (N_lv,) int32
        self.level_cell_ids: List[Optional[torch.Tensor]] = [] # (N_lv,) int64, sorted
        # World ↔ [-1,1] transform
        self.center: np.ndarray = np.zeros(3, dtype=np.float32)
        self.scale: float = 1.0
        # Finest level sorted by frame (Scene compatibility)
        self.sorted_xyz: Optional[np.ndarray] = None    # (M, 3) float32
        self.sorted_rgb: Optional[np.ndarray] = None    # (M, 3) float32
        self.sorted_frames: Optional[np.ndarray] = None  # (M,) int32

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, all_xyz: torch.Tensor, all_rgb: torch.Tensor,
              all_frames: torch.Tensor, log=None):
        """Build octree + centroid hierarchy from raw unprojected points.

        Args:
            all_xyz:    (N, 3) float32 CPU tensor, world coordinates.
            all_rgb:    (N, 3) float32 CPU tensor, values in [0, 1].
            all_frames: (N,)   int32   CPU tensor, frame indices.
            log:        Optional logging callback.
        """
        _log = log or (lambda m: None)

        # --- Move to GPU ---
        xyz_gpu = all_xyz.cuda().contiguous()
        rgb_gpu = all_rgb.cuda().contiguous()
        frames_gpu = all_frames.int().cuda().contiguous()

        N = len(xyz_gpu)
        _log(f"[octree] {N:,} raw points → building level-{self.max_level} octree")

        # --- Bounding box + normalise to [-1, 1] ---
        mins = xyz_gpu.min(dim=0).values
        maxs = xyz_gpu.max(dim=0).values
        center = (mins + maxs) / 2
        extent = (maxs - mins).max().item()
        self.scale = extent / 2 * 1.01          # small margin
        self.center = center.cpu().numpy()

        normalized = (xyz_gpu - center) / self.scale  # [-1, 1]

        # --- Build Kaolin SPC (validation only; capped at 15) ---
        _SPC_MAX = 15
        spc_level = min(self.max_level, _SPC_MAX)
        self.spc = kaolin.ops.conversions.pointcloud.unbatched_pointcloud_to_spc(
            normalized.contiguous(), level=spc_level)
        pyr = self.spc.pyramids[0]
        _log(f"[octree] SPC(L{spc_level}) — "
             + ", ".join(f"L{i}:{pyr[0,i].item()}"
                         for i in range(0, spc_level + 1, 2)))

        # --- Centroid hierarchy ---
        self._compute_centroids(xyz_gpu, rgb_gpu, frames_gpu, normalized, _log)

        # --- Sorted finest level for Scene ---
        self._sort_finest_level()
        _log(f"[octree] finest level: {len(self.sorted_xyz):,} cells")

        # Cleanup
        del xyz_gpu, rgb_gpu, frames_gpu, normalized
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_ptrs(self, num_frames: int) -> np.ndarray:
        """Frame-boundary pointer array (same semantics as VoxelGridCUDA)."""
        assert self.sorted_frames is not None
        return np.searchsorted(
            self.sorted_frames,
            np.arange(num_frames, dtype=np.int32),
            side='right').astype(np.int32)

    # ------------------------------------------------------------------
    # LOD selection (Phase 2)
    # ------------------------------------------------------------------

    def lod_select(self, w2c: np.ndarray,
                   fx: float, fy: float, cx: float, cy: float,
                   width: int, height: int,
                   near: float, far: float,
                   frame_idx: int,
                   target_pixels: float = 1.5,
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Select visible points via octree LOD traversal.

        Traverses the octree from root to leaves.  At each level, cells are
        classified as:
        - **emit**: projected size ≤ target_pixels (or at max level) → output
        - **refine**: projected size > target_pixels → expand to children
        - **skip**: outside frustum or not yet revealed (frame filter)

        Returns:
            (vis_xyz, vis_rgb): numpy float32 arrays ready for Open3D.
        """
        max_lv = self.max_level
        scene_extent = self.scale * 2
        focal = fy  # fx == fy for symmetric FOV

        R_cw = torch.from_numpy(w2c[:3, :3].astype(np.float32)).cuda()
        t_cw = torch.from_numpy(w2c[:3, 3].astype(np.float32)).cuda()
        child_offsets = _get_child_offsets()

        emit_xyz: List[torch.Tensor] = []
        emit_rgb: List[torch.Tensor] = []

        # Start with all level-0 cells
        active_idx = torch.arange(len(self.level_xyz[0]),
                                  dtype=torch.long, device='cuda')

        for lv in range(max_lv + 1):
            if len(active_idx) == 0:
                break

            cell_size = scene_extent / (2 ** lv)
            half_diag = cell_size * 0.866  # sqrt(3)/2

            # Gather active cells' data
            xyz = self.level_xyz[lv][active_idx]
            rgb = self.level_rgb[lv][active_idx]
            frames = self.level_frames[lv][active_idx]

            # Frame filter: skip cells not yet revealed
            visible = frames <= frame_idx

            # Transform to camera space
            cam_pts = (R_cw @ xyz.T + t_cw[:, None]).T
            z = cam_pts[:, 2]

            # Conservative frustum check
            depth_ok = (z > near - half_diag) & (z < far + half_diag)
            safe_z = z.clamp(min=near * 0.5)
            proj_x = cam_pts[:, 0] * fx / safe_z + cx
            proj_y = cam_pts[:, 1] * fy / safe_z + cy
            margin = half_diag * focal / safe_z
            screen_ok = ((proj_x > -margin) & (proj_x < width + margin) &
                         (proj_y > -margin) & (proj_y < height + margin))

            in_frustum = visible & depth_ok & screen_ok

            # Projected cell size (pixels)
            proj_px = cell_size * focal / safe_z

            # Emit: small enough or at finest level
            do_emit = in_frustum & ((proj_px <= target_pixels) | (lv == max_lv))
            # Refine: too large, need finer detail
            do_refine = in_frustum & (proj_px > target_pixels) & (lv < max_lv)

            if do_emit.any():
                emit_xyz.append(xyz[do_emit])
                emit_rgb.append(rgb[do_emit])

            # Expand refined cells to children at next level
            if lv < max_lv and do_refine.any():
                refine_global = active_idx[do_refine]
                refine_ids = self.level_cell_ids[lv][refine_global]

                # Recover integer coords from cell IDs
                psz = 2 ** lv if lv > 0 else 1
                r_z = refine_ids % psz
                r_y = (refine_ids // psz) % psz
                r_x = refine_ids // (psz * psz)
                coords = torch.stack([r_x, r_y, r_z], dim=1)  # (R, 3)

                # 8 children per cell
                child_coords = (coords[:, None, :] * 2 +
                                child_offsets[None, :, :]).reshape(-1, 3)

                # Child cell IDs at next level
                csz = 2 ** (lv + 1)
                child_ids = (child_coords[:, 0] * csz * csz +
                             child_coords[:, 1] * csz +
                             child_coords[:, 2])

                # Binary search in sorted level_cell_ids of next level
                next_ids = self.level_cell_ids[lv + 1]
                pos = torch.searchsorted(next_ids, child_ids)
                valid = pos < len(next_ids)
                pos_clamped = pos.clamp(max=len(next_ids) - 1)
                valid = valid & (next_ids[pos_clamped] == child_ids)

                active_idx = pos_clamped[valid]
            else:
                active_idx = active_idx[:0]  # empty, same dtype

        # Concat and move to CPU
        if emit_xyz:
            vis_xyz = torch.cat(emit_xyz).cpu().numpy()
            vis_rgb = torch.cat(emit_rgb).cpu().numpy()
        else:
            vis_xyz = np.empty((0, 3), dtype=np.float32)
            vis_rgb = np.empty((0, 3), dtype=np.float32)

        return vis_xyz, vis_rgb

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_centroids(self, xyz_gpu, rgb_gpu, frames_gpu, normalized, _log):
        """Compute centroid xyz / mean rgb / min frame at every level."""
        max_lv = self.max_level
        grid_size = 2 ** max_lv

        # --- Quantise to finest grid ---
        # normalized ∈ [-1, 1] → [0, grid_size-1]
        # Compute cell_id column-by-column to avoid a full (N,3) int64 copy
        cell_id = ((normalized[:, 0] + 1) * 0.5 * grid_size).long().clamp(0, grid_size - 1)
        cell_id = cell_id * grid_size + \
            ((normalized[:, 1] + 1) * 0.5 * grid_size).long().clamp(0, grid_size - 1)
        cell_id = cell_id * grid_size + \
            ((normalized[:, 2] + 1) * 0.5 * grid_size).long().clamp(0, grid_size - 1)
        del normalized  # free early

        # --- Unique cells + inverse mapping ---
        unique_cells, inverse = torch.unique(cell_id, return_inverse=True)
        del cell_id
        num_cells = len(unique_cells)

        # --- Accumulate at finest level ---
        sum_xyz = torch.zeros(num_cells, 3, device='cuda')
        sum_rgb = torch.zeros(num_cells, 3, device='cuda')
        counts = torch.zeros(num_cells, device='cuda')

        inv3 = inverse.unsqueeze(1).expand(-1, 3)
        sum_xyz.scatter_add_(0, inv3, xyz_gpu)
        sum_rgb.scatter_add_(0, inv3, rgb_gpu)
        counts.scatter_add_(0, inverse, torch.ones(len(inverse), device='cuda'))

        min_frames = torch.full((num_cells,), 2**30, dtype=torch.int32, device='cuda')
        min_frames.scatter_reduce_(0, inverse, frames_gpu,
                                   reduce='amin', include_self=True)
        del inv3, inverse, xyz_gpu, rgb_gpu, frames_gpu

        centroid_xyz = sum_xyz / counts.unsqueeze(1)
        centroid_rgb = sum_rgb / counts.unsqueeze(1)
        del sum_xyz, sum_rgb

        # --- Initialise per-level storage ---
        self.level_xyz = [None] * (max_lv + 1)
        self.level_rgb = [None] * (max_lv + 1)
        self.level_frames = [None] * (max_lv + 1)
        self.level_cell_ids = [None] * (max_lv + 1)

        self.level_xyz[max_lv] = centroid_xyz
        self.level_rgb[max_lv] = centroid_rgb
        self.level_frames[max_lv] = min_frames
        self.level_cell_ids[max_lv] = unique_cells  # sorted by torch.unique

        # --- Recover integer coords of finest cells ---
        iz = unique_cells % grid_size
        iy = (unique_cells // grid_size) % grid_size
        ix = unique_cells // (grid_size * grid_size)
        prev_coords = torch.stack([ix, iy, iz], dim=1)
        del ix, iy, iz

        prev_xyz = centroid_xyz
        prev_rgb = centroid_rgb
        prev_frames = min_frames
        prev_counts = counts

        # --- Bottom-up aggregation to coarser levels ---
        for lv in range(max_lv - 1, -1, -1):
            parent_coords = prev_coords >> 1
            psz = 2 ** lv if lv > 0 else 1
            parent_id = (parent_coords[:, 0] * psz * psz
                         + parent_coords[:, 1] * psz
                         + parent_coords[:, 2])

            unique_parents, inv = torch.unique(parent_id, return_inverse=True)
            n_par = len(unique_parents)

            # Weighted centroid: Σ(centroid_i × count_i) / Σ(count_i)
            w = prev_counts.unsqueeze(1)
            inv3 = inv.unsqueeze(1).expand(-1, 3)

            p_sum_xyz = torch.zeros(n_par, 3, device='cuda')
            p_sum_rgb = torch.zeros(n_par, 3, device='cuda')
            p_counts = torch.zeros(n_par, device='cuda')
            p_min_frames = torch.full((n_par,), 2**30, dtype=torch.int32,
                                      device='cuda')

            p_sum_xyz.scatter_add_(0, inv3, prev_xyz * w)
            p_sum_rgb.scatter_add_(0, inv3, prev_rgb * w)
            p_counts.scatter_add_(0, inv, prev_counts)
            p_min_frames.scatter_reduce_(0, inv, prev_frames,
                                         reduce='amin', include_self=True)

            self.level_xyz[lv] = p_sum_xyz / p_counts.unsqueeze(1)
            self.level_rgb[lv] = p_sum_rgb / p_counts.unsqueeze(1)
            self.level_frames[lv] = p_min_frames
            self.level_cell_ids[lv] = unique_parents  # sorted by torch.unique

            # Prepare next iteration
            if lv > 0:
                p_iz = unique_parents % psz
                p_iy = (unique_parents // psz) % psz
                p_ix = unique_parents // (psz * psz)
                prev_coords = torch.stack([p_ix, p_iy, p_iz], dim=1)
            prev_xyz = self.level_xyz[lv]
            prev_rgb = self.level_rgb[lv]
            prev_frames = self.level_frames[lv]
            prev_counts = p_counts

        _log(f"[octree] centroids: "
             + ", ".join(f"L{i}:{len(self.level_xyz[i])}"
                         for i in range(0, max_lv + 1, 2)))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _sort_finest_level(self):
        """Sort finest-level centroids by frame → CPU numpy for Scene."""
        lv = self.max_level
        sort_idx = torch.argsort(self.level_frames[lv].long(), stable=True)
        self.sorted_xyz = self.level_xyz[lv][sort_idx].cpu().numpy()
        self.sorted_rgb = self.level_rgb[lv][sort_idx].cpu().numpy()
        self.sorted_frames = self.level_frames[lv][sort_idx].cpu().numpy()
