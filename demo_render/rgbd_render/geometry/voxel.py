"""CUDA-accelerated voxel grid using Morton coding + GPU sort."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from render_cuda_ext import voxelize_frame as cuda_voxelize_frame


class VoxelGridCUDA:
    """CUDA-accelerated voxel grid using Morton coding + GPU sort."""

    def __init__(self, voxel_size: float, color_update: str = 'first',
                 batch_size: int = 30, defer_merge: bool = True):
        self.voxel_size = voxel_size
        self.color_update = color_update
        self.batch_size = batch_size
        self.defer_merge = defer_merge
        self._finalized = False
        self._batch_morton: List[torch.Tensor] = []
        self._batch_centers: List[torch.Tensor] = []
        self._batch_colors: List[torch.Tensor] = []
        self._batch_frames: List[torch.Tensor] = []
        self._global_morton = torch.empty(0, dtype=torch.int64, device='cuda')
        self._global_xyz = torch.empty((0, 3), dtype=torch.float32, device='cuda')
        self._global_rgb = torch.empty((0, 3), dtype=torch.float32, device='cuda')
        self._global_frames = torch.empty(0, dtype=torch.int32, device='cuda')

    def insert(self, pts_xyz: np.ndarray, pts_rgb: np.ndarray, frame_idx: int):
        assert not self._finalized
        if len(pts_xyz) == 0: return
        xyz_gpu = torch.from_numpy(pts_xyz.astype(np.float32)).cuda()
        rgb_gpu = torch.from_numpy(pts_rgb.astype(np.float32)).cuda()
        self._insert_tensors(*cuda_voxelize_frame(xyz_gpu, rgb_gpu, self.voxel_size), frame_idx)
        del xyz_gpu, rgb_gpu

    def insert_gpu(self, pts_xyz_gpu, pts_rgb_gpu, frame_idx: int):
        assert not self._finalized
        if len(pts_xyz_gpu) == 0: return
        self._insert_tensors(*cuda_voxelize_frame(pts_xyz_gpu, pts_rgb_gpu, self.voxel_size), frame_idx)

    def _insert_tensors(self, centers, colors, morton, frame_idx):
        self._batch_morton.append(morton)
        self._batch_centers.append(centers)
        self._batch_colors.append(colors)
        self._batch_frames.append(torch.full((len(morton),), frame_idx,
                                             dtype=torch.int32, device='cuda'))
        if not self.defer_merge and len(self._batch_morton) >= self.batch_size:
            self._merge_batch()

    def _merge_batch(self):
        if not self._batch_morton: return
        bm = torch.cat(self._batch_morton)
        bc = torch.cat(self._batch_centers)
        br = torch.cat(self._batch_colors)
        bf = torch.cat(self._batch_frames)
        cm = torch.cat([self._global_morton, bm])
        cc = torch.cat([self._global_xyz, bc])
        cr = torch.cat([self._global_rgb, br])
        cf = torch.cat([self._global_frames, bf])
        del bm, bc, br, bf
        sm, si = torch.sort(cm)
        sc, sr, sf = cc[si], cr[si], cf[si]
        del cm, cc, cr, cf
        if len(sm) > 0:
            mask = torch.cat([torch.tensor([True], device='cuda'), sm[1:] != sm[:-1]])
            if self.color_update == 'latest':
                rev = torch.arange(len(mask) - 1, -1, -1, device='cuda')
                rmask = torch.cat([torch.tensor([True], device='cuda'), sm[:-1] != sm[1:]])
                mask = rmask[rev][rev]
                del rev, rmask
            self._global_morton = sm[mask]
            self._global_xyz = sc[mask]
            self._global_rgb = sr[mask]
            self._global_frames = sf[mask]
        self._batch_morton.clear()
        self._batch_centers.clear()
        self._batch_colors.clear()
        self._batch_frames.clear()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def finalize(self):
        self._merge_batch()
        if len(self._global_morton) == 0:
            self.sorted_xyz = np.zeros((0, 3), dtype=np.float32)
            self.sorted_rgb = np.zeros((0, 3), dtype=np.float32)
            self.sorted_frames = np.zeros((0,), dtype=np.int32)
        else:
            idx = torch.argsort(self._global_frames, stable=True)
            self.sorted_xyz = self._global_xyz[idx].cpu().numpy()
            self.sorted_rgb = self._global_rgb[idx].cpu().numpy()
            self.sorted_frames = self._global_frames[idx].cpu().numpy()
            del idx
        del self._global_morton, self._global_xyz, self._global_rgb, self._global_frames
        self._global_morton = self._global_xyz = self._global_rgb = self._global_frames = None
        self._finalized = True
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def current_voxel_count(self) -> int:
        g = len(self._global_morton) if self._global_morton is not None else 0
        return g + sum(len(t) for t in self._batch_morton)

    def compute_ptrs(self, num_frames: int) -> np.ndarray:
        assert self._finalized
        return np.searchsorted(self.sorted_frames,
                               np.arange(num_frames, dtype=np.int32),
                               side='right').astype(np.int32)

    def save(self, path: str):
        assert self._finalized
        np.savez_compressed(path, sorted_xyz=self.sorted_xyz, sorted_rgb=self.sorted_rgb,
                            sorted_frames=self.sorted_frames, voxel_size=self.voxel_size,
                            color_update=self.color_update, impl='cuda')

    @staticmethod
    def load(path: str, batch_size: int = 20) -> 'VoxelGridCUDA':
        data = np.load(path)
        grid = VoxelGridCUDA(float(data['voxel_size']), str(data['color_update']), batch_size)
        grid.sorted_xyz = data['sorted_xyz']
        grid.sorted_rgb = data['sorted_rgb']
        grid.sorted_frames = data['sorted_frames']
        grid._finalized = True
        return grid


def create_voxel_grid(voxel_size, color_update='first', batch_size=20,
                      defer_merge=True) -> VoxelGridCUDA:
    return VoxelGridCUDA(voxel_size, color_update, batch_size, defer_merge)
