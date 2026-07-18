"""GPU memory manager: auto-calculate batch sizes based on available VRAM.

Usage:
    mgr = GpuMemoryManager(memory_limit_gb=0)  # 0 = auto 85%
    batch = mgr.build_batch_size(H, W, downsample)
"""

from __future__ import annotations

import torch


class GpuMemoryManager:
    """Auto-calculate batch sizes to stay within GPU memory budget."""

    def __init__(self, memory_limit_gb: float = 0):
        total = torch.cuda.get_device_properties(0).total_memory
        if memory_limit_gb > 0:
            self.limit_bytes = int(memory_limit_gb * 1e9)
        else:
            self.limit_bytes = int(total * 0.85)

    def _free_bytes(self) -> int:
        allocated = torch.cuda.memory_allocated()
        return max(self.limit_bytes - allocated, 0)

    def build_batch_size(self, H: int, W: int, downsample: int,
                         override: int = 0) -> int:
        """Calculate how many frames to unproject in one GPU batch.

        Args:
            H, W: frame dimensions
            downsample: spatial downsample factor
            override: if > 0, use this value directly

        Returns:
            batch size (at least 1)
        """
        if override > 0:
            return override

        # Per-frame GPU memory estimate:
        #   depth (H*W*4) + rgb (H*W*3) + K (9*4) + c2w (16*4) +
        #   pts_cam (Hd*Wd*3*4) + pts_world (Hd*Wd*3*4) + output
        Hd = H // downsample
        Wd = W // downsample
        per_frame = (H * W * 4  # depth float32
                     + H * W * 3  # rgb uint8
                     + H * W * 4  # sky mask (worst case)
                     + Hd * Wd * 3 * 4 * 3  # pts_cam + pts_world + margin
                     + 1024)  # intrinsics etc.

        free = self._free_bytes()
        # Reserve 500MB for voxel grid operations
        usable = max(free - 500 * 1024 * 1024, per_frame)
        batch = max(1, int(usable // per_frame))
        return min(batch, 256)  # cap at 256
