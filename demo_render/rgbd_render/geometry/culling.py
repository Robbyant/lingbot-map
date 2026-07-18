"""GPU frustum culling."""

from __future__ import annotations

import torch

from render_cuda_ext import frustum_cull as cuda_frustum_cull


def frustum_cull_gpu(pts_t, R_t, t_t, fx, fy, cx, cy, W, H, near=0.1, far=100.0):
    """GPU frustum cull. Returns int64 index array (K,) on CPU."""
    visible = cuda_frustum_cull(pts_t, R_t, t_t, fx, fy, cx, cy, W, H, near, far)
    return torch.nonzero(visible, as_tuple=False).squeeze(1).cpu().numpy()
