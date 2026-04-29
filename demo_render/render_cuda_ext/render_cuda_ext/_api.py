import torch

import frustum_cull_ext as _frustum_cull_ext
import voxel_morton_ext as _voxel_morton_ext


def voxelize_frame(pts_xyz: torch.Tensor, pts_rgb: torch.Tensor, voxel_size: float):
    return _voxel_morton_ext.voxelize_frame(pts_xyz, pts_rgb, voxel_size)


def frustum_cull(
    pts: torch.Tensor,
    R_cw: torch.Tensor,
    t_cw: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    W: int,
    H: int,
    near_plane: float,
    far_plane: float,
):
    return _frustum_cull_ext.frustum_cull(
        pts,
        R_cw,
        t_cw,
        fx,
        fy,
        cx,
        cy,
        W,
        H,
        near_plane,
        far_plane,
    )
