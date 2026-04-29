"""Depth unprojection to world-space points (GPU batch)."""

from __future__ import annotations

import torch


def unproject_depth_batch_gpu(depth_batch, rgb_batch, K_batch, c2w_batch,
                              max_depth, downsample, add_jitter):
    """Batch GPU unproject. Returns (pts_xyz, pts_rgb, valid_counts) on CUDA."""
    B, H, W = depth_batch.shape
    d = depth_batch[:, ::downsample, ::downsample]
    r = rgb_batch[:, ::downsample, ::downsample]
    _, Hd, Wd = d.shape
    us = torch.arange(0, W, downsample, dtype=torch.float32, device='cuda')
    vs = torch.arange(0, H, downsample, dtype=torch.float32, device='cuda')
    ug = us[None, None, :].expand(B, Hd, Wd)
    vg = vs[None, :, None].expand(B, Hd, Wd)
    if add_jitter and downsample > 1:
        ug = ug + (torch.rand(B, Hd, Wd, device='cuda') - 0.5) * downsample * 0.8
        vg = vg + (torch.rand(B, Hd, Wd, device='cuda') - 0.5) * downsample * 0.8
    fx, fy = K_batch[:, 0, 0:1, None], K_batch[:, 1, 1:2, None]
    cx, cy = K_batch[:, 0, 2:3, None], K_batch[:, 1, 2:3, None]
    pts_cam = torch.stack([(ug - cx) * d / fx, (vg - cy) * d / fy, d], dim=-1).view(B, -1, 3)
    R, t = c2w_batch[:, :3, :3], c2w_batch[:, :3, 3]
    pts_world = torch.bmm(pts_cam, R.transpose(1, 2)) + t[:, None, :]
    valid = (d > 0) & (d < max_depth)
    valid_counts = valid.view(B, -1).sum(1).to(torch.int32)
    valid_flat = valid.view(-1)
    return (pts_world.view(-1, 3)[valid_flat],
            r.reshape(-1, 3)[valid_flat].float().div(255.0),
            valid_counts)
