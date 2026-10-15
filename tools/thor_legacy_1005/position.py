"""FP32 position arithmetic used by both arms of the historical benchmark."""
import types

import torch

from lingbot_map.heads.utils import create_uv_grid
from lingbot_map.layers.rope import get_1d_rotary_pos_embed


def sincos(embed_dim, pos, omega_0=100):
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega
    out = torch.einsum("m,d->md", pos.reshape(-1).to(omega.dtype), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def apply_pos_embed(self, x, W, H, ratio=0.1):
    patch_h, patch_w = x.shape[-2:]
    grid = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
    h, w, _ = grid.shape
    pos = grid.reshape(-1, 2)
    embedding = torch.cat([
        sincos(x.shape[1] // 2, pos[:, 0]), sincos(x.shape[1] // 2, pos[:, 1]),
    ], dim=-1).view(h, w, x.shape[1])
    embedding = (embedding * ratio).permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
    return x + embedding


def install_historical_positions(model):
    rope = model.aggregator.rope3d
    rope.freqs = torch.cat([
        get_1d_rotary_pos_embed(dim, rope.max_seq_len, 10000.0, use_real=False,
                               repeat_interleave_real=False, freqs_dtype=torch.float32)
        for dim in rope.fhw_dim
    ], dim=1)
    model.depth_head._apply_pos_embed = types.MethodType(apply_pos_embed, model.depth_head)
