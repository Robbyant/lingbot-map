"""
Rotary Position Embeddings for MLX.

Two variants:
  RotaryEmbedding2D   — per-frame spatial RoPE (replaces RotaryPositionEmbedding2D)
  RotaryEmbedding3D   — temporal+spatial RoPE for camera head (replaces WanRotaryPosEmbed)
"""

from typing import Optional, Dict, Tuple
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _freq_components(dim: int, base: float = 100.0, scaling: float = 1.0,
                     max_seq: int = 512) -> Tuple[mx.array, mx.array]:
    """Pre-compute (cos, sin) frequency tables of shape [max_seq, dim].

    Kept as CPU computation to avoid any device-buffer-size issues.
    """
    exponents = np.arange(0, dim, 2, dtype=np.float32) / dim
    inv_freq = 1.0 / (base ** exponents) / scaling          # [dim//2]
    positions = np.arange(max_seq, dtype=np.float32)        # [max_seq]
    angles = np.outer(positions, inv_freq)                   # [max_seq, dim//2]
    angles = np.concatenate([angles, angles], axis=-1)       # [max_seq, dim]
    return mx.array(np.cos(angles)), mx.array(np.sin(angles))


def _lookup(positions: mx.array, table: mx.array) -> mx.array:
    """Embed integer positions using a pre-computed frequency table.

    positions: [B, T]  (integer grid indices)
    table:     [max_seq, dim]
    returns:   [B, 1, T, dim]  (ready to broadcast over heads)
    """
    # MLX take is equivalent to F.embedding
    emb = table[positions]   # [B, T, dim]
    return mx.expand_dims(emb, axis=1)  # [B, 1, T, dim]


# ---------------------------------------------------------------------------
# 2D spatial RoPE (used in frame-level blocks)
# ---------------------------------------------------------------------------

class RotaryEmbedding2D(nn.Module):
    """2D Rotary Position Embedding for patch grids.

    Mirrors RotaryPositionEmbedding2D from lingbot_map/layers/rope.py.
    Splits the head dimension into vertical and horizontal halves.
    """

    def __init__(self, base: float = 100.0, scaling: float = 1.0):
        super().__init__()
        self.base = base
        self.scaling = scaling
        # Tables are built lazily (depend on head_dim + max_position)
        self._cache: Dict[Tuple, Tuple[mx.array, mx.array]] = {}

    def _get_tables(self, head_dim: int, max_pos: int) -> Tuple[mx.array, mx.array]:
        key = (head_dim, max_pos)
        if key not in self._cache:
            cos_t, sin_t = _freq_components(
                head_dim, base=self.base, scaling=self.scaling,
                max_seq=max_pos + 1,
            )
            self._cache[key] = (cos_t, sin_t)
        return self._cache[key]

    def get_cos_sin(self, q: mx.array, positions: mx.array,
                    head_dim: Optional[int] = None) -> Tuple[mx.array, mx.array]:
        """Return (cos, sin) tensors shaped [B, 1, T, D] for the given spatial positions.

        positions: [B, T, 2]  (y, x integer coords)
        q:         [B, H, T, D]  — used to infer head_dim when head_dim is None
        head_dim:  explicit head dim (avoids needing a dummy q tensor)
        """
        if head_dim is None:
            head_dim = q.shape[-1] // 2   # each spatial axis gets head_dim // 2 dims
        else:
            head_dim = head_dim // 2      # convert full head_dim → per-axis dim
        max_pos = int(mx.max(positions).item()) + 1

        cos_t, sin_t = self._get_tables(head_dim, max_pos)

        pos_y = positions[..., 0]  # [B, T]
        pos_x = positions[..., 1]  # [B, T]

        cos_y = _lookup(pos_y, cos_t)   # [B, 1, T, head_dim]
        sin_y = _lookup(pos_y, sin_t)
        cos_x = _lookup(pos_x, cos_t)
        sin_x = _lookup(pos_x, sin_t)

        # Concatenate vertical and horizontal components
        cos = mx.concatenate([cos_y, cos_x], axis=-1)  # [B, 1, T, D]
        sin = mx.concatenate([sin_y, sin_x], axis=-1)
        return cos, sin


class PositionGetter2D:
    """Generates y,x grid positions for an H×W patch grid, with caching."""

    def __init__(self):
        self._cache: Dict[Tuple[int, int], mx.array] = {}

    def __call__(self, B: int, H: int, W: int) -> mx.array:
        """Returns positions of shape [B, H*W, 2]."""
        key = (H, W)
        if key not in self._cache:
            y = mx.arange(H)
            x = mx.arange(W)
            # cartesian product → [H*W, 2]
            yy = mx.repeat(mx.expand_dims(y, 1), W, axis=1).reshape(-1)
            xx = mx.tile(mx.expand_dims(x, 0), (H, 1)).reshape(-1)
            self._cache[key] = mx.stack([yy, xx], axis=1)   # [H*W, 2]
        base = self._cache[key]                              # [H*W, 2]
        return mx.repeat(mx.expand_dims(base, 0), B, axis=0)  # [B, H*W, 2]


# ---------------------------------------------------------------------------
# 3D temporal+spatial RoPE (used in camera head, disabled by default)
# ---------------------------------------------------------------------------

class RotaryEmbedding3D(nn.Module):
    """3D Rotary Position Embedding for streaming video tokens.

    Mirrors WanRotaryPosEmbed from lingbot_map/layers/rope.py.
    Allocates head_dim across (temporal, height, width) dimensions.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 1024,
                 theta: float = 10000.0,
                 fhw_dim: Optional[Tuple[int, int, int]] = None):
        super().__init__()
        if fhw_dim is not None:
            t_dim, h_dim, w_dim = fhw_dim
        else:
            h_dim = w_dim = 2 * (head_dim // 6)
            t_dim = head_dim - h_dim - w_dim
        self.fhw_dim = (t_dim, h_dim, w_dim)

        # Pre-compute frequency tables for each axis (on CPU, then convert)
        def make_freqs(d: int) -> mx.array:
            exps = np.arange(0, d, 2, dtype=np.float32) / d
            inv_freq = 1.0 / (theta ** exps)  # [d//2]
            pos = np.arange(max_seq_len, dtype=np.float32)
            return mx.array(np.outer(pos, inv_freq))  # [max_seq, d//2]

        self.freqs_t = make_freqs(t_dim)
        self.freqs_h = make_freqs(h_dim)
        self.freqs_w = make_freqs(w_dim)

    def __call__(self, ppf: int, pph: int, ppw: int, patch_start_idx: int,
                 f_start: int = 0, f_end: Optional[int] = None) -> mx.array:
        """Build 3D RoPE frequency tensor.

        Returns real-valued angles of shape [1, 1, T, head_dim//2]
        where T = ppf * (patch_start_idx + pph * ppw).
        """
        if f_end is not None:
            ppf = f_end - f_start
            frame_slice = slice(f_start, f_end)
        else:
            frame_slice = slice(0, ppf)

        ft, fh, fw = self.freqs_t, self.freqs_h, self.freqs_w

        if patch_start_idx > 0:
            # Special tokens: position (f, i, i) on the diagonal
            ff_s = ft[frame_slice].reshape(ppf, 1, -1).broadcast_to((ppf, patch_start_idx, ft.shape[-1]))
            fh_s = fh[:patch_start_idx].reshape(1, patch_start_idx, -1).broadcast_to((ppf, patch_start_idx, fh.shape[-1]))
            fw_s = fw[:patch_start_idx].reshape(1, patch_start_idx, -1).broadcast_to((ppf, patch_start_idx, fw.shape[-1]))
            freqs_special = mx.concatenate([ff_s, fh_s, fw_s], axis=-1)  # [ppf, N_sp, dim/2]

            # Patch tokens
            ff_p = ft[frame_slice].reshape(ppf, 1, 1, -1).broadcast_to((ppf, pph, ppw, ft.shape[-1]))
            fh_p = fh[patch_start_idx:patch_start_idx + pph].reshape(1, pph, 1, -1).broadcast_to((ppf, pph, ppw, fh.shape[-1]))
            fw_p = fw[patch_start_idx:patch_start_idx + ppw].reshape(1, 1, ppw, -1).broadcast_to((ppf, pph, ppw, fw.shape[-1]))
            freqs_patches = mx.concatenate([ff_p, fh_p, fw_p], axis=-1).reshape(ppf, pph * ppw, -1)

            freqs = mx.concatenate([freqs_special, freqs_patches], axis=1)  # [ppf, N_sp+N_p, dim/2]
        else:
            ff_p = ft[frame_slice].reshape(ppf, 1, 1, -1).broadcast_to((ppf, pph, ppw, ft.shape[-1]))
            fh_p = fh[:pph].reshape(1, pph, 1, -1).broadcast_to((ppf, pph, ppw, fh.shape[-1]))
            fw_p = fw[:ppw].reshape(1, 1, ppw, -1).broadcast_to((ppf, pph, ppw, fw.shape[-1]))
            freqs = mx.concatenate([ff_p, fh_p, fw_p], axis=-1).reshape(ppf * pph * ppw, -1)

        total_tokens = freqs.shape[0] if patch_start_idx == 0 else ppf * (patch_start_idx + pph * ppw)
        return freqs.reshape(1, 1, total_tokens, -1)   # [1, 1, T, dim/2]
