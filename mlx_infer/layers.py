"""
Core MLX layers: Linear, LayerNorm, MLP, Attention, Transformer Block.

MLX notes vs PyTorch:
- mx.array is the tensor type; created lazily (no data until evaluated)
- mlx.nn.Linear has weight shape [out, in] like PyTorch, but no transpose in matmul needed
- mlx.core.matmul broadcasts differently; prefer nn.Linear for weight contractions
- mx.fast.scaled_dot_product_attention is available and supports causal masking
"""

import math
from typing import Optional, List, Tuple, Dict, Any
import mlx.core as mx
import mlx.nn as nn
import numpy as np


class LayerNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.dims = dims
        self.eps = eps
        if affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.layer_norm(x, self.weight, self.bias, self.eps)


class Linear(nn.Linear):
    """Thin wrapper so we can load PyTorch weight dicts by name."""
    pass


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int,
                 bias: bool = True, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = mx.full((dim,), init_values)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


def rotate_half(x: mx.array) -> mx.array:
    """Rotate the last dimension by splitting and negating halves."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return mx.concatenate([-x2, x1], axis=-1)


def _rope_1d(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Standard 1D RoPE on a D-dim vector: x * cos + rotate_half(x) * sin."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return mx.concatenate([x1 * cos[..., :d] - x2 * sin[..., :d],
                            x2 * cos[..., d:] + x1 * sin[..., d:]], axis=-1)


def apply_rope_2d(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> Tuple[mx.array, mx.array]:
    """Apply 2D RoPE to q and k. cos/sin shape: [B, 1, T, D].

    The first D//2 dims get 1D RoPE with y-positions; the last D//2 dims get
    1D RoPE with x-positions, each applied independently (matching PyTorch's
    RotaryPositionEmbedding2D which processes vertical/horizontal halves separately).
    """
    h = q.shape[-1] // 2
    cos_y, cos_x = cos[..., :h], cos[..., h:]
    sin_y, sin_x = sin[..., :h], sin[..., h:]
    q_rot = mx.concatenate([_rope_1d(q[..., :h], cos_y, sin_y),
                             _rope_1d(q[..., h:], cos_x, sin_x)], axis=-1)
    k_rot = mx.concatenate([_rope_1d(k[..., :h], cos_y, sin_y),
                             _rope_1d(k[..., h:], cos_x, sin_x)], axis=-1)
    return q_rot, k_rot


def apply_rotary_emb(x: mx.array, freqs: mx.array) -> mx.array:
    """Apply real-valued rotary position embedding (matches rope.apply_rotary_emb).

    x:     [B, H, T, D]
    freqs: [1, 1, T, D//2]  (real angles)
    """
    # Split x into pairs and apply rotation
    x_f32 = x.astype(mx.float32)
    x_r = x_f32[..., 0::2]   # [B, H, T, D//2]
    x_i = x_f32[..., 1::2]
    cos = mx.cos(freqs)       # [1, 1, T, D//2]
    sin = mx.sin(freqs)
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos
    # Interleave back
    out = mx.stack([out_r, out_i], axis=-1)  # [B, H, T, D//2, 2]
    out = out.reshape(*x.shape[:-1], x.shape[-1])
    return out.astype(x.dtype)


class Attention(nn.Module):
    """Multi-head self-attention with optional 2D RoPE and QK-norm."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 proj_bias: bool = True, qk_norm: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.q_norm = LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def __call__(self, x: mx.array, rope_cos: Optional[mx.array] = None,
                 rope_sin: Optional[mx.array] = None) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(0, 2, 3, 1, 4)   # [B, 3, H, N, D]  -- mlx transpose
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each [B, H, N, D]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope_cos is not None:
            q, k = apply_rope_2d(q, k, rope_cos, rope_sin)

        # mlx SDPA: [B, H, N, D]
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, proj_bias: bool = True, ffn_bias: bool = True,
                 qk_norm: bool = False, init_values: Optional[float] = None):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              proj_bias=proj_bias, qk_norm=qk_norm)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()

    def __call__(self, x: mx.array,
                 rope_cos: Optional[mx.array] = None,
                 rope_sin: Optional[mx.array] = None) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x), rope_cos=rope_cos, rope_sin=rope_sin))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# KV-cache attention for streaming (causal) global blocks
# ---------------------------------------------------------------------------

class CausalAttentionMLX(nn.Module):
    """Self-attention with a simple list-based KV cache for streaming inference.

    The KV cache stores (k, v) pairs with shape [B, H, T_cached, D].
    Each new frame appends its K/V; eviction keeps scale_frames + sliding_window.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 proj_bias: bool = True, qk_norm: bool = False,
                 sliding_window: int = 64, scale_frames: int = 8,
                 keep_special: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sliding_window = sliding_window
        self.scale_frames = scale_frames
        self.keep_special = keep_special

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.q_norm = LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def __call__(self, x: mx.array,
                 kv_cache: Optional[Dict[str, Any]] = None,
                 num_frame_per_block: int = 1,
                 rope_freqs: Optional[mx.array] = None,
                 rope_cos: Optional[mx.array] = None,
                 rope_sin: Optional[mx.array] = None,
                 patch_start_idx: int = 6) -> mx.array:
        B, N, C = x.shape
        tokens_per_frame = N // num_frame_per_block

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(0, 2, 3, 1, 4)  # [B, 3, H, N, D]
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope_cos is not None:
            q, k = apply_rope_2d(q, k, rope_cos, rope_sin)
        elif rope_freqs is not None:
            q = apply_rotary_emb(q, rope_freqs)
            k = apply_rotary_emb(k, rope_freqs)

        if kv_cache is None:
            # Batch mode (Phase 1 scale frames) — simple causal-within-batch attention
            x_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            # Streaming mode: append new K/V then attend to full cache
            k_new = k.reshape(B, self.num_heads, num_frame_per_block,
                               tokens_per_frame, self.head_dim)
            v_new = v.reshape(B, self.num_heads, num_frame_per_block,
                               tokens_per_frame, self.head_dim)

            skip = kv_cache.get("_skip_append", False)

            if kv_cache.get("k") is None:
                if not skip:
                    kv_cache["k"] = k_new
                    kv_cache["v"] = v_new
                k_full = k_new.reshape(B, self.num_heads, -1, self.head_dim)
                v_full = v_new.reshape(B, self.num_heads, -1, self.head_dim)
            else:
                if not skip:
                    k_cat = mx.concatenate([kv_cache["k"], k_new], axis=2)
                    v_cat = mx.concatenate([kv_cache["v"], v_new], axis=2)
                    # Evict: keep scale_frames + last sliding_window frames
                    n_frames = k_cat.shape[2]
                    total_keep = self.scale_frames + self.sliding_window
                    if n_frames > total_keep:
                        # Preserve special tokens from evicted frames
                        if self.keep_special:
                            evict_k = k_cat[:, :, self.scale_frames:n_frames - self.sliding_window, :patch_start_idx, :]
                            evict_v = v_cat[:, :, self.scale_frames:n_frames - self.sliding_window, :patch_start_idx, :]
                            if kv_cache.get("k_special") is None:
                                kv_cache["k_special"] = evict_k
                                kv_cache["v_special"] = evict_v
                            else:
                                kv_cache["k_special"] = mx.concatenate([kv_cache["k_special"], evict_k], axis=2)
                                kv_cache["v_special"] = mx.concatenate([kv_cache["v_special"], evict_v], axis=2)
                        kv_cache["k"] = mx.concatenate([
                            k_cat[:, :, :self.scale_frames],
                            k_cat[:, :, -self.sliding_window:],
                        ], axis=2)
                        kv_cache["v"] = mx.concatenate([
                            v_cat[:, :, :self.scale_frames],
                            v_cat[:, :, -self.sliding_window:],
                        ], axis=2)
                    else:
                        kv_cache["k"] = k_cat
                        kv_cache["v"] = v_cat

                k_cached = kv_cache["k"]
                v_cached = kv_cache["v"]
                if skip:
                    # Non-keyframe: attend to cache + current but don't store
                    k_cached = mx.concatenate([k_cached, k_new], axis=2)
                    v_cached = mx.concatenate([v_cached, v_new], axis=2)

                k_full = k_cached.reshape(B, self.num_heads, -1, self.head_dim)
                v_full = v_cached.reshape(B, self.num_heads, -1, self.head_dim)

                # Prepend preserved special tokens
                if kv_cache.get("k_special") is not None:
                    ks = kv_cache["k_special"]
                    vs = kv_cache["v_special"]
                    sa, sb, sc, sd, se = ks.shape
                    ks = ks.reshape(sa, sb, sc * sd, se)
                    vs = vs.reshape(sa, sb, sc * sd, se)
                    k_full = mx.concatenate([ks, k_full], axis=2)
                    v_full = mx.concatenate([vs, v_full], axis=2)

            x_out = mx.fast.scaled_dot_product_attention(q, k_full, v_full, scale=self.scale)

        x_out = x_out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x_out)


class StreamingBlock(nn.Module):
    """Global (cross-frame) transformer block for streaming with KV cache."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, proj_bias: bool = True, ffn_bias: bool = True,
                 qk_norm: bool = False, init_values: Optional[float] = None,
                 sliding_window: int = 64, scale_frames: int = 8,
                 keep_special: bool = True):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = CausalAttentionMLX(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
            qk_norm=qk_norm, sliding_window=sliding_window,
            scale_frames=scale_frames, keep_special=keep_special,
        )
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()

    def __call__(self, x: mx.array,
                 kv_cache: Optional[Dict[str, Any]] = None,
                 num_frame_per_block: int = 1,
                 rope_freqs: Optional[mx.array] = None,
                 rope_cos: Optional[mx.array] = None,
                 rope_sin: Optional[mx.array] = None,
                 patch_start_idx: int = 6) -> mx.array:
        attn_out = self.attn(self.norm1(x), kv_cache=kv_cache,
                             num_frame_per_block=num_frame_per_block,
                             rope_freqs=rope_freqs, rope_cos=rope_cos, rope_sin=rope_sin,
                             patch_start_idx=patch_start_idx)
        x = x + self.ls1(attn_out)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
