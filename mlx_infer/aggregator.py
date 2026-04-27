"""
MLX Streaming Aggregator.

Mirrors AggregatorBase + AggregatorStream from lingbot_map/aggregator/.
Architecture:
  DINOv2 ViT-L backbone (ViTMLX)
  → frame blocks  (TransformerBlock + 2D RoPE, per-frame)
  → global blocks (StreamingBlock + KV cache, cross-frame causal)

Special tokens: camera [1] + register [4] + scale [1] = patch_start_idx = 6
Selected output indices (block groups): [4, 11, 17, 23]
Output list shape: [B, S, P, 2C] per element (frame + global concatenated).
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .layers import LayerNorm, TransformerBlock, StreamingBlock
from .rope import RotaryEmbedding2D, PositionGetter2D

# ImageNet normalisation constants (same as PyTorch side)
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slice_expand_and_flatten(token: mx.array, B: int, S: int,
                              first_num_frame: int = 1) -> mx.array:
    """
    Expand a [1, 2, N, C] special-token parameter to [B*S, N, C].

    The first variant (index 0) is used for the first `first_num_frame` frames;
    the second variant (index 1) is used for the remaining frames.
    """
    N, C = token.shape[2], token.shape[3]
    if first_num_frame > 1:
        t_first = mx.broadcast_to(token[:, :1], (B, first_num_frame, N, C))
        t_rest  = mx.broadcast_to(token[:, 1:], (B, S - first_num_frame, N, C))
    else:
        t_first = mx.broadcast_to(token[:, :1], (B, 1, N, C))
        t_rest  = mx.broadcast_to(token[:, 1:], (B, S - 1, N, C))
    return mx.concatenate([t_first, t_rest], axis=1).reshape(B * S, N, C)


# ---------------------------------------------------------------------------
# DINOv2 ViT-L backbone
# ---------------------------------------------------------------------------

class ViTMLX(nn.Module):
    """DINOv2 ViT-L patch-embedding backbone (channel-last).

    Accepts images in NHWC format [B, H, W, 3].
    Returns x_norm_patchtokens: [B, N_patch, embed_dim].

    Weight loading note: PyTorch stores the patch projection weight as
    Conv2d [O, I, kH, kW]; MLX expects [O, kH, kW, I].  The weights.py
    conversion handles this transposition automatically.
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        init_values: float = 1.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        num_patches = (img_size // patch_size) ** 2

        # Patch projection (channel-last Conv2d)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable tokens and positional embedding
        # pos_embed covers CLS slot + all patch positions (registers have no pos_embed)
        self.cls_token      = mx.zeros((1, 1, embed_dim))
        self.register_tokens = mx.zeros((1, num_register_tokens, embed_dim))
        self.pos_embed      = mx.zeros((1, num_patches + 1, embed_dim))  # +1 for CLS

        # Transformer blocks (standard ViT, no RoPE — uses absolute pos_embed)
        # DINOv2 ViT-L uses qk_norm=False in its backbone blocks (unlike the
        # aggregator's own frame/global blocks which do use qk_norm=True).
        self.blocks = [
            TransformerBlock(
                embed_dim, num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                qk_norm=False,
                init_values=init_values,
            )
            for _ in range(depth)
        ]

        self.norm = LayerNorm(embed_dim)

    def _interpolate_pos_embed(self, h: int, w: int) -> mx.array:
        """Interpolate patch positional embeddings to (h, w) patch grid.

        Matches DINOv2's interpolate_pos_encoding: bicubic resize of the MxM
        learnt grid to h×w using scipy (CPU, cached per shape).
        """
        M = int(round(np.sqrt(self.pos_embed.shape[1] - 1)))
        if h == M and w == M:
            return self.pos_embed
        key = (h, w)
        if not hasattr(self, "_pos_embed_cache"):
            self._pos_embed_cache = {}
        if key not in self._pos_embed_cache:
            from scipy.ndimage import zoom
            patch_pe = np.array(self.pos_embed[0, 1:])      # [M*M, C]
            patch_pe = patch_pe.reshape(M, M, -1)            # [M, M, C]
            scale_h = h / M
            scale_w = w / M
            patch_pe = zoom(patch_pe, (scale_h, scale_w, 1), order=3)  # bicubic
            patch_pe = patch_pe.reshape(1, h * w, -1)        # [1, h*w, C]
            cls_pe = np.array(self.pos_embed[0:1, :1])       # [1, 1, C]
            full_pe = np.concatenate([cls_pe, patch_pe], axis=1)
            self._pos_embed_cache[key] = mx.array(full_pe.astype(np.float32))
        return self._pos_embed_cache[key]

    def __call__(self, x: mx.array) -> mx.array:
        """x: [B, H, W, 3] → patch tokens [B, N_patch, embed_dim]."""
        B = x.shape[0]
        C = x.shape[-1]  # 3

        # Patch embed: Conv2d [B, H, W, 3] → [B, H', W', embed_dim] → [B, N, embed_dim]
        tokens = self.patch_embed(x)
        h, w = tokens.shape[1], tokens.shape[2]
        tokens = tokens.reshape(B, h * w, -1)                  # [B, N_patch, C]

        # Add patch positional embeddings (interpolating if needed for non-square input)
        pos_embed = self._interpolate_pos_embed(h, w)
        tokens = tokens + pos_embed[:, 1:]

        # Prepend CLS token (with its positional embedding)
        cls = mx.broadcast_to(self.cls_token, (B, 1, tokens.shape[-1]))
        cls = cls + pos_embed[:, :1]

        # Insert register tokens (no positional embedding)
        regs = mx.broadcast_to(self.register_tokens, (B, self.num_register_tokens, tokens.shape[-1]))

        # Layout: [CLS, registers, patches]
        tokens = mx.concatenate([cls, regs, tokens], axis=1)   # [B, 1+R+N, C]

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # Return only patch tokens (skip CLS + register prefix)
        return tokens[:, 1 + self.num_register_tokens:]        # [B, N_patch, C]


# ---------------------------------------------------------------------------
# Streaming Aggregator
# ---------------------------------------------------------------------------

class AggregatorMLX(nn.Module):
    """Streaming causal aggregator for GCTStream (MLX version).

    Architecture mirrors AggregatorStream (use_sdpa backend):
      - ViTMLX backbone → patch tokens
      - Special tokens (camera + register + scale) prepended
      - aa_block_num groups of [frame_block, global_block]
      - Global blocks share a per-block KV cache for causal streaming

    Parameters
    ----------
    img_size, patch_size, embed_dim, depth, num_heads:
        Match the PyTorch checkpoint (ViT-L: 518, 14, 1024, 24, 16).
    aa_block_size:
        Number of frame/global blocks per alternating-attention group (default 1).
    num_register_tokens:
        DINOv2 register tokens (default 4).
    kv_cache_sliding_window, kv_cache_scale_frames:
        KV cache eviction policy (match PyTorch defaults: 64, 8).
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        aa_block_size: int = 1,
        rope_freq: float = 100.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        init_values: float = 0.01,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_keep_special: bool = True,
    ):
        super().__init__()
        assert depth % aa_block_size == 0
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.aa_block_size = aa_block_size
        self.aa_block_num = depth // aa_block_size
        self.num_register_tokens = num_register_tokens

        # Image normalisation buffers (not trainable)
        self._mean = mx.array(_RESNET_MEAN, dtype=mx.float32).reshape(1, 1, 1, 3)
        self._std  = mx.array(_RESNET_STD,  dtype=mx.float32).reshape(1, 1, 1, 3)

        # DINOv2 ViT-L backbone
        self.patch_embed = ViTMLX(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_register_tokens=num_register_tokens,
        )

        # 2D RoPE (used in both frame and global blocks)
        self.rope = RotaryEmbedding2D(base=rope_freq)
        self.position_getter = PositionGetter2D()

        block_kw = dict(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
            qk_norm=qk_norm, init_values=init_values,
        )

        # Frame blocks: standard TransformerBlock (per-frame, no KV cache)
        self.frame_blocks = [TransformerBlock(**block_kw) for _ in range(depth)]

        # Global blocks: StreamingBlock with causal KV cache
        self.global_blocks = [
            StreamingBlock(
                **block_kw,
                sliding_window=kv_cache_sliding_window,
                scale_frames=kv_cache_scale_frames,
                keep_special=kv_cache_keep_special,
            )
            for _ in range(depth)
        ]

        # Special tokens: camera [1, 2, 1, C], register [1, 2, R, C], scale [1, 2, 1, C]
        # shape [1, 2, N, C]: dim-1 indexes first-frame vs rest-of-frames variant
        self.camera_token   = mx.zeros((1, 2, 1, embed_dim))
        self.register_token = mx.zeros((1, 2, num_register_tokens, embed_dim))
        self.scale_token    = mx.ones((1, 2, 1, embed_dim))

        # patch_start_idx: camera(1) + register(num_register_tokens) + scale(1)
        self.patch_start_idx = 1 + num_register_tokens + 1
        self.num_special_tokens = self.patch_start_idx

        # KV cache: one dict per global block, initialised lazily
        self._kv_cache: Optional[List[Dict[str, Any]]] = None
        self.total_frames_processed = 0

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def _init_kv_cache(self):
        """Create fresh per-block KV cache dicts."""
        self._kv_cache = [
            {"k": None, "v": None, "k_special": None, "v_special": None,
             "_skip_append": False}
            for _ in range(self.depth)
        ]
        self.total_frames_processed = 0

    def clean_kv_cache(self):
        self._init_kv_cache()

    def set_skip_append(self, skip: bool):
        if self._kv_cache is not None:
            for d in self._kv_cache:
                d["_skip_append"] = skip

    # ------------------------------------------------------------------
    # Position embeddings
    # ------------------------------------------------------------------

    def _get_positions(self, B: int, S: int, H: int, W: int) -> mx.array:
        """2D patch positions [B*S, P, 2] with offset=1 for special tokens."""
        pph = H // self.patch_size
        ppw = W // self.patch_size
        pos = self.position_getter(B * S, pph, ppw)  # [B*S, N_patch, 2]
        pos = pos + 1                                 # patches start at position index 1
        # Special tokens sit at position (0, 0)
        pos_special = mx.zeros((B * S, self.num_special_tokens, 2), dtype=mx.int32)
        return mx.concatenate([pos_special, pos], axis=1)  # [B*S, P, 2]

    # ------------------------------------------------------------------
    # Special token preparation
    # ------------------------------------------------------------------

    def _prepare_special_tokens(
        self, B: int, S: int, C: int, scale_frames: int,
        in_streaming: bool = False, S_cached: int = 0,
    ) -> mx.array:
        """
        Build [B*S, num_special_tokens, C] special token tensor.

        In streaming mode (S_cached > 0) we expand to the full historical
        length and slice the last S rows to match PyTorch behaviour.
        """
        S_true = S_cached + S if in_streaming else S
        eff_scale = min(scale_frames, S_true)

        if in_streaming and S_cached > 0:
            cam   = slice_expand_and_flatten(self.camera_token,   B, S_true)[-S:]
            reg   = slice_expand_and_flatten(self.register_token, B, S_true)[-S:]
            scale = slice_expand_and_flatten(self.scale_token, B, S_true,
                                              first_num_frame=eff_scale)[-S:]
        else:
            cam   = slice_expand_and_flatten(self.camera_token,   B, S)
            reg   = slice_expand_and_flatten(self.register_token, B, S)
            scale = slice_expand_and_flatten(self.scale_token, B, S,
                                              first_num_frame=eff_scale)

        return mx.concatenate([cam, reg, scale], axis=1)  # [B*S, N_sp, C]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        images: mx.array,
        selected_idx: Optional[List[int]] = None,
        num_frame_for_scale: Optional[int] = None,
        num_frame_per_block: int = 1,
    ) -> Tuple[List[mx.array], int]:
        """
        Parameters
        ----------
        images: [B, S, 3, H, W] in [0, 1] range (PyTorch channel-first convention).
        selected_idx: block group indices to include in the output list (None = all).
        num_frame_for_scale: frames treated as scale frames (affects scale_token).
        num_frame_per_block: frames processed as one block (1 for streaming).

        Returns
        -------
        (output_list, patch_start_idx)
        output_list: List of [B, S, P, 2*embed_dim] tensors.
        """
        B, S, _, H, W = images.shape
        scale_frames = num_frame_for_scale if num_frame_for_scale is not None else 1

        # Determine streaming state
        in_streaming = (self._kv_cache is not None and
                        self._kv_cache[0]["k"] is not None)
        S_cached = 0
        if in_streaming:
            cached_k = self._kv_cache[0]["k"]  # [B, H, n_frames, T, D]
            S_cached = cached_k.shape[2] if cached_k is not None else 0

        # ---- Normalise images ----
        # images: [B, S, 3, H, W] (channel-first)
        # transpose to NHWC: [B*S, H, W, 3]
        imgs = images.reshape(B * S, 3, H, W)
        imgs = imgs.transpose(0, 2, 3, 1)                   # [B*S, H, W, 3]
        mean = mx.array(_RESNET_MEAN, dtype=imgs.dtype).reshape(1, 1, 1, 3)
        std  = mx.array(_RESNET_STD,  dtype=imgs.dtype).reshape(1, 1, 1, 3)
        imgs = (imgs - mean) / std

        # ---- DINOv2 patch embedding ----
        patch_tokens = self.patch_embed(imgs)                # [B*S, N_patch, C]
        C = patch_tokens.shape[-1]

        # ---- Special tokens ----
        special = self._prepare_special_tokens(
            B, S, C, scale_frames,
            in_streaming=in_streaming, S_cached=S_cached,
        )                                                    # [B*S, N_sp, C]
        tokens = mx.concatenate([special, patch_tokens], axis=1)  # [B*S, P, C]
        P = tokens.shape[1]

        # ---- 2D RoPE positions ----
        pos = self._get_positions(B, S, H, W)                # [B*S, P, 2]
        head_dim = self.embed_dim // self.num_heads          # 64 for ViT-L
        cos, sin = self.rope.get_cos_sin(
            None, pos, head_dim=head_dim)                    # [B*S, 1, P, 64]

        # ---- Alternating frame / global attention ----
        output_list: List[mx.array] = []
        frame_idx = 0
        global_idx = 0

        # Reshape for global blocks: [B, S*P, C]
        tokens_global = tokens.reshape(B, S * P, C)
        # Reshape cos/sin for global attention: [B, 1, S*P, D]
        cos_g = cos.reshape(B, 1, S * P, cos.shape[-1])
        sin_g = sin.reshape(B, 1, S * P, sin.shape[-1])

        for group in range(self.aa_block_num):
            frame_outs = []
            global_outs = []

            # -- Frame attention (aa_block_size blocks, per-frame) --
            for _ in range(self.aa_block_size):
                # tokens as [B*S, P, C]; cos/sin [B*S, 1, P, D]
                tokens_flat = tokens_global.reshape(B * S, P, C)
                tokens_flat = self.frame_blocks[frame_idx](
                    tokens_flat, rope_cos=cos, rope_sin=sin)
                tokens_global = tokens_flat.reshape(B, S * P, C)
                frame_outs.append(tokens_global.reshape(B, S, P, C))
                frame_idx += 1

            # -- Global (causal cross-frame) attention --
            for _ in range(self.aa_block_size):
                kv = self._kv_cache[global_idx] if self._kv_cache is not None else None
                tokens_global = self.global_blocks[global_idx](
                    tokens_global,
                    kv_cache=kv,
                    num_frame_per_block=num_frame_per_block,
                    rope_cos=cos_g,
                    rope_sin=sin_g,
                    patch_start_idx=self.patch_start_idx,
                )
                global_outs.append(tokens_global.reshape(B, S, P, C))
                global_idx += 1

            # Collect output for this group
            if selected_idx is None or group in selected_idx:
                for fi, gi in zip(frame_outs, global_outs):
                    output_list.append(mx.concatenate([fi, gi], axis=-1))  # [B, S, P, 2C]

        # Update frame counter (only on keyframe path, skip_append=False)
        if self._kv_cache is not None and not self._kv_cache[0].get("_skip_append", False):
            self.total_frames_processed += S

        return output_list, self.patch_start_idx
