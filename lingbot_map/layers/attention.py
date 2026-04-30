# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import math
import warnings
import torch

from torch import Tensor
from torch import nn
import torch.nn.functional as F

from lingbot_map.layers.rope import apply_rotary_emb

from einops import rearrange

# FlashInfer imports (optional - for paged attention)
try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    print("flashinfer not available")

from typing_extensions import List
from typing import Optional, Tuple


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, num_patches=None, num_special=None, num_frames=None, enable_3d_rope=False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CausalAttention(nn.Module):
    """
    Causal self-attention module with KV cache support for streaming inference.
    Used by CasualBlockCamera in camera_head.py.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        elementwise_attn_output_gate=False,
        # KV cache eviction parameters (matching build_attn_mask)
        kv_cache_sliding_window: int =64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,  # If True, only cache camera token (no scale token)
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        self.gate_proj = nn.Linear(dim, dim, bias=True) if elementwise_attn_output_gate else None

        # Store KV cache eviction parameters
        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.kv_cache_scale_frames = kv_cache_scale_frames
        self.kv_cache_cross_frame_special = kv_cache_cross_frame_special
        self.kv_cache_include_scale_frames = kv_cache_include_scale_frames
        self.kv_cache_camera_only = kv_cache_camera_only

    def forward(self, x: Tensor, block_mask=None, pos=None, pos_kv=None, frame_seqlen=None, video_mask=None, kv_cache=None, current_start=0, current_end=0, global_idx=0, num_frame_per_block=1, num_frame_for_scale=-1, enable_3d_rope=False, sliding_window_size=-1, attend_to_scale_frames=False, num_random_frames=0, attend_to_special_tokens=False, num_register_tokens=4, is_scale_frames=False) -> Tensor:
        B, N, C = x.shape

        # Calculate special token indices
        camera_token_idx = 0
        scale_token_idx = camera_token_idx + num_register_tokens + 1  # camera + register tokens + scale

        # [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.gate_proj is not None:
            gate_score = self.gate_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if kv_cache is None:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope is not None and not enable_3d_rope:
                q = self.rope(q, pos)
                k = self.rope(k, pos)
            elif enable_3d_rope and pos is not None:
                q = apply_rotary_emb(q, pos)
                k = apply_rotary_emb(k, pos)

            with torch.no_grad():
                block_mask = block_mask.squeeze()[:q.shape[2], :k.shape[2]]
                if block_mask.dim() == 2:
                    block_mask = block_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
                block_mask = block_mask.expand(B, 1, block_mask.shape[-2], block_mask.shape[-1])

                video_mask = video_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  if  video_mask is not None else torch.ones_like(block_mask, device=block_mask.device) # [1, 1, N, N]
                video_mask = video_mask.expand(B, 1, block_mask.shape[-2], block_mask.shape[-1])

                mask = block_mask | ~video_mask

                # Apply sliding window mask if sliding_window_size > 0
                # sliding_window_size is in units of num_frame_per_block
                if sliding_window_size > 0 and frame_seqlen is not None:
                    # Create sliding window mask: each frame can only attend to frames within the window
                    num_frames = N // frame_seqlen
                    sliding_mask = torch.zeros_like(mask, dtype=torch.bool)

                    for i in range(num_frames):
                        q_start = i * frame_seqlen
                        q_end = (i + 1) * frame_seqlen
                        # Calculate the window start: sliding_window_size is in units of num_frame_per_block
                        # So the actual window size in frames is sliding_window_size * num_frame_per_block
                        window_size_in_frames = sliding_window_size * num_frame_per_block
                        window_start_frame = max(0, i - window_size_in_frames + 1)
                        k_start = window_start_frame * frame_seqlen
                        k_end = (i + 1) * frame_seqlen  # Can attend up to current frame (causal)
                        sliding_mask[:, :, q_start:q_end, k_start:k_end] = True

                    # Combine with existing mask: both masks need to allow attention
                    mask = mask & sliding_mask

                    # If attend_to_scale_frames is True, also allow attention to first num_frame_for_scale frames
                    if num_frame_for_scale > 0:
                        for i in range(num_frames):
                            q_start = i * frame_seqlen
                            q_end = (i + 1) * frame_seqlen
                            # Allow attending to first num_frame_for_scale frames (directly set to True, not depending on block_mask)
                            mask[:, :, q_start:q_end, :num_frame_for_scale * frame_seqlen] = True

            ## global attention for the first num_frame_for_scale frames
            if num_frame_for_scale > 0:
                mask[:, :, :num_frame_for_scale * frame_seqlen, :num_frame_for_scale * frame_seqlen] = True

            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    attn_mask=mask
                )
        else:
            # Apply RoPE to current k before caching
            q, k = self.q_norm(q), self.k_norm(k)

            if self.rope is not None and not enable_3d_rope:
                q = self.rope(q, pos)
                k = self.rope(k, pos)
            elif enable_3d_rope and pos is not None:
                q = apply_rotary_emb(q, pos)
                k = apply_rotary_emb(k, pos)

            # Check if we should skip appending to cache (non-keyframe in keyframe mode)
            skip_append = kv_cache.get("_skip_append", False)

            k_reshaped = k.view(B, self.num_heads, num_frame_per_block, N // num_frame_per_block, self.head_dim)
            v_reshaped = v.view(B, self.num_heads, num_frame_per_block, N // num_frame_per_block, self.head_dim)

            if not skip_append:
                # KEYFRAME: store in cache (original behavior)
                if kv_cache[f"k_{global_idx}"] is None:
                    kv_cache[f"k_{global_idx}"] = k_reshaped
                    kv_cache[f"v_{global_idx}"] = v_reshaped
                else:
                    num_frame_per_block = k.shape[2] // kv_cache[f"k_{global_idx}"].shape[3]
                    k_reshaped = k.view(B, self.num_heads, num_frame_per_block, N // num_frame_per_block, self.head_dim)
                    v_reshaped = v.view(B, self.num_heads, num_frame_per_block, N // num_frame_per_block, self.head_dim)
                    kv_cache[f"k_{global_idx}"] = torch.cat((kv_cache[f"k_{global_idx}"], k_reshaped), dim=2)
                    kv_cache[f"v_{global_idx}"] = torch.cat((kv_cache[f"v_{global_idx}"], v_reshaped), dim=2)

                # Apply sliding window eviction BEFORE attention to match causal_3drope behavior
                # This ensures current frame only attends to frames within the sliding window
                self._apply_kv_cache_eviction_causal(kv_cache, global_idx, camera_token_idx, scale_token_idx)

                # Retrieve full k, v from cache (already RoPE-applied, already evicted)
                k = kv_cache[f"k_{global_idx}"].clone()
                v = kv_cache[f"v_{global_idx}"].clone()
            else:
                # NON-KEYFRAME: attend to [cached + current] without storing in cache
                if kv_cache[f"k_{global_idx}"] is not None:
                    k = torch.cat((kv_cache[f"k_{global_idx}"], k_reshaped), dim=2)
                    v = torch.cat((kv_cache[f"v_{global_idx}"], v_reshaped), dim=2)
                else:
                    k = k_reshaped
                    v = v_reshaped
            a, b, c, d, e = k.shape

            k = k.reshape(a, b, c*d, e)
            v = v.reshape(a, b, c*d, e)

            # Prepend special tokens (camera + scale) from evicted frames if they exist
            if f"k_{global_idx}_special" in kv_cache and kv_cache[f"k_{global_idx}_special"] is not None:
                special_k = kv_cache[f"k_{global_idx}_special"]  # [B, H, num_evicted_frames, 2, D]
                special_v = kv_cache[f"v_{global_idx}_special"]
                sa, sb, sc, sd, se = special_k.shape
                special_k = special_k.reshape(sa, sb, sc * sd, se)  # [B, H, num_evicted*2, D]
                special_v = special_v.reshape(sa, sb, sc * sd, se)

                # Prepend special tokens (older frames first)
                k = torch.cat([special_k, k], dim=2)
                v = torch.cat([special_v, v], dim=2)

            # Note: k from cache is already RoPE-applied, no need to apply again

            if self.fused_attn:
                # Use mask-based SDPA to ensure same kernel as batch mode
                # The causal constraint is enforced by KV cache contents, not by mask
                mask = torch.ones(B, 1, q.shape[2], k.shape[2], dtype=torch.bool, device=q.device)
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    attn_mask=mask,
                )

        if self.gate_proj is not None:
            x = x * torch.sigmoid(gate_score)
        # Use actual dimensions from attention output, not original input C
        # x shape: [B, H, seq_len, head_dim] -> [B, seq_len, H*head_dim]
        x = x.transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _apply_kv_cache_eviction_causal(self, kv_cache, global_idx, camera_token_idx, scale_token_idx):
        """
        Apply sliding window eviction to KV cache BEFORE attention.

        This ensures current frame only attends to frames within the sliding window,
        matching the behavior of causal_3drope's attention mask.
        """
        sliding_window_frames = self.kv_cache_sliding_window
        scale_frames = self.kv_cache_scale_frames

        if kv_cache[f"k_{global_idx}"].shape[3] > 1:
            num_cached_frames = kv_cache[f"k_{global_idx}"].shape[2]

            if num_cached_frames > sliding_window_frames + scale_frames:
                evict_start = scale_frames
                evict_end = num_cached_frames - sliding_window_frames

                if evict_end > evict_start:
                    evicted_k = kv_cache[f"k_{global_idx}"][:, :, evict_start:evict_end, :, :]
                    evicted_v = kv_cache[f"v_{global_idx}"][:, :, evict_start:evict_end, :, :]

                    if self.kv_cache_cross_frame_special:
                        if self.kv_cache_camera_only:
                            # Only keep camera token
                            new_special_k = evicted_k[:, :, :, camera_token_idx:camera_token_idx+1, :].clone()
                            new_special_v = evicted_v[:, :, :, camera_token_idx:camera_token_idx+1, :].clone()
                        else:
                            # Keep ALL special tokens (camera + register + scale) to match attention_mask behavior
                            # Special tokens are in range [camera_token_idx, scale_token_idx+1)
                            new_special_k = evicted_k[:, :, :, camera_token_idx:scale_token_idx+1, :].clone()
                            new_special_v = evicted_v[:, :, :, camera_token_idx:scale_token_idx+1, :].clone()

                        if f"k_{global_idx}_special" not in kv_cache or kv_cache[f"k_{global_idx}_special"] is None:
                            kv_cache[f"k_{global_idx}_special"] = new_special_k
                            kv_cache[f"v_{global_idx}_special"] = new_special_v
                        else:
                            kv_cache[f"k_{global_idx}_special"] = torch.cat(
                                [kv_cache[f"k_{global_idx}_special"], new_special_k], dim=2)
                            kv_cache[f"v_{global_idx}_special"] = torch.cat(
                                [kv_cache[f"v_{global_idx}_special"], new_special_v], dim=2)

                    if self.kv_cache_include_scale_frames:
                        kv_cache[f"k_{global_idx}"] = torch.cat([
                            kv_cache[f"k_{global_idx}"][:, :, :scale_frames, :, :],
                            kv_cache[f"k_{global_idx}"][:, :, -sliding_window_frames:, :, :]
                        ], dim=2)
                        kv_cache[f"v_{global_idx}"] = torch.cat([
                            kv_cache[f"v_{global_idx}"][:, :, :scale_frames, :, :],
                            kv_cache[f"v_{global_idx}"][:, :, -sliding_window_frames:, :, :]
                        ], dim=2)
                    else:
                        kv_cache[f"k_{global_idx}"] = kv_cache[f"k_{global_idx}"][:, :, -sliding_window_frames:, :, :]
                        kv_cache[f"v_{global_idx}"] = kv_cache[f"v_{global_idx}"][:, :, -sliding_window_frames:, :, :]


class FlashInferAttention(Attention):
    """
    FlashInfer variant of the GCT attention layer.
    Uses FlashInferKVCacheManager for paged KV cache storage and
    FlashInfer attention kernels (BatchPrefillWithPagedKVCacheWrapper).
    Supports the same optimized token layout and KV cache streaming inference.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        # KV cache eviction parameters
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
    ) -> None:
        if not FLASHINFER_AVAILABLE:
            raise RuntimeError("FlashInfer is not available. Please install flashinfer.")

        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        # Store KV cache eviction parameters
        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.kv_cache_scale_frames = kv_cache_scale_frames
        self.kv_cache_cross_frame_special = kv_cache_cross_frame_special
        self.kv_cache_include_scale_frames = kv_cache_include_scale_frames
        self.kv_cache_camera_only = kv_cache_camera_only

    def prepare_qkv(self, x: Tensor, pos=None, enable_3d_rope: bool = False) -> tuple:
        """Fused pre-attention ops for single-frame streaming (Phase 2).

        Computes q/k/v from x, applies q_norm/k_norm/RoPE, and converts to
        [tpf, H, D] format ready for append_frame + compute_attention.

        Extracted as a method so torch.compile can capture all pre-attn ops as one
        CUDA graph (qkv linear -> reshape -> unbind -> q_norm -> k_norm -> RoPE x2 ->
        squeeze/permute/contiguous x3).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: [B, num_heads, N, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and not enable_3d_rope:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        elif self.rope is not None:  # enable_3d_rope=True
            q = apply_rotary_emb(q, pos)
            k = apply_rotary_emb(k, pos)
        # Convert to [tpf, H, D] format for FlashInfer (B=1 in streaming mode)
        q_nhd = q.squeeze(0).permute(1, 0, 2).contiguous()
        k_nhd = k.squeeze(0).permute(1, 0, 2).contiguous()
        v_nhd = v.squeeze(0).permute(1, 0, 2).contiguous()
        return q_nhd, k_nhd, v_nhd

    def forward(self, x: Tensor, pos=None,
                num_patches=None, num_special=None, num_frames=None, enable_3d_rope=False,
                # KV cache parameters (kv_cache is a FlashInferKVCacheManager or None)
                kv_cache=None, global_idx=0, num_frame_per_block=1,
                num_frame_for_scale=-1, num_register_tokens=4) -> Tensor:
        """
        Forward pass with FlashInfer paged KV cache and attention.

        Args:
            x: Input tensor [B, N, C]
            kv_cache: FlashInferKVCacheManager instance or None (batch mode)
            global_idx: Block index for per-block cache access
        """
        from lingbot_map.layers.flashinfer_cache import FlashInferKVCacheManager

        B, N, C = x.shape

        # ========== Batch Mode (no KV cache manager) ==========
        if not isinstance(kv_cache, FlashInferKVCacheManager):
            # [3, B, num_heads, N, head_dim]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # Each: [B, num_heads, N, head_dim]
            q, k = self.q_norm(q), self.k_norm(k)

            if self.rope is not None and not enable_3d_rope:
                q = self.rope(q, pos)
                k = self.rope(k, pos)
            elif self.rope is not None and enable_3d_rope:
                q = apply_rotary_emb(q, pos)
                k = apply_rotary_emb(k, pos)

            # Batch mode: use SDPA for numerical consistency with SDPA variant
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

            x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)

        # ========== Streaming Mode (with FlashInferKVCacheManager) ==========
        else:
            manager = kv_cache  # FlashInferKVCacheManager

            # Phase 1 (scale frames): num_frames > 1 — multi-frame batch
            # Phase 2 (streaming):    num_frames == 1 — single frame
            is_multi_frame = (num_frames is not None and num_frames > 1)

            if is_multi_frame:
                # Phase 1: compute full self-attention via SDPA (all frames attend to each other),
                # then append each frame's K/V to the paged cache one at a time.
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q, k = self.q_norm(q), self.k_norm(k)

                # Apply RoPE before caching (RoPE baked into K before append)
                if self.rope is not None and not enable_3d_rope:
                    q = self.rope(q, pos)
                    k = self.rope(k, pos)
                elif self.rope is not None and enable_3d_rope:
                    q = apply_rotary_emb(q, pos)
                    k = apply_rotary_emb(k, pos)

                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
                x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)

                # Append each frame's K/V to the paged cache individually.
                tpf = manager.tokens_per_frame
                k_all = k.squeeze(0).permute(1, 0, 2)  # [num_frames*tpf, H, D]
                v_all = v.squeeze(0).permute(1, 0, 2)
                for f_idx in range(num_frames):
                    s = f_idx * tpf
                    manager.append_frame(global_idx, k_all[s:s+tpf].contiguous(), v_all[s:s+tpf].contiguous())
                    manager.evict_frames(
                        block_idx=global_idx,
                        scale_frames=self.kv_cache_scale_frames,
                        sliding_window=self.kv_cache_sliding_window,
                        cross_frame_special=self.kv_cache_cross_frame_special,
                        include_scale_frames=self.kv_cache_include_scale_frames,
                        camera_only=self.kv_cache_camera_only,
                        num_register_tokens=num_register_tokens,
                    )
            else:
                # Phase 2: single-frame streaming via FlashInfer paged attention.
                q_nhd, k_nhd, v_nhd = self.prepare_qkv(x, pos=pos, enable_3d_rope=enable_3d_rope)

                # Non-keyframe path: attend to cache+current but don't persist.
                # FlashInfer paged attention can only read from the cache, so we
                # temporarily append (with eviction deferred so it stays clean),
                # attend, and then roll back the append.  Mirrors the
                # ``skip_append`` behavior of the SDPA path.
                skip_append = getattr(manager, '_skip_append', False)
                if skip_append:
                    # Defensive fallback: the hot path for FlashInfer streaming
                    # is `FlashInferBlock.forward` which inlines this logic.
                    # Kept here for any caller that invokes
                    # ``FlashInferAttention.forward`` with a single frame.
                    prev_defer = manager._defer_eviction
                    manager._defer_eviction = True
                    manager.append_frame(global_idx, k_nhd, v_nhd)
                    x = manager.compute_attention(global_idx, q_nhd)
                    manager.rollback_last_frame(global_idx)
                    manager._defer_eviction = prev_defer
                else:
                    # 1. Append to paged cache
                    manager.append_frame(global_idx, k_nhd, v_nhd)

                    # 2. Apply sliding window eviction
                    manager.evict_frames(
                        block_idx=global_idx,
                        scale_frames=self.kv_cache_scale_frames,
                        sliding_window=self.kv_cache_sliding_window,
                        cross_frame_special=self.kv_cache_cross_frame_special,
                        include_scale_frames=self.kv_cache_include_scale_frames,
                        camera_only=self.kv_cache_camera_only,
                        num_register_tokens=num_register_tokens,
                    )

                    # 3. Compute attention via FlashInfer BatchPrefillWithPagedKVCacheWrapper
                    x = manager.compute_attention(global_idx, q_nhd)

                # Convert back: [tpf, H, D] -> [B, tpf, C].
                x = x.reshape(B, q_nhd.shape[0], self.num_heads * self.head_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SDPAAttention(Attention):
    """
    SDPA variant for streaming inference.
    Uses F.scaled_dot_product_attention with dict-based KV cache.
    No FlashInfer dependency required — works on any CUDA GPU.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
    ) -> None:
        super().__init__(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
            attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer,
            qk_norm=qk_norm, fused_attn=fused_attn, rope=rope,
        )
        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.kv_cache_scale_frames = kv_cache_scale_frames
        self.kv_cache_cross_frame_special = kv_cache_cross_frame_special
        self.kv_cache_include_scale_frames = kv_cache_include_scale_frames
        self.kv_cache_camera_only = kv_cache_camera_only

    def forward(self, x: Tensor, pos=None,
                num_patches=None, num_special=None, num_frames=None, enable_3d_rope=False,
                kv_cache=None, global_idx=0, num_frame_per_block=1,
                num_frame_for_scale=-1, num_register_tokens=4) -> Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # ========== Batch Mode (no KV cache) ==========
        if kv_cache is None:
            if self.rope is not None and not enable_3d_rope:
                q = self.rope(q, pos)
                k = self.rope(k, pos)
            elif self.rope is not None and enable_3d_rope:
                q = apply_rotary_emb(q, pos)
                k = apply_rotary_emb(k, pos)

            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)

        # ========== Streaming Mode (with KV cache dict) ==========
        else:
            if self.rope is not None and not enable_3d_rope:
                q = self.rope(q, pos)
                k = self.rope(k, pos)
            elif self.rope is not None and enable_3d_rope:
                q = apply_rotary_emb(q, pos)
                k = apply_rotary_emb(k, pos)

            camera_token_idx = 0
            scale_token_idx = camera_token_idx + num_register_tokens + 1

            if kv_cache[f"k_{global_idx}"] is None:
                kv_cache[f"k_{global_idx}"] = k.view(B, self.num_heads, num_frame_per_block,
                                                     N // num_frame_per_block, self.head_dim)
                kv_cache[f"v_{global_idx}"] = v.view(B, self.num_heads, num_frame_per_block,
                                                     N // num_frame_per_block, self.head_dim)
            else:
                num_frame_per_block = k.shape[2] // kv_cache[f"k_{global_idx}"].shape[3]
                kv_cache[f"k_{global_idx}"] = torch.cat((
                    kv_cache[f"k_{global_idx}"],
                    k.view(B, self.num_heads, num_frame_per_block, N // num_frame_per_block, self.head_dim)
                ), dim=2)
                kv_cache[f"v_{global_idx}"] = torch.cat((
                    kv_cache[f"v_{global_idx}"],
                    v.view(B, self.num_heads, num_frame_per_block, N // num_frame_per_block, self.head_dim)
                ), dim=2)

            self._apply_kv_cache_eviction(
                kv_cache, global_idx, camera_token_idx, scale_token_idx, num_register_tokens
            )

            k_cached = kv_cache[f"k_{global_idx}"].clone()
            v_cached = kv_cache[f"v_{global_idx}"].clone()
            a, b, c, d, e = k_cached.shape
            k_full = k_cached.reshape(a, b, c * d, e)
            v_full = v_cached.reshape(a, b, c * d, e)

            if f"k_{global_idx}_special" in kv_cache and kv_cache[f"k_{global_idx}_special"] is not None:
                special_k = kv_cache[f"k_{global_idx}_special"]
                special_v = kv_cache[f"v_{global_idx}_special"]
                sa, sb, sc, sd, se = special_k.shape
                k_full = torch.cat([special_k.reshape(sa, sb, sc * sd, se), k_full], dim=2)
                v_full = torch.cat([special_v.reshape(sa, sb, sc * sd, se), v_full], dim=2)

            q_seq_len = q.shape[2]
            x = F.scaled_dot_product_attention(
                q, k_full, v_full,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            x = x.transpose(1, 2).reshape(B, q_seq_len, self.num_heads * self.head_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _apply_kv_cache_eviction(self, kv_cache, global_idx, camera_token_idx, scale_token_idx, num_register_tokens):
        """Apply sliding window eviction to KV cache."""
        sliding_window_frames = self.kv_cache_sliding_window
        scale_frames = self.kv_cache_scale_frames

        if kv_cache[f"k_{global_idx}"].shape[3] > 1:
            num_cached_frames = kv_cache[f"k_{global_idx}"].shape[2]
            if num_cached_frames > sliding_window_frames + scale_frames:
                evict_start = scale_frames
                evict_end = num_cached_frames - sliding_window_frames
                if evict_end > evict_start:
                    evicted_k = kv_cache[f"k_{global_idx}"][:, :, evict_start:evict_end, :, :]
                    evicted_v = kv_cache[f"v_{global_idx}"][:, :, evict_start:evict_end, :, :]

                    if self.kv_cache_cross_frame_special:
                        if self.kv_cache_camera_only:
                            new_special_k = evicted_k[:, :, :, camera_token_idx:camera_token_idx+1, :].clone()
                            new_special_v = evicted_v[:, :, :, camera_token_idx:camera_token_idx+1, :].clone()
                        else:
                            new_special_k = evicted_k[:, :, :, camera_token_idx:scale_token_idx+1, :].clone()
                            new_special_v = evicted_v[:, :, :, camera_token_idx:scale_token_idx+1, :].clone()

                        if f"k_{global_idx}_special" not in kv_cache or kv_cache[f"k_{global_idx}_special"] is None:
                            kv_cache[f"k_{global_idx}_special"] = new_special_k
                            kv_cache[f"v_{global_idx}_special"] = new_special_v
                        else:
                            kv_cache[f"k_{global_idx}_special"] = torch.cat(
                                [kv_cache[f"k_{global_idx}_special"], new_special_k], dim=2)
                            kv_cache[f"v_{global_idx}_special"] = torch.cat(
                                [kv_cache[f"v_{global_idx}_special"], new_special_v], dim=2)

                    if self.kv_cache_include_scale_frames:
                        kv_cache[f"k_{global_idx}"] = torch.cat([
                            kv_cache[f"k_{global_idx}"][:, :, :scale_frames, :, :],
                            kv_cache[f"k_{global_idx}"][:, :, -sliding_window_frames:, :, :]
                        ], dim=2)
                        kv_cache[f"v_{global_idx}"] = torch.cat([
                            kv_cache[f"v_{global_idx}"][:, :, :scale_frames, :, :],
                            kv_cache[f"v_{global_idx}"][:, :, -sliding_window_frames:, :, :]
                        ], dim=2)
                    else:
                        kv_cache[f"k_{global_idx}"] = kv_cache[f"k_{global_idx}"][:, :, -sliding_window_frames:, :, :]
                        kv_cache[f"v_{global_idx}"] = kv_cache[f"v_{global_idx}"][:, :, -sliding_window_frames:, :, :]
