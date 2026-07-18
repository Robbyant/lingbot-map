# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings
import math

import torch
from torch import nn, Tensor

from .attention import Attention, CausalAttention, FlashInferAttention, SDPAAttention
from functools import lru_cache, partial
from torch.nn.attention.flex_attention import BlockMask, create_mask
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None,
                num_patches=None, num_special=None, num_frames=None, enable_3d_rope=False) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos,
                                     num_patches=num_patches, num_special=num_special, num_frames=num_frames,
                                     enable_3d_rope=enable_3d_rope))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x, pos=pos, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor, residual_func: Callable[[Tensor], Tensor], sample_drop_ratio: float = 0.0, pos=None
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


class FlashInferBlock(nn.Module):
    """
    FlashInfer variant of causal block for GCT.
    Uses FlashInferAttention (FlashInfer paged KV cache + attention kernels).
    Supports optimized token layout and KV cache streaming inference.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = FlashInferAttention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            rope=rope,
            kv_cache_sliding_window=kv_cache_sliding_window,
            kv_cache_scale_frames=kv_cache_scale_frames,
            kv_cache_cross_frame_special=kv_cache_cross_frame_special,
            kv_cache_include_scale_frames=kv_cache_include_scale_frames,
            kv_cache_camera_only=kv_cache_camera_only,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def attn_pre(self, x: Tensor, pos=None, enable_3d_rope: bool = False) -> tuple:
        """Phase 2 streaming only: norm1 + prepare_qkv fused as one compilable unit.

        Extracted as a named method so torch.compile can capture norm1 + qkv-linear +
        reshape + q_norm + k_norm + RoPE + format as a single CUDA graph.

        Returns:
            (q_nhd, k_nhd, v_nhd) each [tokens_per_frame, num_heads, head_dim],
            ready for manager.append_frame + manager.compute_attention.
        """
        return self.attn.prepare_qkv(self.norm1(x), pos=pos, enable_3d_rope=enable_3d_rope)

    def forward(
        self,
        x: Tensor,
        pos=None,
        num_patches=None,
        num_special=None,
        num_frames=None,
        enable_3d_rope=False,
        kv_cache=None,
        global_idx=0,
        num_frame_per_block=1,
        num_frame_for_scale=-1,
        num_register_tokens=4,
    ) -> Tensor:
        # Phase 2 (streaming): single-frame FlashInfer paged attention.
        # Handle inline so attn_pre (norm1+prepare_qkv) can be compiled as one CUDA graph.
        is_streaming = (kv_cache is not None and (num_frames is None or num_frames <= 1))
        if is_streaming:
            manager = kv_cache
            # Compiled: norm1 + qkv linear + reshape + q_norm + k_norm + RoPE + format
            q_nhd, k_nhd, v_nhd = self.attn_pre(x, pos=pos, enable_3d_rope=enable_3d_rope)

            # Non-keyframe path: attend to cache+current but don't persist the
            # current frame.  FlashInfer paged attention can only read from the
            # paged cache, so we temporarily append (with eviction deferred so
            # it stays clean), attend, and then roll back the append.  Mirrors
            # the ``skip_append`` behavior of the SDPA dict path.
            skip_append = getattr(manager, '_skip_append', False)
            if skip_append:
                prev_defer = manager._defer_eviction
                manager._defer_eviction = True
                manager.append_frame(global_idx, k_nhd, v_nhd)
                attn_x = manager.compute_attention(global_idx, q_nhd)
                manager.rollback_last_frame(global_idx)
                manager._defer_eviction = prev_defer
            else:
                # Eager: write frame K/V to paged cache
                manager.append_frame(global_idx, k_nhd, v_nhd)
                # CPU-only: update eviction state (deque ops, no GPU kernel)
                manager.evict_frames(
                    block_idx=global_idx,
                    scale_frames=self.attn.kv_cache_scale_frames,
                    sliding_window=self.attn.kv_cache_sliding_window,
                    cross_frame_special=self.attn.kv_cache_cross_frame_special,
                    include_scale_frames=self.attn.kv_cache_include_scale_frames,
                    camera_only=self.attn.kv_cache_camera_only,
                    num_register_tokens=num_register_tokens,
                )
                # Eager: FlashInfer BatchPrefillWithPagedKVCacheWrapper
                attn_x = manager.compute_attention(global_idx, q_nhd)

            # [tpf, H, D] -> [B, tpf, C]  (B=1 in streaming, contiguous from FlashInfer output)
            attn_x = attn_x.reshape(x.shape[0], q_nhd.shape[0],
                                    self.attn.num_heads * self.attn.head_dim)
            # Compiled: output projection
            attn_x = self.attn.proj(attn_x)
            x = x + self.ls1(attn_x)
        else:
            # Phase 1 (multi-frame scale pass) or non-streaming training path
            x = x + self.ls1(self.attn(
                self.norm1(x),
                pos=pos,
                num_patches=num_patches,
                num_special=num_special,
                num_frames=num_frames,
                enable_3d_rope=enable_3d_rope,
                kv_cache=kv_cache,
                global_idx=global_idx,
                num_frame_per_block=num_frame_per_block,
                num_frame_for_scale=num_frame_for_scale,
                num_register_tokens=num_register_tokens,
            ))
        x = self.ffn_residual(x)
        return x

    def ffn_residual(self, x: Tensor) -> Tensor:
        """FFN residual branch: norm2 -> mlp -> ls2, WITH residual add fused in.

        Includes the residual add (x + ...) so torch.compile captures the entire
        ffn branch as one CUDA graph.
        """
        return x + self.ls2(self.mlp(self.norm2(x)))


class CameraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        elementwise_attn_output_gate: bool = False,
        sliding_window_size: int = -1,
        attend_to_scale_frames: bool = False,
        num_random_frames: int = 0,
        # KV cache parameters
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = CausalAttention(dim=dim, num_heads=num_heads,
                                    qk_norm=qk_norm, qkv_bias=qkv_bias,
                                    rope=rope, elementwise_attn_output_gate=elementwise_attn_output_gate,
                                    kv_cache_sliding_window=kv_cache_sliding_window,
                                    kv_cache_scale_frames=kv_cache_scale_frames,
                                    kv_cache_cross_frame_special=kv_cache_cross_frame_special,
                                    kv_cache_include_scale_frames=kv_cache_include_scale_frames,
                                    kv_cache_camera_only=kv_cache_camera_only)

        self.sliding_window_size = sliding_window_size
        self.attend_to_scale_frames = attend_to_scale_frames
        self.num_random_frames = num_random_frames

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path
        self.masks = {}

    @torch.no_grad()
    def _prepare_blockwise_causal_attn_mask(self,
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                    KV_LEN=total_length + padded_length, device=device)

        return block_mask

    def forward(self, x: Tensor, pos=None, video_mask=None, num_frames=0, frame_seqlen=0, kv_cache=None, current_start=0, current_end=0, global_idx=0, num_frame_per_block=8, num_frame_for_scale=-1, sliding_window_size=None, full_attention=False, enable_3d_rope=False, is_scale_frames=False) -> Tensor:
        # Use passed sliding_window_size if provided, otherwise use self.sliding_window_size
        effective_sliding_window_size = sliding_window_size if sliding_window_size is not None else self.sliding_window_size

        # Fast path for full attention (camera head) - skip mask computation
        if full_attention:
            def attn_residual_func(x: Tensor, pos=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), pos=pos, full_attention=True, enable_3d_rope=enable_3d_rope))

            def ffn_residual_func(x: Tensor) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            if self.training and self.sample_drop_ratio > 0.0:
                x = x + self.drop_path1(attn_residual_func(x, pos=pos))
                x = x + self.drop_path1(ffn_residual_func(x))
            else:
                x = x + attn_residual_func(x, pos=pos)
                x = x + ffn_residual_func(x)
            return x

        # Skip mask creation when using KV cache (streaming mode) — the streaming
        # attention path in CausalAttention ignores block_mask entirely.
        mask_block = None
        if kv_cache is None:
            mask_block = self._prepare_blockwise_causal_attn_mask(
                device=x.device, num_frames=num_frames, frame_seqlen=frame_seqlen, num_frame_per_block=num_frame_per_block)


        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos, block_mask=mask_block, frame_seqlen=frame_seqlen, video_mask=video_mask, current_start=current_start, current_end=current_end, kv_cache=kv_cache, global_idx=global_idx, num_frame_per_block=num_frame_per_block, num_frame_for_scale=num_frame_for_scale, sliding_window_size=effective_sliding_window_size, attend_to_scale_frames=self.attend_to_scale_frames, num_random_frames=self.num_random_frames,
                                      enable_3d_rope=enable_3d_rope, is_scale_frames=is_scale_frames))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x


class SDPABlock(nn.Module):
    """
    SDPA variant for streaming inference. Uses F.scaled_dot_product_attention
    with dict-based KV cache. No FlashInfer dependency required.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SDPAAttention(
            dim=dim, num_heads=num_heads, qk_norm=qk_norm, qkv_bias=qkv_bias,
            proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop, rope=rope,
            kv_cache_sliding_window=kv_cache_sliding_window,
            kv_cache_scale_frames=kv_cache_scale_frames,
            kv_cache_cross_frame_special=kv_cache_cross_frame_special,
            kv_cache_include_scale_frames=kv_cache_include_scale_frames,
            kv_cache_camera_only=kv_cache_camera_only,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(in_features=dim, hidden_features=int(dim * mlp_ratio),
                             act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None,
                num_patches=None, num_special=None, num_frames=None, enable_3d_rope=False,
                kv_cache=None, global_idx=0, num_frame_per_block=1,
                num_frame_for_scale=-1, num_register_tokens=4) -> Tensor:
        def attn_residual_func(x, pos=None):
            return self.ls1(self.attn(
                self.norm1(x), pos=pos,
                num_patches=num_patches, num_special=num_special, num_frames=num_frames,
                enable_3d_rope=enable_3d_rope, kv_cache=kv_cache, global_idx=global_idx,
                num_frame_per_block=num_frame_per_block, num_frame_for_scale=num_frame_for_scale,
                num_register_tokens=num_register_tokens,
            ))

        def ffn_residual_func(x):
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x
