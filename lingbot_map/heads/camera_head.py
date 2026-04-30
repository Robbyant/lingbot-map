# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lingbot_map.layers import Mlp
from lingbot_map.layers.block import Block
from lingbot_map.layers.block import CameraBlock
from lingbot_map.heads.head_act import activate_pose
from lingbot_map.layers.rope import WanRotaryPosEmbed
from functools import partial
from torch.utils.checkpoint import checkpoint


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4, **kwargs) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape  # S is expected to be 1.
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            for block in self.trunk:
                pose_tokens_modulated = block(pose_tokens_modulated)
            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift


class CameraCausalHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
        num_iterations = 4,
        elementwise_attn_output_gate: bool = False,
        sliding_window_size: int = -1,
        attend_to_scale_frames: bool = False,
        num_random_frames: int = 0,
        attn_class: str = "flexflashattn_varlen",
        # KV cache parameters
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
        # 3D RoPE parameters
        enable_3d_rope: bool = False,
        max_frame_num: int = 1024,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.sliding_window_size = sliding_window_size
        self.num_heads = num_heads

        # 3D RoPE for temporal position encoding
        self.enable_3d_rope = enable_3d_rope
        if enable_3d_rope:
            head_dim = dim_in // num_heads
            # For camera head: each frame has 1 token (frame_seqlen=1)
            # patch_size is (max_frames, h=1, w=1) for 3D RoPE
            # fhw_dim=None lets auto-calculation: h_dim=w_dim=2*(head_dim//6), t_dim=remainder
            self.rope3d = WanRotaryPosEmbed(
                attention_head_dim=head_dim,
                patch_size=(max_frame_num, 1, 1),
                theta=rope_theta,
                fhw_dim=[40, 44, 44],  # Auto-calculate dimension allocation
            )
        else:
            self.rope3d = None

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                CameraBlock(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values, elementwise_attn_output_gate=elementwise_attn_output_gate, sliding_window_size=sliding_window_size, attend_to_scale_frames=attend_to_scale_frames, num_random_frames=num_random_frames, kv_cache_sliding_window=kv_cache_sliding_window, kv_cache_scale_frames=kv_cache_scale_frames, kv_cache_cross_frame_special=kv_cache_cross_frame_special, kv_cache_include_scale_frames=kv_cache_include_scale_frames, kv_cache_camera_only=kv_cache_camera_only)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

        self.num_iterations = num_iterations

        self.kv_cache = None
        self.pos_cache = None
        self.frame_idx = 0

    def clean_kv_cache(self):
        del self.kv_cache
        self.kv_cache = None
        self.frame_idx = 0

    def forward(self, aggregated_tokens_list: list, mask=None, num_iterations: int = None, causal_inference=False, num_frame_per_block=1, num_frame_for_scale=-1, sliding_window_size=None, **kwargs) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps.
                If None, falls back to self.num_iterations (set at construction).
            sliding_window_size (int, optional): Override the sliding window size for this forward pass.
                If None, use the default self.sliding_window_size.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        if num_iterations is None:
            num_iterations = self.num_iterations

        # Use passed sliding_window_size if provided, otherwise use default
        effective_sliding_window_size = sliding_window_size if sliding_window_size is not None else self.sliding_window_size

        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        if causal_inference:
            if self.kv_cache is None:
                self.kv_cache = []
                for i in range(num_iterations):
                    self.kv_cache.append({"_skip_append": False})
                    for j in range(self.trunk_depth):
                        self.kv_cache[i][f"k_{j}"] = None
                        self.kv_cache[i][f"v_{j}"] = None

        pred_pose_enc_list = self.trunk_fn(pose_tokens, mask, num_iterations, num_frame_per_block=num_frame_per_block, num_frame_for_scale=num_frame_for_scale, sliding_window_size=effective_sliding_window_size)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, mask=None, num_iterations: int=4, num_frame_per_block=1, num_frame_for_scale=-1, sliding_window_size=None) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, S, C].
            num_iterations (int): Number of refinement iterations.
            sliding_window_size (int, optional): Sliding window size to use.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []

        # Check if this is the first call (processing scale frames)
        # Scale frames should use batch mode attention for numerical consistency
        is_scale_frames = (self.kv_cache is not None and self.frame_idx == 0)

        # Generate 3D RoPE positions if enabled
        pos3d = None
        if self.rope3d is not None:
            # For camera tokens: shape [B, S, C] where each frame has 1 token
            # Position for frame f is (f, 0, 0) - temporal varies, spatial fixed

            # In streaming mode with KV cache, use frame_idx to track global position
            # Otherwise, generate positions from 0
            if self.kv_cache is not None:
                f_start = self.frame_idx
                f_end = self.frame_idx + S
            else:
                f_start = 0
                f_end = None  # Will use ppf as frame count

            pos3d = self.rope3d(
                ppf=S,                  # Total frames
                pph=1,                  # height = 1 (camera token)
                ppw=1,                  # width = 1 (camera token)
                patch_start_idx=0,      # No special tokens before
                device=pose_tokens.device,
                f_start=f_start,
                f_end=f_end,
            )  # Returns [1, 1, S, head_dim//2] complex

        for i in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            for idx in range(self.trunk_depth):
                pose_tokens_modulated = self.trunk[idx](pose_tokens_modulated, pos=pos3d, video_mask=mask, num_frames=S, frame_seqlen=1, kv_cache=self.kv_cache[i] if self.kv_cache is not None else None, global_idx=idx, num_frame_per_block=num_frame_per_block, num_frame_for_scale=num_frame_for_scale, sliding_window_size=sliding_window_size, enable_3d_rope=self.enable_3d_rope, is_scale_frames=is_scale_frames)
            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)

        # Update frame_idx for streaming mode (KV cache)
        if self.kv_cache is not None:
            self.frame_idx += S

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift




class CameraDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            Block(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                # attn_class=MemEffAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, xpos=None):
        hidden = self.projects(hidden)
        B, V, P, C = hidden.shape
        hidden = hidden.view(hidden.shape[0]*hidden.shape[1], hidden.shape[2], hidden.shape[3])
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, pos=xpos, use_reentrant=False)
            else:
                hidden = blk(hidden, pos=xpos)
        out = self.linear_out(hidden).view(B, V, P, -1)
        
        return out
