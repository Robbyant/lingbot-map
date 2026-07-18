"""
GCTBase - Base class for GCT model implementations.

Provides shared functionality:
- Prediction heads (camera, depth, point)
- Forward pass structure
- Model hub mixin (PyTorchModelHubMixin)
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from huggingface_hub import PyTorchModelHubMixin

from lingbot_map.heads.dpt_head import DPTHead
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3

logger = logging.getLogger(__name__)


class GCTBase(nn.Module, PyTorchModelHubMixin, ABC):
    """
    Base class for GCT model implementations.

    Handles shared components:
    - Prediction heads (camera, depth, point)
    - Forward pass structure
    - Input normalization

    Subclasses must implement:
    - _build_aggregator(): Create mode-specific aggregator
    - _build_camera_head(): Create mode-specific camera head
    """

    def __init__(
        self,
        # Architecture parameters
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        patch_embed: str = 'dinov2_vitl14_reg',
        disable_global_rope: bool = False,
        # Head configuration
        enable_camera: bool = True,
        enable_point: bool = True,
        enable_local_point: bool = False,
        enable_depth: bool = True,
        enable_track: bool = False,
        # Camera head sliding window
        enable_camera_sliding_window: bool = False,
        # 3D RoPE
        enable_3d_rope: bool = False,
        # Normalization
        enable_normalize: bool = False,
        # Prediction normalization
        pred_normalization: bool = False,
        pred_normalization_detach_scale: bool = False,
        # Gradient checkpointing
        use_gradient_checkpoint: bool = True,
    ):
        super().__init__()

        # Store configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = patch_embed
        self.disable_global_rope = disable_global_rope

        self.enable_normalize = enable_normalize
        self.pred_normalization = pred_normalization
        self.pred_normalization_detach_scale = pred_normalization_detach_scale
        self.use_gradient_checkpoint = use_gradient_checkpoint

        # Head flags
        self.enable_camera = enable_camera
        self.enable_point = enable_point
        self.enable_local_point = enable_local_point
        self.enable_depth = enable_depth
        self.enable_track = enable_track
        self.enable_camera_sliding_window = enable_camera_sliding_window
        self.enable_3d_rope = enable_3d_rope

        # Build aggregator (subclass-specific)
        self.aggregator = self._build_aggregator()

        # Build prediction heads (subclass-specific)
        self.camera_head = self._build_camera_head() if enable_camera else None
        self.point_head = self._build_point_head() if enable_point else None
        self.local_point_head = self._build_local_point_head() if enable_local_point else None
        self.depth_head = self._build_depth_head() if enable_depth else None

    @abstractmethod
    def _build_aggregator(self) -> nn.Module:
        pass

    @abstractmethod
    def _build_camera_head(self) -> nn.Module:
        pass

    def _build_depth_head(self) -> nn.Module:
        return DPTHead(
            dim_in=2 * self.embed_dim,
            patch_size=self.patch_size,
            output_dim=2,
            activation="exp",
            conf_activation="expp1"
        )

    def _build_point_head(self) -> nn.Module:
        return DPTHead(
            dim_in=2 * self.embed_dim,
            patch_size=self.patch_size,
            output_dim=4,
            activation="inv_log",
            conf_activation="expp1"
        )

    def _build_local_point_head(self) -> nn.Module:
        return DPTHead(
            dim_in=2 * self.embed_dim,
            patch_size=self.patch_size,
            output_dim=4,
            activation="inv_log",
            conf_activation="expp1"
        )

    def _normalize_input(self, images: torch.Tensor, query_points=None):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        return images, query_points

    @abstractmethod
    def _aggregate_features(
        self,
        images: torch.Tensor,
        num_frame_for_scale: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
        num_frame_per_block: int = 1,
        view_graphs: Optional[torch.Tensor] = None,
        causal_graphs: Optional[Union[torch.Tensor, List[np.ndarray]]] = None,
        ordered_video: Optional[torch.Tensor] = None,
    ) -> tuple:
        pass

    def _predict_camera(
        self,
        aggregated_tokens_list: list,
        mask: Optional[torch.Tensor] = None,
        causal_inference: bool = False,
        num_frame_for_scale: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
        num_frame_per_block: int = 1,
        gather_outputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if self.camera_head is None:
            return {}

        aggregated_tokens_list_fp32 = [t.float() for t in aggregated_tokens_list]

        camera_sliding_window = sliding_window_size if self.enable_camera_sliding_window else -1

        with torch.amp.autocast('cuda', enabled=False):
            pose_enc_list = self.camera_head(
                aggregated_tokens_list_fp32,
                mask=mask,
                causal_inference=causal_inference,
                num_frame_for_scale=num_frame_for_scale if num_frame_for_scale is not None else -1,
                sliding_window_size=camera_sliding_window,
                num_frame_per_block=num_frame_per_block,
            )

        return {
            "pose_enc": pose_enc_list[-1],
            "pose_enc_list": pose_enc_list,
        }

    def _predict_depth(
        self,
        aggregated_tokens_list: list,
        images: torch.Tensor,
        patch_start_idx: int,
        gather_outputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if self.depth_head is None:
            return {}

        aggregated_tokens_list_fp32 = [t.float() for t in aggregated_tokens_list]
        images_fp32 = images.float()

        with torch.amp.autocast('cuda', enabled=False):
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list_fp32,
                images=images_fp32,
                patch_start_idx=patch_start_idx
            )

        return {"depth": depth, "depth_conf": depth_conf}

    def _predict_points(
        self,
        aggregated_tokens_list: list,
        images: torch.Tensor,
        patch_start_idx: int,
        gather_outputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if self.point_head is None:
            return {}

        aggregated_tokens_list_fp32 = [t.float() for t in aggregated_tokens_list]
        images_fp32 = images.float()

        with torch.amp.autocast('cuda', enabled=False):
            pts3d, pts3d_conf = self.point_head(
                aggregated_tokens_list_fp32,
                images=images_fp32,
                patch_start_idx=patch_start_idx
            )

        return {"world_points": pts3d, "world_points_conf": pts3d_conf}

    def _predict_local_points(
        self,
        aggregated_tokens_list: list,
        images: torch.Tensor,
        patch_start_idx: int,
        gather_outputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if self.local_point_head is None:
            return {}

        aggregated_tokens_list_fp32 = [t.float() for t in aggregated_tokens_list]
        images_fp32 = images.float()

        with torch.amp.autocast('cuda', enabled=False):
            pts3d, pts3d_conf = self.local_point_head(
                aggregated_tokens_list_fp32,
                images=images_fp32,
                patch_start_idx=patch_start_idx
            )

        return {"cam_points": pts3d, "cam_points_conf": pts3d_conf}

    def _unproject_depth_to_world(
        self,
        depth: torch.Tensor,
        pose_enc: torch.Tensor,
    ) -> torch.Tensor:
        B, S, H, W, _ = depth.shape
        device = depth.device
        dtype = depth.dtype

        image_size_hw = (H, W)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=image_size_hw, build_intrinsics=True
        )

        extrinsics_flat = extrinsics.view(B * S, 3, 4)
        extrinsics_4x4 = torch.zeros(B * S, 4, 4, device=device, dtype=dtype)
        extrinsics_4x4[:, :3, :] = extrinsics_flat
        extrinsics_4x4[:, 3, 3] = 1.0
        c2w = closed_form_inverse_se3(extrinsics_4x4).view(B, S, 4, 4)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)

        intrinsics_inv = torch.inverse(intrinsics)
        camera_coords = torch.einsum('bsij,hwj->bshwi', intrinsics_inv, pixel_coords)
        camera_points = camera_coords * depth

        ones = torch.ones_like(camera_points[..., :1])
        camera_points_h = torch.cat([camera_points, ones], dim=-1)
        world_points_h = torch.einsum('bsij,bshwj->bshwi', c2w, camera_points_h)

        return world_points_h[..., :3]

    def forward(
        self,
        images: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
        num_frame_for_scale: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
        num_frame_per_block: int = 1,
        mask: Optional[torch.Tensor] = None,
        causal_inference: bool = False,
        ordered_video: Optional[torch.Tensor] = None,
        gather_outputs: bool = True,
        point_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the GCT model.

        Args:
            images: Input images [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1]
            query_points: Optional query points [N, 2] or [B, N, 2]

        Returns:
            Dictionary containing predictions:
                - pose_enc: Camera pose encoding [B, S, 9]
                - depth: Depth maps [B, S, H, W, 1]
                - depth_conf: Depth confidence [B, S, H, W]
                - world_points: 3D world coordinates [B, S, H, W, 3]
                - world_points_conf: Point confidence [B, S, H, W]
        """
        images, query_points = self._normalize_input(images, query_points)

        aggregated_tokens_list, patch_start_idx = self._aggregate_features(
            images,
            num_frame_for_scale=num_frame_for_scale,
            sliding_window_size=sliding_window_size,
            num_frame_per_block=num_frame_per_block,
        )

        predictions = {}

        predictions.update(self._predict_camera(
            aggregated_tokens_list,
            mask=ordered_video,
            causal_inference=causal_inference,
            num_frame_for_scale=num_frame_for_scale,
            sliding_window_size=sliding_window_size,
            num_frame_per_block=num_frame_per_block,
            gather_outputs=gather_outputs,
        ))

        predictions.update(self._predict_depth(
            aggregated_tokens_list, images, patch_start_idx,
            gather_outputs=gather_outputs,
        ))

        predictions.update(self._predict_points(
            aggregated_tokens_list, images, patch_start_idx,
            gather_outputs=gather_outputs,
        ))

        predictions.update(self._predict_local_points(
            aggregated_tokens_list, images, patch_start_idx,
            gather_outputs=gather_outputs,
        ))

        if not self.training:
            predictions["images"] = images

        return predictions
