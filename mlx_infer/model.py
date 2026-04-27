"""
GCTStreamMLX — full streaming model in MLX.

Mirrors GCTStream from lingbot_map/models/gct_stream.py:
  - AggregatorMLX backbone (DINOv2 + frame/global blocks)
  - CameraHeadMLX (pose, 4-iteration refinement)
  - DPTHeadMLX    (depth, world_points)
  - inference_streaming: Phase 1 (scale frames) + Phase 2 (frame-by-frame KV cache)
"""

from typing import Optional, Dict, List, Any
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from tqdm.auto import tqdm

from .aggregator import AggregatorMLX
from .heads import CameraHeadMLX, DPTHeadMLX


class GCTStreamMLX(nn.Module):
    """MLX streaming GCT model.

    Parameters
    ----------
    img_size, patch_size, embed_dim : int
        ViT-L defaults: 518, 14, 1024.
    kv_cache_sliding_window : int
        Sliding window for KV cache eviction (default 64 frames).
    kv_cache_scale_frames : int
        Number of scale frames kept in KV cache (default 8).
    camera_num_iterations : int
        Refinement iterations in camera head (default 4).
    enable_depth, enable_point : bool
        Which dense heads to build.
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        num_register_tokens: int = 4,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_keep_special: bool = True,
        camera_num_iterations: int = 4,
        enable_depth: bool = True,
        enable_point: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        dim_2c = 2 * embed_dim   # concatenated frame+global output dim

        self.aggregator = AggregatorMLX(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_register_tokens=num_register_tokens,
            kv_cache_sliding_window=kv_cache_sliding_window,
            kv_cache_scale_frames=kv_cache_scale_frames,
            kv_cache_keep_special=kv_cache_keep_special,
        )

        self.camera_head = CameraHeadMLX(
            dim_in=dim_2c,
            trunk_depth=4,
            num_heads=num_heads,
            num_iterations=camera_num_iterations,
            kv_cache_sliding_window=kv_cache_sliding_window,
            kv_cache_scale_frames=kv_cache_scale_frames,
            kv_cache_keep_special=kv_cache_keep_special,
        )

        self.depth_head = DPTHeadMLX(
            dim_in=dim_2c, patch_size=patch_size, output_dim=2,
            activation="exp", conf_activation="expp1",
        ) if enable_depth else None

        self.point_head = DPTHeadMLX(
            dim_in=dim_2c, patch_size=patch_size, output_dim=4,
            activation="inv_log", conf_activation="expp1",
        ) if enable_point else None

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def clean_kv_cache(self):
        self.aggregator.clean_kv_cache()
        self.camera_head.clean_kv_cache()

    def _set_skip_append(self, skip: bool):
        self.aggregator.set_skip_append(skip)
        self.camera_head.set_skip_append(skip)

    # ------------------------------------------------------------------
    # Single forward pass
    # ------------------------------------------------------------------

    def __call__(
        self,
        images: mx.array,
        num_frame_for_scale: Optional[int] = None,
        num_frame_per_block: int = 1,
        causal_inference: bool = False,
    ) -> Dict[str, mx.array]:
        """
        images: [B, S, 3, H, W] in [0, 1].
        Returns dict with 'pose_enc', optionally 'depth', 'depth_conf',
        'world_points', 'world_points_conf'.
        """
        if images.ndim == 4:
            images = images[None]   # add batch dim

        # Aggregate features
        agg_list, patch_start_idx = self.aggregator(
            images,
            selected_idx=[4, 11, 17, 23],
            num_frame_for_scale=num_frame_for_scale,
            num_frame_per_block=num_frame_per_block,
        )
        mx.eval(agg_list)   # materialise before heads

        predictions: Dict[str, mx.array] = {}

        # Camera
        pose_list = self.camera_head(
            agg_list,
            causal_inference=causal_inference,
            num_iterations=None,
            num_frame_per_block=num_frame_per_block,
            num_frame_for_scale=num_frame_for_scale if num_frame_for_scale is not None else -1,
        )
        predictions["pose_enc"] = pose_list[-1]

        # Depth
        if self.depth_head is not None:
            depth, depth_conf = self.depth_head(agg_list, images, patch_start_idx)
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        # World points
        if self.point_head is not None:
            pts3d, pts3d_conf = self.point_head(agg_list, images, patch_start_idx)
            predictions["world_points"] = pts3d
            predictions["world_points_conf"] = pts3d_conf

        return predictions

    # ------------------------------------------------------------------
    # Streaming inference
    # ------------------------------------------------------------------

    def inference_streaming(
        self,
        images: mx.array,
        num_scale_frames: Optional[int] = None,
        keyframe_interval: int = 1,
    ) -> Dict[str, mx.array]:
        """
        Streaming inference: process scale frames first, then frame-by-frame.

        Parameters
        ----------
        images : mx.array
            [S, 3, H, W] or [B, S, 3, H, W] in [0, 1].
        num_scale_frames : int, optional
            Initial bidirectional frames (default: aggregator patch_start_idx).
        keyframe_interval : int
            Every N-th frame after scale phase is a keyframe (KV stored).
            1 = every frame (default, original behaviour).

        Returns
        -------
        dict with keys: pose_enc, depth, depth_conf, world_points,
                        world_points_conf, images.
        """
        if images.ndim == 4:
            images = images[None]   # [1, S, 3, H, W]
        B, S, _, H, W = images.shape

        scale_frames = num_scale_frames if num_scale_frames is not None else 1
        scale_frames = min(scale_frames, S)

        # Clean caches before new sequence
        self.clean_kv_cache()

        # ------ Phase 1: scale frames (bidirectional via scale_token) ------
        scale_out = self(
            images[:, :scale_frames],
            num_frame_for_scale=scale_frames,
            num_frame_per_block=scale_frames,
            causal_inference=True,
        )
        mx.eval(scale_out)

        all_pose  = [scale_out["pose_enc"]]
        all_depth = [scale_out["depth"]] if "depth" in scale_out else []
        all_dconf = [scale_out["depth_conf"]] if "depth_conf" in scale_out else []
        all_pts   = [scale_out["world_points"]] if "world_points" in scale_out else []
        all_pconf = [scale_out["world_points_conf"]] if "world_points_conf" in scale_out else []
        del scale_out

        # ------ Phase 2: streaming frame-by-frame ------
        for i in tqdm(range(scale_frames, S), desc="Streaming", initial=scale_frames, total=S):
            is_keyframe = (keyframe_interval <= 1) or ((i - scale_frames) % keyframe_interval == 0)

            if not is_keyframe:
                self._set_skip_append(True)

            frame_out = self(
                images[:, i:i+1],
                num_frame_for_scale=scale_frames,
                num_frame_per_block=1,
                causal_inference=True,
            )
            mx.eval(frame_out)

            if not is_keyframe:
                self._set_skip_append(False)

            all_pose.append(frame_out["pose_enc"])
            if "depth"           in frame_out: all_depth.append(frame_out["depth"])
            if "depth_conf"      in frame_out: all_dconf.append(frame_out["depth_conf"])
            if "world_points"    in frame_out: all_pts.append(frame_out["world_points"])
            if "world_points_conf" in frame_out: all_pconf.append(frame_out["world_points_conf"])
            del frame_out

        self.clean_kv_cache()

        result: Dict[str, mx.array] = {
            "pose_enc": mx.concatenate(all_pose, axis=1),
            "images":   images,
        }
        if all_depth: result["depth"]              = mx.concatenate(all_depth, axis=1)
        if all_dconf: result["depth_conf"]         = mx.concatenate(all_dconf, axis=1)
        if all_pts:   result["world_points"]       = mx.concatenate(all_pts,   axis=1)
        if all_pconf: result["world_points_conf"]  = mx.concatenate(all_pconf, axis=1)
        return result
