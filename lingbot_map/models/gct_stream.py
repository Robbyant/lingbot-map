"""
GCTStream - Streaming GCT with KV cache for online inference.

Provides streaming inference functionality:
- Temporal causal attention with KV cache
- Sliding window support
- Efficient frame-by-frame processing
- 3D RoPE support for temporal consistency
"""

import logging
import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from tqdm.auto import tqdm

from lingbot_map.heads.camera_head import CameraCausalHead
from lingbot_map.models.gct_base import GCTBase
from lingbot_map.aggregator.stream import AggregatorStream

logger = logging.getLogger(__name__)

# Debug switch: export LINGBOT_DEBUG_KV=1 to print per-frame KV cache stats in
# the streaming phase-2 loop.  Value: 1 (every frame), N (every N frames),
# empty/unset (disabled).
_KV_DEBUG = os.environ.get("LINGBOT_DEBUG_KV", "")


def _parse_kv_debug_interval(val: str) -> int:
    if not val:
        return 0
    try:
        n = int(val)
    except ValueError:
        return 1
    return max(0, n)


@torch.no_grad()
def _log_kv_stats(model, label: str = "") -> None:
    """One-line dump of aggregator KV cache occupancy + aux counters."""
    try:
        parts = []
        agg = getattr(model, "aggregator", None)
        if agg is not None:
            tfp = getattr(agg, "total_frames_processed", None)
            if tfp is not None:
                parts.append(f"tfp={tfp}")
            mgr = getattr(agg, "kv_cache_manager", None)
            if mgr is not None and hasattr(mgr, "get_cache_stats"):
                s = mgr.get_cache_stats(block_idx=0)
                parts.append(
                    f"FI[blk0] frames={s['frame_count']} "
                    f"scale_pg={s['scale_pages']} live_pg={s['live_pages']} "
                    f"free_pg={s['free_pages']} special={s['special_tokens']}"
                )
            elif isinstance(getattr(agg, "kv_cache", None), dict):
                kv = agg.kv_cache
                k0 = kv.get("k_0")
                if torch.is_tensor(k0):
                    parts.append(f"SDPA[blk0] k_shape={tuple(k0.shape)}")
                parts.append(f"skip_append={kv.get('_skip_append', False)}")
        cam = getattr(model, "camera_head", None)
        if cam is not None:
            fi = getattr(cam, "frame_idx", None)
            if fi is not None:
                parts.append(f"cam.frame_idx={fi}")
        tqdm.write(f"[KV] {label} | {' | '.join(parts)}")
    except Exception as e:  # pragma: no cover — debug helper, never fatal
        tqdm.write(f"[KV] {label} | stats error: {e}")


class GCTStream(GCTBase):
    """
    Streaming GCT model with KV cache for efficient online inference.

    Features:
    - AggregatorStream with KV cache support (FlashInfer backend)
    - CameraCausalHead for pose refinement
    - Sliding window attention for memory efficiency
    - Frame-by-frame streaming inference
    """

    def __init__(
        self,
        # Architecture parameters
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        patch_embed: str = 'dinov2_vitl14_reg',
        pretrained_path: str = '',
        disable_global_rope: bool = False,
        # Head configuration
        enable_camera: bool = True,
        enable_point: bool = False,
        enable_local_point: bool = False,
        enable_depth: bool = True,
        enable_track: bool = False,
        # Normalization
        enable_normalize: bool = False,
        # Prediction normalization
        pred_normalization: bool = False,
        # Stream-specific parameters
        sliding_window_size: int = -1,
        num_frame_for_scale: int = 1,
        num_random_frames: int = 0,
        attend_to_special_tokens: bool = False,
        attend_to_scale_frames: bool = False,
        enable_stream_inference: bool = True,  # Default to True for streaming
        enable_3d_rope: bool = False,
        max_frame_num: int = 1024,
        # Camera head 3D RoPE (separate from aggregator 3D RoPE)
        enable_camera_3d_rope: bool = False,
        camera_rope_theta: float = 10000.0,
        # Scale token configuration (kept for checkpoint compat, ignored)
        use_scale_token: bool = True,
        # KV cache parameters
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
        # Backend selection
        use_sdpa: bool = False,  # If True, use SDPA (no flashinfer needed); default: FlashInfer
        # Gradient checkpointing
        use_gradient_checkpoint: bool = True,
        # Camera head iterative refinement (lower = faster inference; default 4)
        camera_num_iterations: int = 4,
    ):
        """
        Initialize GCTStream.

        Args:
            img_size: Input image size
            patch_size: Patch size for embedding
            embed_dim: Embedding dimension
            patch_embed: Patch embedding type ("dinov2_vitl14_reg", "conv", etc.)
            pretrained_path: Path to pretrained DINOv2 weights
            disable_global_rope: Disable RoPE in global attention
            enable_camera/point/depth/track: Enable prediction heads
            enable_normalize: Enable normalization
            sliding_window_size: Sliding window size in blocks (-1 for full causal)
            num_frame_for_scale: Number of scale estimation frames
            num_random_frames: Number of random frames for long-range dependencies
            attend_to_special_tokens: Enable cross-frame special token attention
            attend_to_scale_frames: Whether to attend to scale frames
            enable_stream_inference: Enable streaming inference with KV cache
            enable_3d_rope: Enable 3D RoPE for temporal consistency
            max_frame_num: Maximum number of frames for 3D RoPE
            use_scale_token: Kept for checkpoint compatibility, ignored
            kv_cache_sliding_window: Sliding window size for KV cache eviction
            kv_cache_scale_frames: Number of scale frames to keep in KV cache
            kv_cache_cross_frame_special: Keep special tokens from evicted frames
            kv_cache_include_scale_frames: Include scale frames in KV cache
            kv_cache_camera_only: Only keep camera tokens from evicted frames
        """
        # Store stream-specific parameters before calling super().__init__()
        self.pretrained_path = pretrained_path
        self.sliding_window_size = sliding_window_size
        self.num_frame_for_scale = num_frame_for_scale
        self.num_random_frames = num_random_frames
        self.attend_to_special_tokens = attend_to_special_tokens
        self.attend_to_scale_frames = attend_to_scale_frames
        self.enable_stream_inference = enable_stream_inference
        self.enable_3d_rope = enable_3d_rope
        self.max_frame_num = max_frame_num
        # Camera head 3D RoPE settings
        self.enable_camera_3d_rope = enable_camera_3d_rope
        self.camera_rope_theta = camera_rope_theta
        # KV cache parameters
        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.kv_cache_scale_frames = kv_cache_scale_frames
        self.kv_cache_cross_frame_special = kv_cache_cross_frame_special
        self.kv_cache_include_scale_frames = kv_cache_include_scale_frames
        self.kv_cache_camera_only = kv_cache_camera_only
        self.use_sdpa = use_sdpa
        self.camera_num_iterations = camera_num_iterations

        # Call base class __init__ (will call _build_aggregator)
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            patch_embed=patch_embed,
            disable_global_rope=disable_global_rope,
            enable_camera=enable_camera,
            enable_point=enable_point,
            enable_local_point=enable_local_point,
            enable_depth=enable_depth,
            enable_track=enable_track,
            enable_normalize=enable_normalize,
            pred_normalization=pred_normalization,
            enable_3d_rope=enable_3d_rope,
            use_gradient_checkpoint=use_gradient_checkpoint,
        )

    def _build_aggregator(self) -> nn.Module:
        """
        Build streaming aggregator with KV cache support (FlashInfer backend).

        Returns:
            AggregatorStream module
        """
        return AggregatorStream(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            patch_embed=self.patch_embed,
            pretrained_path=self.pretrained_path,
            disable_global_rope=self.disable_global_rope,
            sliding_window_size=self.sliding_window_size,
            num_frame_for_scale=self.num_frame_for_scale,
            num_random_frames=self.num_random_frames,
            attend_to_special_tokens=self.attend_to_special_tokens,
            attend_to_scale_frames=self.attend_to_scale_frames,
            enable_stream_inference=self.enable_stream_inference,
            enable_3d_rope=self.enable_3d_rope,
            max_frame_num=self.max_frame_num,
            # Backend: FlashInfer (default) or SDPA (fallback)
            use_flashinfer=not self.use_sdpa,
            use_sdpa=self.use_sdpa,
            kv_cache_sliding_window=self.kv_cache_sliding_window,
            kv_cache_scale_frames=self.kv_cache_scale_frames,
            kv_cache_cross_frame_special=self.kv_cache_cross_frame_special,
            kv_cache_include_scale_frames=self.kv_cache_include_scale_frames,
            kv_cache_camera_only=self.kv_cache_camera_only,
            use_gradient_checkpoint=self.use_gradient_checkpoint,
        )

    def _build_camera_head(self) -> nn.Module:
        """
        Build causal camera head for streaming inference.

        Returns:
            CameraCausalHead module or None
        """
        return CameraCausalHead(
            dim_in=2 * self.embed_dim,
            sliding_window_size=self.sliding_window_size,
            attend_to_scale_frames=self.attend_to_scale_frames,
            num_iterations=self.camera_num_iterations,
            # KV cache parameters
            kv_cache_sliding_window=self.kv_cache_sliding_window,
            kv_cache_scale_frames=self.kv_cache_scale_frames,
            kv_cache_cross_frame_special=self.kv_cache_cross_frame_special,
            kv_cache_include_scale_frames=self.kv_cache_include_scale_frames,
            kv_cache_camera_only=self.kv_cache_camera_only,
            # Camera head 3D RoPE parameters
            enable_3d_rope=self.enable_camera_3d_rope,
            max_frame_num=self.max_frame_num,
            rope_theta=self.camera_rope_theta,
        )

    def _aggregate_features(
        self,
        images: torch.Tensor,
        num_frame_for_scale: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
        num_frame_per_block: int = 1,
        **kwargs,
    ) -> tuple:
        """
        Run aggregator to get multi-scale features.

        Args:
            images: Input images [B, S, 3, H, W]
            num_frame_for_scale: Number of frames for scale estimation
            sliding_window_size: Override sliding window size
            num_frame_per_block: Number of frames per block

        Returns:
            (aggregated_tokens_list, patch_start_idx)
        """
        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images,
            selected_idx=[4, 11, 17, 23],
            num_frame_for_scale=num_frame_for_scale,
            sliding_window_size=sliding_window_size,
            num_frame_per_block=num_frame_per_block,
        )
        return aggregated_tokens_list, patch_start_idx

    def clean_kv_cache(self):
        """
        Clean KV cache in aggregator.

        Call this method when starting a new video sequence to clear
        cached key-value pairs from previous sequences.
        """
        if hasattr(self.aggregator, 'clean_kv_cache'):
            self.aggregator.clean_kv_cache()
        else:
            logger.warning("Aggregator does not support KV cache cleaning")
        if hasattr(self.camera_head, 'kv_cache'):
            self.camera_head.clean_kv_cache()
        else:
            logger.warning("Camera head does not support KV cache cleaning")

    def _set_skip_append(self, skip: bool):
        """Set _skip_append flag on all KV caches (aggregator + camera head).

        When skip=True, attention layers will attend to [cached_kv + current_kv]
        but will NOT store the current frame's KV in cache. This is used for
        non-keyframe processing in keyframe-based streaming inference.

        Args:
            skip: If True, subsequent forward passes will not append KV to cache.
        """
        if hasattr(self.aggregator, 'kv_cache') and self.aggregator.kv_cache is not None:
            self.aggregator.kv_cache["_skip_append"] = skip
        # FlashInfer manager — honored by attention.py's FlashInfer phase-2
        # branch to avoid persisting non-keyframes in the paged cache.
        if hasattr(self.aggregator, 'kv_cache_manager') and self.aggregator.kv_cache_manager is not None:
            self.aggregator.kv_cache_manager._skip_append = skip
        if self.camera_head is not None and hasattr(self.camera_head, 'kv_cache') and self.camera_head.kv_cache is not None:
            for cache_dict in self.camera_head.kv_cache:
                cache_dict["_skip_append"] = skip

    def get_kv_cache_info(self) -> Dict[str, Any]:
        """
        Get information about current KV cache state.

        Returns:
            Dictionary with cache statistics:
                - num_cached_blocks: Number of blocks with cached KV
                - cache_memory_mb: Approximate memory usage in MB
        """
        if not hasattr(self.aggregator, 'kv_cache') or self.aggregator.kv_cache is None:
            return {"num_cached_blocks": 0, "cache_memory_mb": 0.0}

        kv_cache = self.aggregator.kv_cache
        num_cached = sum(1 for k in kv_cache.keys() if k.startswith('k_') and not k.endswith('_special'))

        # Estimate memory usage
        total_elements = 0
        for _, v in kv_cache.items():
            if v is not None and torch.is_tensor(v):
                total_elements += v.numel()

        # Assume bfloat16 (2 bytes per element)
        cache_memory_mb = (total_elements * 2) / (1024 * 1024)

        return {
            "num_cached_blocks": num_cached,
            "cache_memory_mb": round(cache_memory_mb, 2)
        }

    @torch.no_grad()
    def inference_streaming(
        self,
        images: torch.Tensor,
        num_scale_frames: Optional[int] = None,
        keyframe_interval: int = 1,
        output_device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Streaming inference: process scale frames first, then frame-by-frame.

        This method enables efficient online inference by:
        1. Processing initial scale frames together (bidirectional attention via scale token)
        2. Processing remaining frames one-by-one with KV cache (causal streaming)

        Keyframe mode (keyframe_interval > 1):
        - Every keyframe_interval-th frame (after scale frames) is a keyframe
        - Keyframes: KV is stored in cache (normal behavior)
        - Non-keyframes: KV is NOT stored in cache (attend to cached + own KV, then discard)
        - All frames produce full predictions regardless of keyframe status
        - Reduces KV cache memory growth by ~1/keyframe_interval

        Args:
            images: Input images [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1]
            num_scale_frames: Number of initial frames for scale estimation.
                            If None, uses self.num_frame_for_scale.
            keyframe_interval: Every N-th frame (after scale frames) is a keyframe
                             whose KV persists in cache. 1 = every frame is a
                             keyframe (default, same as original behavior).
            output_device: Device to store output predictions on. If None, keeps on
                         the same device as the model. Set to torch.device('cpu')
                         to offload predictions per-frame and avoid GPU OOM on
                         long sequences.

        Returns:
            Dictionary containing predictions for all frames:
                - pose_enc: [B, S, 9]
                - depth: [B, S, H, W, 1]
                - depth_conf: [B, S, H, W]
                - world_points: [B, S, H, W, 3]
                - world_points_conf: [B, S, H, W]
        """
        # Normalize input shape
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, S, C, H, W = images.shape

        # Determine number of scale frames
        scale_frames = num_scale_frames if num_scale_frames is not None else self.num_frame_for_scale
        scale_frames = min(scale_frames, S)  # Cap to available frames

        # Helper to move tensor to output device
        def _to_out(t: torch.Tensor) -> torch.Tensor:
            if output_device is not None:
                return t.to(output_device)
            return t

        # Clean KV caches before starting new sequence
        self.clean_kv_cache()

        # Images may live on CPU for very-long-sequence memory efficiency.
        # We slice-then-move per iteration so peak GPU memory is O(scale) or
        # O(1) frames rather than O(S).
        _model_device = next(self.parameters()).device

        # Phase 1: Process scale frames together
        # These frames get bidirectional attention among themselves via scale token
        logger.info(f'Processing {scale_frames} scale frames...')
        scale_images = images[:, :scale_frames].to(_model_device, non_blocking=True)
        # No-op unless hot modules were compiled with mode="reduce-overhead";
        # required then to reset CUDA-graph step state between replays.
        torch.compiler.cudagraph_mark_step_begin()
        scale_output = self.forward(
            scale_images,
            num_frame_for_scale=scale_frames,
            num_frame_per_block=scale_frames,  # Process all scale frames as one block
            causal_inference=True,
        )

        # Initialize output lists with scale frame predictions (offload if needed)
        all_pose_enc = [_to_out(scale_output["pose_enc"])]
        all_depth = [_to_out(scale_output["depth"])] if "depth" in scale_output else []
        all_depth_conf = [_to_out(scale_output["depth_conf"])] if "depth_conf" in scale_output else []
        all_world_points = [_to_out(scale_output["world_points"])] if "world_points" in scale_output else []
        all_world_points_conf = [_to_out(scale_output["world_points_conf"])] if "world_points_conf" in scale_output else []
        del scale_output

        # Phase 2: Process remaining frames one-by-one
        pbar = tqdm(
            range(scale_frames, S),
            desc='Streaming inference',
            initial=scale_frames,
            total=S,
        )
        dbg_every = _parse_kv_debug_interval(_KV_DEBUG)
        if dbg_every:
            _log_kv_stats(self, label="after phase-1 scale")
        for i in pbar:
            frame_image = images[:, i:i+1].to(_model_device, non_blocking=True)

            # Determine if this frame is a keyframe
            is_keyframe = (keyframe_interval <= 1) or ((i - scale_frames) % keyframe_interval == 0)

            if not is_keyframe:
                self._set_skip_append(True)

            torch.compiler.cudagraph_mark_step_begin()
            frame_output = self.forward(
                frame_image,
                num_frame_for_scale=scale_frames,  # Keep same for scale token logic
                num_frame_per_block=1,  # Single frame per block
                causal_inference=True,
            )

            if not is_keyframe:
                self._set_skip_append(False)

            if dbg_every and ((i - scale_frames) % dbg_every == 0):
                _log_kv_stats(
                    self,
                    label=f"i={i} {'KF' if is_keyframe else 'nkf'}",
                )

            all_pose_enc.append(_to_out(frame_output["pose_enc"]))
            if "depth" in frame_output:
                all_depth.append(_to_out(frame_output["depth"]))
            if "depth_conf" in frame_output:
                all_depth_conf.append(_to_out(frame_output["depth_conf"]))
            if "world_points" in frame_output:
                all_world_points.append(_to_out(frame_output["world_points"]))
            if "world_points_conf" in frame_output:
                all_world_points_conf.append(_to_out(frame_output["world_points_conf"]))
            del frame_output

        # Free GPU memory before concatenation
        if output_device is not None:
            # Move images to output device, then free GPU copy
            images_out = _to_out(images)
            del images
            # Clean KV cache (no longer needed after inference)
            self.clean_kv_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            images_out = images

        # Concatenate all predictions along sequence dimension
        predictions = {
            "pose_enc": torch.cat(all_pose_enc, dim=1),
        }
        del all_pose_enc
        if all_depth:
            predictions["depth"] = torch.cat(all_depth, dim=1)
        del all_depth
        if all_depth_conf:
            predictions["depth_conf"] = torch.cat(all_depth_conf, dim=1)
        del all_depth_conf
        if all_world_points:
            predictions["world_points"] = torch.cat(all_world_points, dim=1)
        del all_world_points
        if all_world_points_conf:
            predictions["world_points_conf"] = torch.cat(all_world_points_conf, dim=1)
        del all_world_points_conf

        # Store images for visualization
        predictions["images"] = images_out

        # Apply prediction normalization if enabled
        if self.pred_normalization:
            predictions = self._normalize_predictions(predictions)

        return predictions
