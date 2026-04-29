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

# Debug switch: export LINGBOT_DEBUG_KV=1 to print per-frame KV cache stats in
# windowed phase-2 loop.  Value can be 1 (every frame), or an integer N to print
# every N frames.
_KV_DEBUG = os.environ.get("LINGBOT_DEBUG_KV", "")

from lingbot_map.utils.rotation import quat_to_mat, mat_to_quat

from lingbot_map.heads.camera_head import CameraCausalHead
from lingbot_map.models.gct_base import GCTBase
from lingbot_map.aggregator.stream import AggregatorStream
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3

logger = logging.getLogger(__name__)


def _parse_kv_debug_interval(val: str) -> int:
    """Parse LINGBOT_DEBUG_KV env var into a print-every-N interval.

    Accepts: "" (disabled) | "0" (disabled) | "1" (every frame)
             | "N" (every N frames) | any other truthy string → every frame.
    """
    if not val:
        return 0
    try:
        n = int(val)
    except ValueError:
        return 1
    return max(0, n)


@torch.no_grad()
def _log_kv_stats(model, label: str = "") -> None:
    """Dump a one-line summary of the aggregator's KV cache occupancy.

    Prints FlashInfer manager stats (if present) and SDPA dict stats otherwise,
    plus aggregator ``total_frames_processed``.  Camera head frame_idx is also
    reported — it is unused when 3D RoPE is disabled but helps catch mismatches.
    """
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


@torch.no_grad()
def _compute_flow_magnitude(
    cur_pose_enc: torch.Tensor,
    kf_pose_enc: torch.Tensor,
    cur_depth: torch.Tensor,
    image_size_hw: tuple,
    stride: int = 8,
) -> float:
    """Compute mean optical flow magnitude induced by camera motion.

    Projects current frame pixels into the last keyframe camera using the
    current depth map and both frames' poses, then returns the average
    pixel displacement (L2 norm of flow) over valid pixels.

    Args:
        cur_pose_enc: Current frame pose encoding [B, 1, 9].
        kf_pose_enc: Last keyframe pose encoding [B, 1, 9].
        cur_depth: Current frame depth map [B, 1, H, W, 1].
        image_size_hw: (H, W) of the depth map.
        stride: Subsampling stride for efficiency.

    Returns:
        Mean flow magnitude in pixels (scalar float).
    """
    H, W = image_size_hw
    device = cur_pose_enc.device
    dtype = cur_depth.dtype

    cur_ext, cur_intr = pose_encoding_to_extri_intri(
        cur_pose_enc, image_size_hw=image_size_hw
    )
    kf_ext, kf_intr = pose_encoding_to_extri_intri(
        kf_pose_enc, image_size_hw=image_size_hw
    )
    B = cur_ext.shape[0]

    cur_ext = cur_ext[:, 0]
    cur_intr = cur_intr[:, 0]
    kf_ext = kf_ext[:, 0]
    kf_intr = kf_intr[:, 0]

    depth = cur_depth[:, 0, ::stride, ::stride, 0].to(dtype)
    Hs, Ws = depth.shape[1], depth.shape[2]

    v_coords = torch.arange(0, H, stride, device=device, dtype=dtype)
    u_coords = torch.arange(0, W, stride, device=device, dtype=dtype)
    v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')
    ones = torch.ones_like(u_grid)
    pixel_coords = torch.stack([u_grid, v_grid, ones], dim=-1)

    intr_inv = torch.inverse(cur_intr)
    cam_coords = torch.einsum('bij,hwj->bhwi', intr_inv, pixel_coords)
    cam_pts = cam_coords * depth.unsqueeze(-1)

    c2w = torch.zeros(B, 4, 4, device=device, dtype=dtype)
    c2w[:, :3, :] = cur_ext
    c2w[:, 3, 3] = 1.0

    ones_hw = torch.ones(B, Hs, Ws, 1, device=device, dtype=dtype)
    cam_pts_h = torch.cat([cam_pts, ones_hw], dim=-1)
    world_pts = torch.einsum('bij,bhwj->bhwi', c2w, cam_pts_h)[..., :3]

    kf_c2w = torch.zeros(B, 4, 4, device=device, dtype=dtype)
    kf_c2w[:, :3, :] = kf_ext
    kf_c2w[:, 3, 3] = 1.0
    kf_w2c = closed_form_inverse_se3(kf_c2w)
    world_pts_h = torch.cat([world_pts, ones_hw], dim=-1)
    kf_cam_pts = torch.einsum('bij,bhwj->bhwi', kf_w2c, world_pts_h)[..., :3]

    z = kf_cam_pts[..., 2:3].clamp(min=1e-6)
    kf_cam_norm = kf_cam_pts / z
    kf_pixels = torch.einsum('bij,bhwj->bhwi', kf_intr, kf_cam_norm)[..., :2]

    orig_pixels = torch.stack([u_grid, v_grid], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    flow = kf_pixels - orig_pixels
    valid = (depth > 1e-6) & (kf_cam_pts[..., 2] > 1e-6)

    flow_mag = flow.norm(dim=-1)
    valid_count = valid.float().sum()
    if valid_count < 1:
        return 0.0

    mean_mag = (flow_mag * valid.float()).sum() / valid_count
    return mean_mag.item()


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
        enable_point: bool = True,
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
        # FlashInfer manager
        if hasattr(self.aggregator, 'kv_cache_manager') and self.aggregator.kv_cache_manager is not None:
            self.aggregator.kv_cache_manager._skip_append = skip
        if self.camera_head is not None and hasattr(self.camera_head, 'kv_cache') and self.camera_head.kv_cache is not None:
            for cache_dict in self.camera_head.kv_cache:
                cache_dict["_skip_append"] = skip

    # ── Flow-based keyframe helpers ────────────────────────────────────────

    def _set_defer_eviction(self, defer: bool):
        """Set defer-eviction flag on FlashInfer manager and SDPA caches.

        While True, eviction is suppressed so that rollback can cleanly undo
        the most recent append without having to restore evicted frames.
        """
        # FlashInfer manager
        if hasattr(self.aggregator, 'kv_cache_manager') and self.aggregator.kv_cache_manager is not None:
            self.aggregator.kv_cache_manager._defer_eviction = defer
        # SDPA aggregator cache (dict)
        if hasattr(self.aggregator, 'kv_cache') and isinstance(self.aggregator.kv_cache, dict):
            self.aggregator.kv_cache["_defer_eviction"] = defer
        # Camera head SDPA caches
        if self.camera_head is not None and hasattr(self.camera_head, 'kv_cache') and self.camera_head.kv_cache is not None:
            for cache_dict in self.camera_head.kv_cache:
                cache_dict["_defer_eviction"] = defer

    def _rollback_last_frame(self):
        """Rollback the most recent frame from all caches.

        Undoes append_frame on FlashInfer manager (all blocks), trims the
        camera head SDPA cache, and decrements the aggregator frame counter.
        Must be called while eviction is still deferred.
        """
        # FlashInfer manager — rollback each transformer block
        if hasattr(self.aggregator, 'kv_cache_manager') and self.aggregator.kv_cache_manager is not None:
            mgr = self.aggregator.kv_cache_manager
            for block_idx in range(mgr.num_blocks):
                mgr.rollback_last_frame(block_idx)

        # SDPA aggregator cache — trim last frame along dim=2
        if hasattr(self.aggregator, 'kv_cache') and isinstance(self.aggregator.kv_cache, dict):
            kv = self.aggregator.kv_cache
            for key in list(kv.keys()):
                if key.startswith(("k_", "v_")) and kv[key] is not None and torch.is_tensor(kv[key]):
                    if kv[key].dim() >= 3 and kv[key].shape[2] > 1:
                        kv[key] = kv[key][:, :, :-1]
                    elif kv[key].dim() >= 3:
                        kv[key] = None

        # Camera head
        if self.camera_head is not None and hasattr(self.camera_head, 'rollback_last_frame'):
            self.camera_head.rollback_last_frame()

        # Aggregator frame counter (used for 3D RoPE temporal positions)
        self.aggregator.total_frames_processed -= 1

    def _execute_deferred_eviction(self):
        """Execute the eviction that was deferred during the last forward pass."""
        # FlashInfer manager
        if hasattr(self.aggregator, 'kv_cache_manager') and self.aggregator.kv_cache_manager is not None:
            mgr = self.aggregator.kv_cache_manager
            for block_idx in range(mgr.num_blocks):
                mgr.execute_deferred_eviction(
                    block_idx,
                    scale_frames=self.kv_cache_scale_frames,
                    sliding_window=self.kv_cache_sliding_window,
                )

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
        flow_threshold: float = 0.0,
        max_non_keyframe_gap: int = 30,
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

        Flow-based keyframe mode (flow_threshold > 0):
        - Takes precedence over keyframe_interval
        - Computes optical flow magnitude between current frame and last keyframe
        - Frame becomes keyframe if flow exceeds threshold or gap exceeds max_non_keyframe_gap
        - Uses defer-eviction + rollback for non-keyframes

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
            flow_threshold: Mean flow magnitude threshold (pixels) for flow-based
                keyframe selection. >0 enables flow-based mode (takes precedence
                over keyframe_interval).
            max_non_keyframe_gap: Max consecutive non-keyframe frames before
                forcing a keyframe (flow mode only).

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

        # Slice-then-move per iteration so `images` may live on CPU and we
        # don't blow up GPU memory on tens-of-thousands-of-frame sequences.
        _model_device = next(self.parameters()).device

        # Phase 1: Process scale frames together
        # These frames get bidirectional attention among themselves via scale token
        logger.info(f'Processing {scale_frames} scale frames...')
        scale_images = images[:, :scale_frames].to(_model_device, non_blocking=True)
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
        use_flow_keyframe = flow_threshold > 0.0

        # Flow state: last keyframe = last scale frame
        if use_flow_keyframe:
            last_kf_pose_enc = all_pose_enc[0][:, -1:]  # last scale frame
            last_kf_idx = scale_frames - 1

        pbar = tqdm(
            range(scale_frames, S),
            desc='Streaming inference',
            initial=scale_frames,
            total=S,
        )
        for i in pbar:
            frame_image = images[:, i:i+1].to(_model_device, non_blocking=True)

            if use_flow_keyframe:
                # Flow-based: defer eviction, forward, then decide
                self._set_defer_eviction(True)

                frame_output = self.forward(
                    frame_image,
                    num_frame_for_scale=scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )

                self._set_defer_eviction(False)

                # Compute flow to decide keyframe
                cur_depth = frame_output.get("depth", None)
                if cur_depth is not None:
                    H_pred, W_pred = cur_depth.shape[2], cur_depth.shape[3]
                    flow_mag = _compute_flow_magnitude(
                        frame_output["pose_enc"], last_kf_pose_enc,
                        cur_depth, (H_pred, W_pred),
                    )
                else:
                    flow_mag = flow_threshold + 1.0

                frames_since_kf = i - last_kf_idx
                is_keyframe = (
                    (i == scale_frames)  # first streaming frame
                    or (flow_mag > flow_threshold)
                    or (frames_since_kf >= max_non_keyframe_gap)
                )

                if is_keyframe:
                    self._execute_deferred_eviction()
                    last_kf_pose_enc = frame_output["pose_enc"]
                    last_kf_idx = i
                else:
                    self._rollback_last_frame()
            else:
                # Fixed-interval keyframe mode
                is_keyframe = (keyframe_interval <= 1) or ((i - scale_frames) % keyframe_interval == 0)

                if not is_keyframe:
                    self._set_skip_append(True)

                frame_output = self.forward(
                    frame_image,
                    num_frame_for_scale=scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )

                if not is_keyframe:
                    self._set_skip_append(False)

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

    # ══════════════════════════════════════════════════════════════════════
    # Window stitching & cross-window alignment
    # ══════════════════════════════════════════════════════════════════════

    _FRAME_AXIS_KEYS = frozenset({
        "pose_enc", "depth", "depth_conf",
        "world_points", "world_points_conf",
        "frame_type", "is_keyframe",
    })

    def _stitch_windows(
        self,
        windows: List[Dict],
        window_size: int,
        overlap: int,
    ) -> Dict:
        """Concatenate per-window predictions while de-duplicating overlaps.

        For each temporal key the method builds a slice table first — every
        window contributes ``[0, effective_end)`` frames where
        ``effective_end = total_frames - overlap`` for non-final windows.
        Non-temporal entries simply keep the latest available value.
        """
        if len(windows) == 0:
            return {}
        if len(windows) == 1:
            return windows[0]

        n_win = len(windows)
        all_keys = list(windows[0].keys())
        stitched: Dict = {}

        for key in all_keys:
            values = [w.get(key) for w in windows]
            if all(v is None for v in values):
                continue

            # Non-temporal entries: take latest
            if key not in self._FRAME_AXIS_KEYS:
                stitched[key] = next(v for v in reversed(values) if v is not None)
                continue

            # Build slice table: (start, end) for each window's contribution
            slices = []
            for wi, tensor in enumerate(values):
                if tensor is None:
                    slices.append(None)
                    continue
                total = tensor.shape[1]
                is_last = (wi == n_win - 1)
                end = total if is_last else max(total - overlap, 0)
                slices.append((0, end) if end > 0 else None)

            parts = [
                values[i][:, s:e]
                for i, s_e in enumerate(slices)
                if s_e is not None
                for s, e in [s_e]
            ]
            if parts:
                stitched[key] = torch.cat(parts, dim=1)
            else:
                fallback = next((v for v in reversed(values) if v is not None), None)
                if fallback is not None:
                    stitched[key] = fallback

        return stitched

    @staticmethod
    def _depth_ratio_scale(
        anchor_depth: torch.Tensor,
        target_depth: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Estimate per-batch scale as the median depth ratio anchor/target."""
        a = anchor_depth.to(torch.float32).reshape(batch_size, -1)
        t = target_depth.to(torch.float32).reshape(batch_size, -1)
        ok = torch.isfinite(a) & torch.isfinite(t) & (t.abs() > torch.finfo(torch.float32).eps)

        scales = []
        for b in range(batch_size):
            m = ok[b]
            if m.any():
                scales.append((a[b, m] / t[b, m]).median())
            else:
                scales.append(torch.tensor(1.0, device=device, dtype=torch.float32))
        return torch.stack(scales).clamp(min=1e-3, max=1e3)

    @staticmethod
    def _pairwise_alignment(
        prev_pred: Dict,
        curr_pred: Dict,
        overlap: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Compute (scale, R, t) that maps *curr* into *prev*'s coordinate frame.

        Anchors on the latest physical time step inside the overlap region
        that is a keyframe in BOTH windows, and aggregates depth across all
        such paired keyframes for scale estimation. Falls back to the first
        overlap frame (legacy behavior) when no keyframe pairing is available.
        """
        unit_s = torch.ones(batch_size, device=device, dtype=dtype)
        eye_R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
        zero_t = torch.zeros(batch_size, 3, device=device, dtype=dtype)

        if overlap <= 0:
            return unit_s, eye_R, zero_t

        pe_prev = prev_pred.get("pose_enc")
        pe_curr = curr_pred.get("pose_enc")
        if pe_prev is None or pe_curr is None:
            return unit_s, eye_R, zero_t

        total_prev = pe_prev.shape[1]
        total_curr = pe_curr.shape[1]
        start = max(total_prev - overlap, 0)
        eff_overlap = min(overlap, total_prev - start, total_curr)

        # Pick keyframe-paired offsets inside overlap (keyframe in both windows).
        is_kf_prev = prev_pred.get("is_keyframe")
        is_kf_curr = curr_pred.get("is_keyframe")
        paired_offsets = None
        if is_kf_prev is not None and is_kf_curr is not None and eff_overlap > 0:
            kf_prev_tail = is_kf_prev[0, start:start + eff_overlap].to(torch.bool)
            kf_curr_head = is_kf_curr[0, :eff_overlap].to(torch.bool)
            paired_mask = kf_prev_tail & kf_curr_head
            if paired_mask.any():
                paired_offsets = torch.nonzero(paired_mask, as_tuple=False).flatten().to(device)

        if paired_offsets is not None:
            anchor_offset = int(paired_offsets[-1].item())
            idx_a = start + anchor_offset
            idx_b = anchor_offset
            prev_depth_idx = start + paired_offsets
            curr_depth_idx = paired_offsets
        else:
            idx_a = start
            idx_b = 0
            prev_depth_idx = torch.tensor([idx_a], device=device, dtype=torch.long)
            curr_depth_idx = torch.tensor([0], device=device, dtype=torch.long)

        # Decompose C2W: center ([:3]) + quaternion ([3:7])
        Ra = quat_to_mat(pe_prev[:, idx_a, 3:7])
        ca = pe_prev[:, idx_a, :3]
        Rb = quat_to_mat(pe_curr[:, idx_b, 3:7])
        cb = pe_curr[:, idx_b, :3]

        R_ab = torch.bmm(Ra, Rb.transpose(1, 2))

        # Scale from depth — aggregate across all paired keyframes in overlap.
        s_ab = unit_s.clone()
        da = prev_pred.get("depth")
        db = curr_pred.get("depth")
        if (da is not None and db is not None
                and int(prev_depth_idx.max().item()) < da.shape[1]
                and int(curr_depth_idx.max().item()) < db.shape[1]):
            s_ab = GCTStream._depth_ratio_scale(
                da[:, prev_depth_idx, ..., 0],
                db[:, curr_depth_idx, ..., 0],
                batch_size, device,
            ).to(dtype)

        # ca = s_ab * R_ab @ cb + t_ab  =>  t_ab = ca - s_ab * R_ab @ cb
        t_ab = ca - s_ab.unsqueeze(-1) * torch.bmm(R_ab, cb.unsqueeze(-1)).squeeze(-1)

        return s_ab, R_ab.to(dtype), t_ab.to(dtype)

    @staticmethod
    def _warp_predictions(
        pred: Dict,
        R: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        batch_size: int,
    ) -> Dict:
        """Apply a similarity transform (s, R, t) to one window's predictions."""
        warped: Dict = {}

        # Pose encoding: center + quaternion + intrinsics
        pe = pred.get("pose_enc")
        if pe is not None:
            nf = pe.shape[1]
            local_rot = quat_to_mat(pe[:, :, 3:7])
            local_ctr = pe[:, :, :3]

            R_exp = R[:, None].expand(-1, nf, -1, -1)
            new_rot = torch.matmul(R_exp, local_rot)
            new_ctr = (
                s.view(batch_size, 1, 1) * torch.matmul(R_exp, local_ctr.unsqueeze(-1)).squeeze(-1)
                + t.view(batch_size, 1, 3)
            )
            out_pe = pe.clone()
            out_pe[:, :, :3] = new_ctr
            out_pe[:, :, 3:7] = mat_to_quat(new_rot)
            warped["pose_enc"] = out_pe
        else:
            warped["pose_enc"] = None

        # Depth: scale by s
        d = pred.get("depth")
        if d is not None:
            warped["depth"] = d * s.view(batch_size, 1, 1, 1, 1)
        else:
            warped["depth"] = None

        # World points: p_global = s * R @ p_local + t
        wp = pred.get("world_points")
        if wp is not None:
            b, nf, h, w, _ = wp.shape
            flat = wp.reshape(b, nf * h * w, 3)
            transformed = torch.bmm(flat, R.transpose(1, 2)) * s.view(b, 1, 1)
            transformed = transformed + t[:, None, :]
            warped["world_points"] = transformed.reshape(b, nf, h, w, 3)
        else:
            warped["world_points"] = None

        # Pass through all other keys untouched
        for k, v in pred.items():
            if k not in warped:
                warped[k] = v

        return warped

    def _align_and_stitch_windows(
        self,
        windows: List[Dict],
        scale_mode: str = 'median',
    ) -> Dict:
        """Bring all windows into the first window's coordinate frame, then stitch.

        Iterates over consecutive window pairs, estimates the pairwise
        scaled alignment, warps each window, and finally concatenates
        via :meth:`_stitch_windows`.
        """
        if len(windows) == 0:
            return {}
        if len(windows) == 1:
            out = windows[0].copy()
            out["alignment_mode"] = "scaled"
            return out

        # Discover batch / device / dtype from any available tensor
        ref = next(
            v
            for w in windows
            for k in ("pose_enc", "world_points", "depth")
            if (v := w.get(k)) is not None
        )
        dev, dt, nb = ref.device, ref.dtype, ref.shape[0]

        overlap = getattr(self, "_last_overlap_size", 0)
        win_sz = getattr(self, "_last_window_size", -1)

        warped_windows: List[Dict] = []
        per_window_scales: List[torch.Tensor] = []
        per_window_transforms: List[torch.Tensor] = []

        for idx, raw in enumerate(windows):
            if idx == 0:
                s_rel = torch.ones(nb, device=dev, dtype=dt)
                R_rel = torch.eye(3, device=dev, dtype=dt).unsqueeze(0).expand(nb, -1, -1).clone()
                t_rel = torch.zeros(nb, 3, device=dev, dtype=dt)
            else:
                s_rel, R_rel, t_rel = self._pairwise_alignment(
                    warped_windows[-1], raw, overlap, nb, dev, dt,
                )

            per_window_scales.append(s_rel.clone())
            T = torch.eye(4, device=dev, dtype=dt).unsqueeze(0).expand(nb, -1, -1).clone()
            T[:, :3, :3] = R_rel
            T[:, :3, 3] = t_rel
            per_window_transforms.append(T)

            warped_windows.append(
                self._warp_predictions(raw, R_rel, t_rel, s_rel, nb)
            )

        merged = self._stitch_windows(warped_windows, win_sz, overlap)

        # Attach alignment metadata
        if per_window_scales:
            merged["chunk_scales"] = torch.stack(per_window_scales, dim=1)
        if per_window_transforms:
            merged["chunk_transforms"] = torch.stack(per_window_transforms, dim=1)
        merged["alignment_mode"] = "scaled"
        return merged

    @torch.no_grad()
    def inference_windowed(
        self,
        images: torch.Tensor,
        window_size: int = 16,
        overlap_size: Optional[int] = None,
        overlap_keyframes: Optional[int] = None,
        num_scale_frames: Optional[int] = None,
        scale_mode: str = 'median',
        output_device: Optional[torch.device] = None,
        keyframe_interval: int = 1,
        flow_threshold: float = 0.0,
        max_non_keyframe_gap: int = 30,
    ) -> Dict[str, torch.Tensor]:
        """
        Windowed inference with keyframe detection and cross-window alignment.

        Each window is processed independently with a fresh KV cache.
        Overlap frames between windows are the next window's scale frames
        (bidirectional attention), ensuring the highest quality predictions
        at alignment boundaries.

        ``window_size`` counts **keyframes** (frames stored in KV cache),
        including scale frames.  When ``keyframe_interval > 1``, each window
        covers more actual frames than ``window_size``:

            actual_frames = scale_frames + (window_size - scale_frames) * keyframe_interval

        Args:
            images: Input images [S, 3, H, W] or [B, S, 3, H, W] in [0, 1].
            window_size: Number of **keyframes** per window (including scale
                frames).  Directly controls KV cache memory.
            overlap_size: Number of overlapping **actual frames** between
                windows.  Defaults to ``num_scale_frames`` (overlap = scale
                frames).  Ignored when ``overlap_keyframes`` is provided.
            overlap_keyframes: Overlap expressed in **keyframes** (takes
                precedence over ``overlap_size``).  Internally converted to
                ``max(num_scale_frames, overlap_keyframes * keyframe_interval)``
                so the overlap region always spans at least that many keyframe
                intervals and is large enough to host the next window's scale
                phase.  Use this when ``keyframe_interval > 1`` to guarantee
                the pairwise alignment has at least one paired keyframe.
            num_scale_frames: Number of frames used as scale reference within
                each window.  Defaults to ``self.num_frame_for_scale``.
            scale_mode: Scale estimation strategy for alignment.
            output_device: Device to store per-window outputs.
            keyframe_interval: Every N-th Phase 2 frame is a keyframe whose
                KV persists in cache.  1 = every frame (default).
            flow_threshold: Mean flow magnitude threshold (pixels) for
                flow-based keyframe selection.  >0 enables flow-based mode
                (takes precedence over ``keyframe_interval``).
            max_non_keyframe_gap: Max consecutive non-keyframe frames before
                forcing a keyframe (flow mode only).

        Returns:
            Merged prediction dict with all frames.
        """
        use_flow_keyframe = flow_threshold > 0.0

        # Normalize input shape
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, S, C, H, W = images.shape

        # Slice-then-move per iteration so `images` may live on CPU and we
        # keep peak GPU memory at one window (not the full sequence).
        _model_device = next(self.parameters()).device

        ws = (num_scale_frames if num_scale_frames is not None
            else self.num_frame_for_scale)
        ws = min(ws, S)

        # Resolve overlap in *actual frames*.  Priority:
        #   1. overlap_keyframes (preferred; converted via keyframe_interval)
        #   2. overlap_size (legacy, already in actual frames)
        #   3. default = num_scale_frames
        # overlap_keyframes is clamped up to ``ws`` because the next window's
        # scale phase needs at least ``ws`` overlapping frames to work.
        if overlap_keyframes is not None:
            kf = max(keyframe_interval, 1)
            eff_overlap = max(ws, overlap_keyframes * kf)
        elif overlap_size is not None:
            eff_overlap = overlap_size
        else:
            eff_overlap = ws
        eff_overlap = min(eff_overlap, S - 1) if S > 1 else 0

        def _to_out(t: torch.Tensor) -> torch.Tensor:
            return t.to(output_device) if output_device is not None else t

        def _collect_frame(out, w_lists):
            w_lists['pose_enc'].append(_to_out(out["pose_enc"]))
            if "depth" in out:
                w_lists['depth'].append(_to_out(out["depth"]))
            if "depth_conf" in out:
                w_lists['depth_conf'].append(_to_out(out["depth_conf"]))
            if "world_points" in out:
                w_lists['world_points'].append(_to_out(out["world_points"]))
            if "world_points_conf" in out:
                w_lists['world_pts_conf'].append(_to_out(out["world_points_conf"]))

        def _make_window_pred(w_lists):
            pred: Dict = {"pose_enc": torch.cat(w_lists['pose_enc'], dim=1)}
            if w_lists['depth']:
                pred["depth"] = torch.cat(w_lists['depth'], dim=1)
            if w_lists['depth_conf']:
                pred["depth_conf"] = torch.cat(w_lists['depth_conf'], dim=1)
            if w_lists['world_points']:
                pred["world_points"] = torch.cat(w_lists['world_points'], dim=1)
            if w_lists['world_pts_conf']:
                pred["world_points_conf"] = torch.cat(w_lists['world_pts_conf'], dim=1)
            # Frame type: 0=scale, 1=keyframe, 2=non-keyframe
            ft = torch.tensor(w_lists['frame_type'], dtype=torch.uint8).unsqueeze(0)  # [1, T]
            pred["frame_type"] = ft
            pred["is_keyframe"] = (ft != 2)  # scale + keyframe = True
            return pred

        def _new_lists():
            return {
                'pose_enc': [], 'depth': [], 'depth_conf': [],
                'world_points': [], 'world_pts_conf': [],
                'frame_type': [],  # list of ints: 0=scale, 1=keyframe, 2=non-keyframe
            }

        # ================================================================
        # Flow-based mode: dynamic windows (can't precompute window list)
        # ================================================================
        if use_flow_keyframe:
            all_window_predictions: List[Dict] = []
            cursor = 0
            window_idx = 0
            pbar = tqdm(total=S, desc='Windowed inference (flow)', initial=0)

            while cursor < S:
                window_start = cursor
                window_scale = min(ws, S - cursor)

                # Fresh KV cache
                self.clean_kv_cache()

                # ---------- Phase 1: scale frames ----------
                scale_images = images[:, cursor:cursor + window_scale].to(
                    _model_device, non_blocking=True
                )
                scale_out = self.forward(
                    scale_images,
                    num_frame_for_scale=window_scale,
                    num_frame_per_block=window_scale,
                    causal_inference=True,
                )
                w_lists = _new_lists()
                _collect_frame(scale_out, w_lists)
                w_lists['frame_type'].extend([0] * window_scale)  # scale frames

                # Flow state: last keyframe = last scale frame
                last_kf_pose_enc = scale_out["pose_enc"][:, -1:]
                last_kf_local_idx = window_scale - 1
                del scale_out

                cursor += window_scale
                pbar.update(window_scale)

                # ---------- Phase 2: stream until enough keyframes ----------
                target_kf = window_size - window_scale  # keyframes to collect
                kf_count = 0

                while cursor < S and kf_count < target_kf:
                    frame_image = images[:, cursor:cursor + 1].to(
                        _model_device, non_blocking=True
                    )

                    self._set_defer_eviction(True)
                    frame_out = self.forward(
                        frame_image,
                        num_frame_for_scale=window_scale,
                        num_frame_per_block=1,
                        causal_inference=True,
                    )
                    self._set_defer_eviction(False)

                    # Compute flow
                    cur_depth = frame_out.get("depth", None)
                    if cur_depth is not None:
                        H_pred, W_pred = cur_depth.shape[2], cur_depth.shape[3]
                        flow_mag = _compute_flow_magnitude(
                            frame_out["pose_enc"], last_kf_pose_enc,
                            cur_depth, (H_pred, W_pred),
                        )
                    else:
                        flow_mag = flow_threshold + 1.0

                    local_idx = window_scale + (cursor - window_start - window_scale)
                    frames_since_kf = local_idx - last_kf_local_idx
                    is_keyframe = (
                        (kf_count == 0)  # first streaming frame
                        or (flow_mag > flow_threshold)
                        or (frames_since_kf >= max_non_keyframe_gap)
                    )

                    if is_keyframe:
                        self._execute_deferred_eviction()
                        last_kf_pose_enc = frame_out["pose_enc"]
                        last_kf_local_idx = local_idx
                        kf_count += 1
                        w_lists['frame_type'].append(1)  # keyframe
                    else:
                        self._rollback_last_frame()
                        w_lists['frame_type'].append(2)  # non-keyframe

                    _collect_frame(frame_out, w_lists)
                    del frame_out
                    cursor += 1
                    pbar.update(1)

                all_window_predictions.append(_make_window_pred(w_lists))
                window_idx += 1

                # Next window starts overlap_size frames back (= scale frames)
                if cursor < S:
                    cursor = max(cursor - eff_overlap, window_start + window_scale)

            pbar.close()

        # ================================================================
        # Fixed-interval / default mode: precomputable windows
        # ================================================================
        else:
            # Compute actual frames per window
            phase2_kf = max(window_size - ws, 0)
            kf_int = max(keyframe_interval, 1)
            phase2_frames = phase2_kf * kf_int
            actual_window_frames = ws + phase2_frames

            eff_window = min(actual_window_frames, S)
            step = max(eff_window - eff_overlap, 1)

            # Build window list
            if eff_window >= S:
                windows = [(0, S)]
            else:
                windows = []
                for start_idx in range(0, S, step):
                    end_idx = min(start_idx + eff_window, S)
                    if end_idx - start_idx >= eff_overlap or end_idx == S:
                        windows.append((start_idx, end_idx))
                    if end_idx == S:
                        break

            all_window_predictions: List[Dict] = []
            for start, end in tqdm(windows, desc='Windowed inference'):
                # Slice on whichever device `images` lives on, then move just
                # this window to the model device.  Keeps peak memory at one
                # window (O(actual_window_frames)) instead of full sequence.
                window_images = images[:, start:end].to(
                    _model_device, non_blocking=True
                )
                window_len = end - start

                # Fresh KV cache
                self.clean_kv_cache()

                window_scale = min(ws, window_len)

                # ---------- Phase 1: scale frames ----------
                scale_out = self.forward(
                    window_images[:, :window_scale],
                    num_frame_for_scale=window_scale,
                    num_frame_per_block=window_scale,
                    causal_inference=True,
                )
                w_lists = _new_lists()
                _collect_frame(scale_out, w_lists)
                w_lists['frame_type'].extend([0] * window_scale)  # scale frames
                del scale_out

                # ---------- Phase 2: stream remaining frames ----------
                dbg_every = _parse_kv_debug_interval(_KV_DEBUG)
                if dbg_every:
                    _log_kv_stats(self, label=f"w{start}-{end} after phase-1 scale")

                for i in range(window_scale, window_len):
                    is_keyframe = (
                        kf_int <= 1
                        or ((i - window_scale) % kf_int == 0)
                    )

                    if not is_keyframe:
                        self._set_skip_append(True)

                    frame_out = self.forward(
                        window_images[:, i:i + 1],
                        num_frame_for_scale=window_scale,
                        num_frame_per_block=1,
                        causal_inference=True,
                    )

                    if not is_keyframe:
                        self._set_skip_append(False)

                    _collect_frame(frame_out, w_lists)
                    w_lists['frame_type'].append(1 if is_keyframe else 2)
                    del frame_out

                    if dbg_every and ((i - window_scale) % dbg_every == 0):
                        _log_kv_stats(
                            self,
                            label=f"w{start}-{end} i={i} (global={start + i}) "
                                f"{'KF' if is_keyframe else 'nkf'}",
                        )

                all_window_predictions.append(_make_window_pred(w_lists))
                if dbg_every:
                    _log_kv_stats(self, label=f"w{start}-{end} END")

        # Store for merge helpers
        self._last_window_size = eff_overlap  # not used directly, but kept for compat
        self._last_overlap_size = eff_overlap

        # Align and stitch windows
        predictions = self._align_and_stitch_windows(
            all_window_predictions, scale_mode=scale_mode
        )

        predictions["images"] = _to_out(images)

        if self.pred_normalization:
            predictions = self._normalize_predictions(predictions)

        return predictions
