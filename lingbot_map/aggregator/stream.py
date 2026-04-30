"""
AggregatorStream - Streaming causal aggregator with FlashInfer KV cache.

Provides:
- Temporal causal attention
- Sliding window support
- Scale token for scale estimation frames
- Streaming inference with FlashInfer paged KV cache
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from lingbot_map.layers.block import Block, FlashInferBlock, SDPABlock
from lingbot_map.layers.rope import WanRotaryPosEmbed
from lingbot_map.aggregator.base import AggregatorBase, slice_expand_and_flatten

logger = logging.getLogger(__name__)


class AggregatorStream(AggregatorBase):
    """
    Streaming causal aggregator with FlashInfer paged KV cache.

    Features:
    - Temporal causal attention (each frame only attends to past frames)
    - Sliding window support to limit attention scope
    - Scale token for scale estimation frames
    - Streaming inference with FlashInfer KV cache
    """

    def __init__(
        self,
        # Causal-specific parameters
        sliding_window_size: int = -1,
        num_frame_for_scale: int = 1,
        num_random_frames: int = 0,
        attend_to_special_tokens: bool = False,
        attend_to_scale_frames: bool = False,
        enable_3d_rope: bool = False,
        max_frame_num: int = 1024,
        # KV cache parameters
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
        # Base class parameters via **kwargs
        **kwargs
    ):
        """
        Initialize AggregatorStream.

        Args:
            sliding_window_size: Sliding window size in blocks (-1 for full causal)
            num_frame_for_scale: Number of scale estimation frames
            num_random_frames: Number of random frames for long-range dependencies
            attend_to_special_tokens: Enable cross-frame special token attention
            attend_to_scale_frames: Include scale frames in attention
            enable_3d_rope: Enable 3D RoPE for temporal dimension in KV cache
            max_frame_num: Maximum number of frames for 3D RoPE
            kv_cache_sliding_window: Sliding window size for KV cache eviction
            kv_cache_scale_frames: Number of scale frames to keep in KV cache
            kv_cache_cross_frame_special: Keep special tokens from evicted frames
            kv_cache_include_scale_frames: Include scale frames in KV cache
            kv_cache_camera_only: Only keep camera tokens from evicted frames
            **kwargs: Base class parameters
        """
        self.sliding_window_size = sliding_window_size
        self.num_frame_for_scale = num_frame_for_scale
        self.num_random_frames = num_random_frames
        self.attend_to_special_tokens = attend_to_special_tokens
        self.attend_to_scale_frames = attend_to_scale_frames
        self.enable_3d_rope = enable_3d_rope
        self.max_frame_num = max_frame_num
        # KV cache parameters
        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.kv_cache_scale_frames = kv_cache_scale_frames
        self.kv_cache_cross_frame_special = kv_cache_cross_frame_special
        self.kv_cache_include_scale_frames = kv_cache_include_scale_frames
        self.kv_cache_camera_only = kv_cache_camera_only

        # Pop kwargs that are passed but not needed by base class
        kwargs.pop('enable_stream_inference', None)
        use_flashinfer = kwargs.pop('use_flashinfer', True)
        kwargs.pop('use_flexflash', None)
        use_sdpa = kwargs.pop('use_sdpa', False)

        # Backend selection: SDPA (no extra deps) or FlashInfer (paged KV cache)
        self.use_sdpa = use_sdpa
        self.use_flashinfer = not use_sdpa  # FlashInfer is default unless SDPA requested

        # Call parent __init__
        super().__init__(**kwargs)

        # Initialize KV cache
        self._init_kv_cache()

        # Initialize 3D RoPE if enabled
        if self.enable_3d_rope:
            self._init_3d_rope()

    def _build_blocks(
        self,
        block_fn,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        proj_bias: bool,
        ffn_bias: bool,
        init_values: float,
        qk_norm: bool,
    ):
        """Build frame and global blocks for streaming causal mode."""
        block_params = dict(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            init_values=init_values,
            qk_norm=qk_norm,
        )

        # Frame blocks: Standard Block + RoPE
        self.frame_blocks = nn.ModuleList([
            block_fn(**block_params, rope=self.rope)
            for _ in range(depth)
        ])

        # Global blocks: FlashInferBlock (default) or SDPABlock (fallback)
        GlobalBlockCls = SDPABlock if self.use_sdpa else FlashInferBlock
        self.global_blocks = nn.ModuleList([
            GlobalBlockCls(
                **block_params,
                rope=self.rope if not self.disable_global_rope else None,
                kv_cache_sliding_window=self.kv_cache_sliding_window,
                kv_cache_scale_frames=self.kv_cache_scale_frames,
                kv_cache_cross_frame_special=self.kv_cache_cross_frame_special,
                kv_cache_include_scale_frames=self.kv_cache_include_scale_frames,
                kv_cache_camera_only=self.kv_cache_camera_only,
            )
            for _ in range(depth)
        ])

    def _setup_special_tokens(self):
        """Setup camera, register, and scale tokens for causal mode."""
        # Camera token
        self.camera_token = nn.Parameter(
            torch.randn(1, 2, 1, self.embed_dim)
        )

        # Register tokens
        if self.num_register_tokens > 0:
            self.register_token = nn.Parameter(
                torch.randn(1, 2, self.num_register_tokens, self.embed_dim)
            )

        # Scale token (causal mode specific)
        self.scale_token = nn.Parameter(
            torch.ones(1, 2, 1, self.embed_dim)
        )

        # Initialize
        nn.init.normal_(self.camera_token, std=1e-6)
        if self.num_register_tokens > 0:
            nn.init.normal_(self.register_token, std=1e-6)
        nn.init.normal_(self.scale_token, std=1e-6)

        # Token indexing (includes scale token)
        self.patch_start_idx = 1 + self.num_register_tokens + 1  # camera + register + scale
        self.num_special_tokens = 1 + self.num_register_tokens + 1

    def _init_kv_cache(self):
        """Initialize KV cache for streaming inference."""
        self.kv_cache_manager = None  # FlashInfer (lazy-initialized)
        self.kv_cache = {}  # Dict-based cache for SDPA
        self.total_frames_processed = 0
        self._cached_pos3d = None

        if self.use_sdpa:
            # Dict-based KV cache for SDPA
            if hasattr(self, 'depth'):
                for i in range(self.depth):
                    self.kv_cache[f"k_{i}"] = None
                    self.kv_cache[f"v_{i}"] = None
                    self.kv_cache[f"k_{i}_special"] = None
                    self.kv_cache[f"v_{i}_special"] = None
                logger.info(f"SDPA KV cache initialized with {self.depth} blocks")
        else:
            logger.info("FlashInfer KV cache will be lazily initialized on first forward")

    def _get_flashinfer_manager(self, device, dtype, tokens_per_frame=None):
        """Lazily initialize FlashInferKVCacheManager on first use.

        Args:
            device: Device for cache tensors.
            dtype: Data type for cache tensors.
            tokens_per_frame: Actual number of tokens per frame (patches + specials).
                If None, falls back to assuming square images of self.img_size.
        """
        if self.kv_cache_manager is None:
            from lingbot_map.layers.flashinfer_cache import FlashInferKVCacheManager
            num_heads = self.embed_dim // 64  # head_dim = 64 for ViT-L
            head_dim = 64
            if tokens_per_frame is None:
                tokens_per_frame = (self.img_size // self.patch_size) ** 2 + self.num_special_tokens
            # max_num_frames: scale + window + headroom
            max_num_frames = self.kv_cache_scale_frames + self.kv_cache_sliding_window + 16
            self.kv_cache_manager = FlashInferKVCacheManager(
                num_blocks=self.depth,
                max_num_frames=max_num_frames,
                tokens_per_frame=tokens_per_frame,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                num_special_tokens=self.num_special_tokens,
                scale_frames=self.kv_cache_scale_frames,
                sliding_window=self.kv_cache_sliding_window,
                max_total_frames=self.max_frame_num + 100,
                force_fp32=getattr(self, 'kv_cache_force_fp32', False),
                fa3=getattr(self, 'kv_cache_fa3', False),
            )
            logger.info(
                f"FlashInfer KV cache manager initialized: {self.depth} blocks, "
                f"max_frames={max_num_frames}, tokens_per_frame={tokens_per_frame}"
            )
        return self.kv_cache_manager

    def clean_kv_cache(self):
        """Clean KV cache (call this when starting a new sequence)."""
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.reset()
        if self.kv_cache:
            for key in list(self.kv_cache.keys()):
                if key == "_skip_append":
                    self.kv_cache[key] = False
                else:
                    self.kv_cache[key] = None
        self.total_frames_processed = 0
        self._cached_pos3d = None
        logger.info("KV cache cleaned")

    def _init_3d_rope(self):
        """Initialize 3D RoPE for streaming inference."""
        if not self.enable_3d_rope:
            self.rope3d = None
            return

        num_heads = 16
        head_dim = self.embed_dim // num_heads

        self.rope3d = WanRotaryPosEmbed(
            attention_head_dim=head_dim,
            patch_size=(1, self.patch_size, self.patch_size),
            max_seq_len=self.max_frame_num,
        )
        logger.info(f"3D RoPE initialized for max {self.max_frame_num} frames, head_dim={head_dim}")

    def _get_3d_positions_streaming(self, num_frames, H, W, device, f_start, f_end):
        """
        Generate 3D RoPE positions for streaming mode with correct global frame indices.

        Args:
            num_frames: Number of frames in current batch
            H, W: Image height and width
            device: Device to create positions on
            f_start: Global start frame index
            f_end: Global end frame index

        Returns:
            pos3d: [1, 1, num_frames * P, head_dim//2] complex tensor
        """
        if self.rope3d is None:
            return None

        pph = H // self.patch_size
        ppw = W // self.patch_size

        pos3d = self.rope3d(
            ppf=num_frames,
            pph=pph,
            ppw=ppw,
            patch_start_idx=self.num_special_tokens,
            device=device,
            f_start=f_start,
            f_end=f_end
        )
        return pos3d

    def _prepare_special_tokens(
        self,
        B: int,
        S_local: int,
        S_global: int,
        C: int,
        num_frame_for_scale: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Prepare camera, register, and scale tokens.

        Args:
            B: Batch size
            S_local: Local sequence length
            S_global: Global sequence length
            C: Embedding dimension
            num_frame_for_scale: Number of frames for scale estimation

        Returns:
            Special tokens [B*S_global, N_special, C]
        """
        # Get effective num_frame_for_scale
        scale_frames = self.num_frame_for_scale if num_frame_for_scale is None else num_frame_for_scale

        # Check cache state for both backends
        has_flashinfer_cache = self.kv_cache_manager is not None and self.kv_cache_manager.num_frames > 0
        has_sdpa_cache = self.kv_cache is not None and self.kv_cache.get("k_0") is not None

        # Determine if we're in causal inference mode based on KV cache state
        causal_inference = True

        if causal_inference and has_flashinfer_cache:
            S_cached = self.kv_cache_manager.num_frames
            S_true = S_cached + S_global
        elif causal_inference and has_sdpa_cache:
            _, _, S_cached, _, _ = self.kv_cache["k_0"].shape
            S_true = S_cached + S_global
        else:
            S_true = S_global

        # Expand tokens based on mode
        if causal_inference and S_true > S_global:
            # Streaming mode: expand with S_true, then slice to get current frames
            effective_scale_frames = min(scale_frames, S_true)

            camera_token_full = slice_expand_and_flatten(self.camera_token, B, S_true)
            camera_token = camera_token_full[-S_global:, :, :]

            register_token_full = slice_expand_and_flatten(self.register_token, B, S_true)
            register_token = register_token_full[-S_global:, :, :]
            scale_token_full = slice_expand_and_flatten(
                self.scale_token, B, S_true, first_num_frame=effective_scale_frames
            )
            scale_token = scale_token_full[-S_global:, :, :]
        else:
            # Batch mode or first inference: expand directly
            effective_scale_frames = min(scale_frames, S_global)

            camera_token = slice_expand_and_flatten(self.camera_token, B, S_global)
            register_token = slice_expand_and_flatten(self.register_token, B, S_global)
            scale_token = slice_expand_and_flatten(
                self.scale_token, B, S_global, first_num_frame=effective_scale_frames
            )

        special_tokens = torch.cat([camera_token, register_token, scale_token], dim=1)

        # Verify shape
        expected_shape = (B * S_global, self.num_special_tokens, C)
        assert special_tokens.shape == expected_shape, \
            f"Expected {expected_shape}, got {special_tokens.shape}"

        return special_tokens

    def _process_global_attention(
        self,
        tokens: torch.Tensor,
        B: int,
        S_local: int,
        S_global: int,
        P: int,
        C: int,
        global_idx: int,
        pos: Optional[torch.Tensor] = None,
        # Mode-specific parameters
        num_frame_for_scale: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
        num_frame_per_block: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """
        Process causal global attention via FlashInfer streaming path.

        Args:
            tokens: Input tokens
            B: Batch size
            S_local: Local sequence length
            S_global: Global sequence length
            P: Tokens per frame
            C: Embedding dimension
            global_idx: Current global block index
            pos: Position embeddings
            num_frame_for_scale: Number of frames for scale estimation
            sliding_window_size: Sliding window size in blocks
            num_frame_per_block: Number of frames per processing block

        Returns:
            (tokens, global_idx, intermediates)
        """
        # Extract image dimensions from kwargs for 3D RoPE
        image_height = kwargs.get('image_height', self.img_size)
        image_width = kwargs.get('image_width', self.img_size)

        return self._process_causal_stream(
            tokens, B, S_local, S_global, P, C, global_idx, pos,
            num_frame_per_block, sliding_window_size, num_frame_for_scale,
            image_height=image_height, image_width=image_width
        )

    def _process_causal_stream(
        self,
        tokens: torch.Tensor,
        B: int,
        S_local: int,
        S_global: int,
        P: int,
        C: int,
        global_idx: int,
        pos: Optional[torch.Tensor] = None,
        num_frame_per_block: int = 1,
        sliding_window_size: Optional[int] = None,
        num_frame_for_scale: Optional[int] = None,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
    ):
        """
        Causal attention for streaming inference using FlashInfer KV cache.

        Args:
            tokens: Input tokens [B*S_local, P, C]
            B: Batch size
            S_local: Local sequence length
            S_global: Global sequence length
            P: Number of patches per frame (includes special tokens)
            C: Channel dimension
            global_idx: Starting block index
            pos: Position embeddings [B*S_global, P, 2]
            num_frame_per_block: Number of frames per block
            sliding_window_size: Sliding window size in blocks
            num_frame_for_scale: Number of scale frames
            image_height: Image height for 3D RoPE calculation
            image_width: Image width for 3D RoPE calculation

        Returns:
            (tokens, global_idx, intermediates): Updated tokens, next block index, intermediate outputs
        """
        # Get effective parameters
        scale_frames = num_frame_for_scale if num_frame_for_scale is not None else self.num_frame_for_scale

        # Reshape tokens: [B*S_local, P, C] -> [B, S_local*P, C]
        if tokens.shape != (B, S_local * P, C):
            tokens = tokens.view(B, S_local, P, C).view(B, S_local * P, C)

        # Calculate number of frames for block mask
        num_frames = S_global
        num_patches = P - self.num_special_tokens

        # Check if this is the first block group
        is_first_block_group = (global_idx < self.aa_block_size)

        if self.enable_3d_rope and self.rope3d is not None:
            if is_first_block_group:
                f_start = self.total_frames_processed
                f_end = self.total_frames_processed + S_global

                H = image_height if image_height is not None else self.img_size
                W = image_width if image_width is not None else self.img_size
                pos3d = self._get_3d_positions_streaming(
                    S_global, H, W, tokens.device, f_start, f_end
                )
                self._cached_pos3d = pos3d
            else:
                pos3d = self._cached_pos3d
            pos = pos3d
        else:
            # Reshape pos: [B*S_global, P, 2] -> [B, S_global*P, 2]
            if pos is not None and pos.shape != (B, S_global * P, 2):
                pos = pos.view(B, S_global, P, 2).view(B, S_global * P, 2)

        intermediates = []

        # Process blocks with KV cache
        for _ in range(self.aa_block_size):
            num_patches = P - self.num_special_tokens
            if self.use_sdpa:
                # SDPA: dict-based KV cache
                tokens = self.global_blocks[global_idx](
                    tokens,
                    pos=pos,
                    num_patches=num_patches,
                    num_special=self.num_special_tokens,
                    num_frames=num_frames,
                    enable_3d_rope=self.enable_3d_rope,
                    kv_cache=self.kv_cache,
                    global_idx=global_idx,
                    num_frame_per_block=num_frame_per_block,
                    num_frame_for_scale=scale_frames,
                    num_register_tokens=self.num_register_tokens,
                )
            else:
                # FlashInfer: paged KV cache manager
                manager = self._get_flashinfer_manager(tokens.device, tokens.dtype, tokens_per_frame=P)
                tokens = self.global_blocks[global_idx](
                    tokens,
                    pos=pos,
                    num_patches=num_patches,
                    num_special=self.num_special_tokens,
                    num_frames=num_frames,
                    enable_3d_rope=self.enable_3d_rope,
                    kv_cache=manager,
                    global_idx=global_idx,
                    num_frame_per_block=num_frame_per_block,
                    num_frame_for_scale=scale_frames,
                    num_register_tokens=self.num_register_tokens,
                )

            global_idx += 1
            intermediates.append(tokens.view(B, S_local, P, C))

        # Update total frames processed counter only on the first block group
        if is_first_block_group and not (isinstance(self.kv_cache, dict) and self.kv_cache.get("_skip_append", False)):
            self.total_frames_processed += S_global

        return tokens, global_idx, intermediates
