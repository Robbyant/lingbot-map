"""
AggregatorBase - Base class for all Aggregator implementations.

Provides shared functionality:
- Patch embedding (DINOv2)
- Special tokens (camera, register, scale)
- Block building
- Common forward pass structure

Subclasses implement mode-specific attention logic.
"""

import logging
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from lingbot_map.layers import PatchEmbed
from lingbot_map.layers.block import Block
from lingbot_map.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from lingbot_map.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def slice_expand_and_flatten(token, B, S, first_num_frame=1):
    """
    Helper function to slice, expand and flatten tokens.

    Args:
        token: Token tensor [1, 2, N, C] where first index is for first frames
        B: Batch size
        S: Sequence length
        first_num_frame: Number of frames to use first token for

    Returns:
        Flattened tokens [B*S, N, C]
    """
    # token shape: [1, 2, N, C]
    # Expand to [B, S, N, C]
    if first_num_frame > 1:
        # Use first token for first first_num_frame frames, second for rest
        token_first = token[:, :1].expand(B, first_num_frame, -1, -1)  # [B, first_num_frame, N, C]
        token_rest = token[:, 1:].expand(B, S - first_num_frame, -1, -1)  # [B, S-first_num_frame, N, C]
        token_expanded = torch.cat([token_first, token_rest], dim=1)  # [B, S, N, C]
    else:
        # Use first token for first frame, second for rest
        token_first = token[:, :1].expand(B, 1, -1, -1)  # [B, 1, N, C]
        token_rest = token[:, 1:].expand(B, S - 1, -1, -1)  # [B, S-1, N, C]
        token_expanded = torch.cat([token_first, token_rest], dim=1)  # [B, S, N, C]

    # Flatten to [B*S, N, C]
    return token_expanded.reshape(B * S, -1, token.shape[-1])


class AggregatorBase(nn.Module, ABC):
    """
    Base class for all Aggregator implementations.

    Handles shared components:
    - Patch embedding (DINOv2 or conv)
    - Special tokens (camera, register, optionally scale)
    - Block creation (frame + global)
    - RoPE (2D rotary position embeddings)
    - Common forward pass scaffolding

    Subclasses must implement:
    - _process_global_attention(): Mode-specific cross-frame attention logic
    """

    def __init__(
        self,
        # Architecture parameters
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        # Block configuration
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        init_values=0.01,
        # Patch embedding
        patch_embed="dinov2_vitl14_reg",
        pretrained_path=None,
        # Attention pattern
        aa_order=["frame", "global"],
        aa_block_size=1,
        # RoPE
        rope_freq=100,
        disable_global_rope=False,
        # Gradient checkpointing
        use_reentrant: bool = False,
        use_gradient_checkpoint: bool = True,
    ):
        super().__init__()

        # Store configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_register_tokens = num_register_tokens
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.disable_global_rope = disable_global_rope
        self.use_reentrant = use_reentrant
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.pretrained_path = pretrained_path

        print("pretrained_path:", self.pretrained_path)

        # Validate depth
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")
        self.aa_block_num = self.depth // self.aa_block_size

        # Build patch embedding
        self._build_patch_embed(
            patch_embed=patch_embed,
            img_size=img_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            embed_dim=embed_dim,
            pretrained_path=pretrained_path
        )

        # Initialize RoPE
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # Build blocks (frame + global)
        self._build_blocks(
            block_fn=block_fn,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            init_values=init_values,
            qk_norm=qk_norm,
        )

        # Setup special tokens (camera, register, optionally scale)
        self._setup_special_tokens()

        # Register normalization constants
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        # Initialize from DINO checkpoint if available
        if hasattr(self, '_dino_checkpoint') and self._dino_checkpoint is not None:
            self._init_blocks_from_dino(self._dino_checkpoint)
            del self._dino_checkpoint  # Free memory

    def _build_patch_embed(
        self,
        patch_embed: str,
        img_size: int,
        patch_size: int,
        num_register_tokens: int,
        embed_dim: int,
        pretrained_path: str,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
    ):
        """
        Build patch embedding layer.

        Supports:
        - "conv": Simple convolutional patch embedding
        - "dinov2_*": DINOv2 ViT variants (vitl14, vitb14, vits14, vitg2)
        """
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim
            )
            self._dino_checkpoint = None

        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            if patch_embed not in vit_models:
                raise NotImplementedError(f"Unknown patch_embed type: {patch_embed}")

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Load pretrained weights
            try:
                ckpt = torch.load(pretrained_path)
                del ckpt['pos_embed']
                logger.info("Loading pretrained weights for DINOv2")
                missing, unexpected = self.patch_embed.load_state_dict(ckpt, strict=False)
                logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

                # Store checkpoint for block initialization
                self._dino_checkpoint = ckpt
            except Exception as e:
                logger.warning(f"Failed to load pretrained weights: {e}")
                self._dino_checkpoint = None

            # Disable gradients for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    @abstractmethod
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
        """
        Build frame_blocks and global_blocks.

        Subclasses implement mode-specific block creation.

        Must create:
        - self.frame_blocks: nn.ModuleList of frame attention blocks
        - self.global_blocks: nn.ModuleList of global attention blocks
        """
        pass

    @abstractmethod
    def _setup_special_tokens(self):
        """
        Setup camera token, register tokens, and optionally scale token.

        Subclasses implement mode-specific token initialization.

        Must create:
        - self.camera_token
        - self.register_token
        - self.scale_token (optional, for causal mode)
        - self.patch_start_idx
        - self.num_special_tokens
        """
        pass

    def _init_blocks_from_dino(self, dino_ckpt: dict):
        """
        Initialize frame_blocks and global_blocks from DINOv2 pretrained weights.

        Args:
            dino_ckpt: Checkpoint dictionary from DINOv2 model
        """
        logger.info("Initializing blocks from DINOv2 pretrained weights")

        # Extract block keys
        dino_block_keys = [k for k in dino_ckpt.keys() if k.startswith('blocks.')]
        if not dino_block_keys:
            logger.warning("No 'blocks' found in DINO checkpoint")
            return

        # Get block indices
        block_indices = set()
        for key in dino_block_keys:
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.add(int(parts[1]))

        num_dino_blocks = len(block_indices)
        print(f"Found {num_dino_blocks} blocks in DINO checkpoint")

        # Initialize frame_blocks
        for i, block in enumerate(self.frame_blocks):
            dino_block_idx = i % num_dino_blocks
            block_state_dict = {}
            prefix = f'blocks.{dino_block_idx}.'
            for key, value in dino_ckpt.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    block_state_dict[new_key] = value

            if block_state_dict:
                missing, unexpected = block.load_state_dict(block_state_dict, strict=False)
                if i == 0:  # Only log for first block to avoid spam
                    print(f"Frame block 0: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        # Initialize global_blocks
        for i, block in enumerate(self.global_blocks):
            dino_block_idx = i % num_dino_blocks
            block_state_dict = {}
            prefix = f'blocks.{dino_block_idx}.'
            for key, value in dino_ckpt.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    block_state_dict[new_key] = value

            if block_state_dict:
                missing, unexpected = block.load_state_dict(block_state_dict, strict=False)
                if i == 0:  # Only log for first block to avoid spam
                    print(f"Global block 0: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        logger.info("Successfully initialized blocks from DINOv2 weights")

    def _embed_images(
        self,
        images: torch.Tensor,
        num_frame_for_scale: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, int, int, int, int]:
        """
        Embed images and prepare for attention processing.

        Handles:
        - Image normalization
        - Patch embedding
        - Special token concatenation
        - Position embedding

        Args:
            images: Input images [B, S, 3, H, W] in range [0, 1]
            num_frame_for_scale: Number of frames for scale estimation (passed to special tokens)

        Returns:
            (tokens, B, S, S, P, C):
                tokens: Embedded tokens [B*S, P, C]
                B: Batch size
                S: Sequence length
                S: Same as above (no CP slicing)
                P: Number of tokens per frame (patches + special tokens)
                C: Embedding dimension
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images
        images = (images - self._resnet_mean) / self._resnet_std

        # No CP slicing: S_local == S_global
        S_local = S
        S_global = S

        # Reshape for patch embedding [B*S, C, H, W]
        images = images.view(B * S, C_in, H, W)

        # Patch embedding
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P_patch, C = patch_tokens.shape

        # Prepare special tokens
        special_tokens = self._prepare_special_tokens(
            B, S_local, S_global, C,
            num_frame_for_scale=num_frame_for_scale
        )

        # Concatenate special tokens + patch tokens
        tokens = torch.cat([special_tokens, patch_tokens], dim=1)

        _, P, C = tokens.shape

        return tokens, B, S_local, S_global, P, C

    @abstractmethod
    def _prepare_special_tokens(self, B: int, S_local: int, S_global: int, C: int, **kwargs) -> torch.Tensor:
        """
        Prepare special tokens (camera, register, optionally scale).

        Subclasses implement mode-specific token preparation.

        Args:
            B: Batch size
            S_local: Local sequence length
            S_global: Global sequence length
            C: Embedding dimension
            **kwargs: Mode-specific parameters (e.g., num_frame_for_scale for causal mode)

        Returns:
            Special tokens [B*S, N_special, C]
        """
        pass

    def _get_positions(self, B: int, S: int, H: int, W: int, device) -> Optional[torch.Tensor]:
        """
        Get 2D position embeddings for RoPE.

        Args:
            B: Batch size
            S: Sequence length
            H: Image height
            W: Image width
            device: Device to create positions on

        Returns:
            Position tensor [B*S, P, 2] or None if rope is disabled
        """
        if self.rope is None:
            return None

        # Get patch positions
        pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=device)

        # Add offset for patch tokens (skip special tokens at pos=0)
        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, dtype=pos.dtype, device=device)
            pos = torch.cat([pos_special, pos], dim=1)

        return pos

    def _process_frame_attention(
        self,
        tokens: torch.Tensor,
        B: int,
        S: int,
        P: int,
        C: int,
        frame_idx: int,
        pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """
        Process frame attention blocks.

        Frame attention operates independently per frame (no cross-frame communication).
        Tokens stay in shape [B*S, P, C].

        Args:
            tokens: Input tokens [B*S, P, C]
            B: Batch size
            S: Sequence length
            P: Tokens per frame
            C: Embedding dimension
            frame_idx: Current frame block index
            pos: Position embeddings [B*S, P, 2]

        Returns:
            (tokens, frame_idx, intermediates):
                tokens: Output tokens [B*S, P, C]
                frame_idx: Updated frame block index
                intermediates: List of intermediate outputs [B, S, P, C]
        """
        # Ensure correct shape
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B * S, P, 2)

        intermediates = []

        # Process blocks
        for i in range(self.aa_block_size):
            if self.training and self.use_gradient_checkpoint:
                from torch.utils.checkpoint import checkpoint
                tokens = checkpoint(
                    self.frame_blocks[frame_idx],
                    tokens,
                    pos,
                    use_reentrant=self.use_reentrant
                )
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)

            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    @abstractmethod
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
        **kwargs
    ) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """
        Process global (cross-frame) attention blocks.

        Subclasses implement mode-specific attention logic.

        Args:
            tokens: Input tokens
            B: Batch size
            S_local: Local sequence length
            S_global: Global sequence length
            P: Tokens per frame
            C: Embedding dimension
            global_idx: Current global block index
            pos: Position embeddings
            **kwargs: Mode-specific parameters

        Returns:
            (tokens, global_idx, intermediates):
                tokens: Output tokens
                global_idx: Updated global block index
                intermediates: List of intermediate outputs
        """
        pass

    def forward(
        self,
        images: torch.Tensor,
        selected_idx: Optional[List[int]] = None,
        # Mode-specific parameters
        num_frame_for_scale: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
        num_frame_per_block: int = 1,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Forward pass.

        Args:
            images: Input images [B, S, 3, H, W] in range [0, 1]
            selected_idx: Which block indices to output (None = all)
            num_frame_for_scale: Number of frames for scale estimation (causal mode)
            sliding_window_size: Sliding window size in blocks (causal mode)
            num_frame_per_block: Number of frames per processing block (causal mode)

        Returns:
            (output_list, patch_start_idx):
                output_list: List of block outputs [B, S, P, 2C]
                patch_start_idx: Index where patch tokens start
        """
        B, S_input, _, H, W = images.shape

        # Embed images
        tokens, B, S_local, S_global, P, C = self._embed_images(
            images,
            num_frame_for_scale=num_frame_for_scale,
        )

        # Get position embeddings
        pos_local = self._get_positions(B, S_local, H, W, device=images.device)
        pos_global = self._get_positions(B, S_global, H, W, device=images.device)

        # Alternating attention
        frame_idx = 0
        global_idx = 0
        output_list = []

        for block_group_idx in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S_local, P, C, frame_idx, pos=pos_local
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S_local, S_global, P, C, global_idx,
                        pos=pos_global,
                        num_frame_for_scale=num_frame_for_scale,
                        sliding_window_size=sliding_window_size,
                        num_frame_per_block=num_frame_per_block,
                        image_height=H,
                        image_width=W,
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            # Collect outputs
            if selected_idx is None or block_group_idx in selected_idx:
                for i in range(len(frame_intermediates)):
                    # Concatenate frame and global intermediates [B, S, P, 2C]
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    output_list.append(concat_inter)

        return output_list, self.patch_start_idx
