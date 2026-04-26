"""
MLX prediction heads.

CameraHeadMLX — iterative pose refinement with KV cache.
DPTHeadMLX    — dense prediction (depth / world points) via DPT architecture.

Both mirror their PyTorch equivalents in lingbot_map/heads/.
"""

from typing import Optional, List, Tuple, Dict, Any
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .layers import LayerNorm, MLP, StreamingBlock


# ---------------------------------------------------------------------------
# Activations (mirrors head_act.py)
# ---------------------------------------------------------------------------

def activate_pose(pred: mx.array,
                  trans_act: str = "linear",
                  quat_act: str = "linear",
                  fl_act: str = "relu") -> mx.array:
    T    = _base_act(pred[..., :3], trans_act)
    quat = _base_act(pred[..., 3:7], quat_act)
    fl   = _base_act(pred[..., 7:],  fl_act)
    return mx.concatenate([T, quat, fl], axis=-1)


def _base_act(x: mx.array, act_type: str) -> mx.array:
    if act_type == "linear":
        return x
    elif act_type == "inv_log":
        return mx.sign(x) * (mx.expm1(mx.abs(x)))
    elif act_type == "exp":
        return mx.exp(x)
    elif act_type == "relu":
        return mx.maximum(x, 0)
    raise ValueError(f"Unknown act_type: {act_type}")


def activate_head(out: mx.array,
                  activation: str = "inv_log",
                  conf_activation: str = "expp1") -> Tuple[mx.array, mx.array]:
    """out: [B, H, W, C] (channel-last) → (pts3d, conf)."""
    # out layout: [..., :-1] are xyz-like values, [..., -1] is confidence logit
    xyz  = out[..., :-1]
    conf = out[..., -1]

    if activation == "inv_log":
        pts3d = mx.sign(xyz) * mx.expm1(mx.abs(xyz))
    elif activation == "exp":
        pts3d = mx.exp(xyz)
    elif activation == "norm_exp":
        d = mx.sqrt(mx.sum(xyz * xyz, axis=-1, keepdims=True)).clip(1e-8)
        pts3d = (xyz / d) * mx.expm1(d)
    elif activation == "relu":
        pts3d = mx.maximum(xyz, 0)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if conf_activation == "expp1":
        conf_out = 1 + mx.exp(conf)
    elif conf_activation == "expp0":
        conf_out = mx.exp(conf)
    elif conf_activation == "sigmoid":
        conf_out = mx.sigmoid(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out


# ---------------------------------------------------------------------------
# Camera head
# ---------------------------------------------------------------------------

def _modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1 + scale) + shift


class CameraHeadMLX(nn.Module):
    """Iterative camera pose refinement head (causal streaming version).

    Corresponds to CameraCausalHead in lingbot_map/heads/camera_head.py.

    The trunk is a list of StreamingBlock layers so the same KV-cache
    eviction logic from the aggregator global blocks applies here too.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        init_values: float = 0.01,
        num_iterations: int = 4,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_keep_special: bool = True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.trunk_depth = trunk_depth
        self.num_iterations = num_iterations
        self.target_dim = 9   # absT_quaR_FoV

        # Trunk: causal transformer blocks (same eviction policy as aggregator)
        self.trunk = [
            StreamingBlock(
                dim_in, num_heads,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
                sliding_window=kv_cache_sliding_window,
                scale_frames=kv_cache_scale_frames,
                keep_special=kv_cache_keep_special,
            )
            for _ in range(trunk_depth)
        ]

        self.token_norm = LayerNorm(dim_in)
        self.trunk_norm = LayerNorm(dim_in)

        # Learnable empty pose token
        self.empty_pose_tokens = mx.zeros((1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in, bias=True)

        # AdaLN modulation: SiLU → Linear(dim_in, 3*dim_in)
        self.poseLN_silu   = nn.SiLU()
        self.poseLN_linear = nn.Linear(dim_in, 3 * dim_in, bias=True)

        # AdaLN norm (no affine params — elementwise_affine=False)
        self.adaln_norm = LayerNorm(dim_in, affine=False)

        # Output branch: MLP(dim_in → dim_in//2 → target_dim)
        self.pose_branch = MLP(dim_in, dim_in // 2, self.target_dim, bias=True)

        # KV cache (list of dicts, one per trunk block, per iteration)
        # Layout: kv_cache[iteration][block_key] = tensor
        self.kv_cache: Optional[List[List[Dict[str, Any]]]] = None
        self.frame_idx = 0

    # ------------------------------------------------------------------

    def clean_kv_cache(self):
        self.kv_cache = None
        self.frame_idx = 0

    def set_skip_append(self, skip: bool):
        if self.kv_cache is not None:
            for iter_cache in self.kv_cache:
                for d in iter_cache:
                    d["_skip_append"] = skip

    def _ensure_kv_cache(self):
        if self.kv_cache is None:
            self.kv_cache = [
                [
                    {"k": None, "v": None, "k_special": None, "v_special": None,
                     "_skip_append": False}
                    for _ in range(self.trunk_depth)
                ]
                for _ in range(self.num_iterations)
            ]

    # ------------------------------------------------------------------

    def __call__(
        self,
        aggregated_tokens_list: List[mx.array],
        causal_inference: bool = False,
        num_iterations: Optional[int] = None,
        num_frame_per_block: int = 1,
        num_frame_for_scale: int = -1,
    ) -> List[mx.array]:
        """
        aggregated_tokens_list: List of [B, S, P, 2C] tensors.
        Returns: list of [B, S, 9] pose encodings, one per iteration.
        """
        if num_iterations is None:
            num_iterations = self.num_iterations

        if causal_inference:
            self._ensure_kv_cache()

        # Camera token is at index 0 of the token sequence
        tokens = aggregated_tokens_list[-1]   # [B, S, P, 2C]
        pose_tokens = tokens[:, :, 0, :]      # [B, S, 2C]
        pose_tokens = self.token_norm(pose_tokens)

        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list: List[mx.array] = []

        for i in range(num_iterations):
            # Build module_input from current pose estimate
            if pred_pose_enc is None:
                module_input = self.embed_pose(
                    mx.broadcast_to(self.empty_pose_tokens, (B, S, self.target_dim))
                )
            else:
                module_input = self.embed_pose(mx.stop_gradient(pred_pose_enc))

            # AdaLN modulation
            mod = self.poseLN_linear(self.poseLN_silu(module_input))  # [B, S, 3C]
            shift_msa = mod[..., :C]
            scale_msa = mod[..., C:2*C]
            gate_msa  = mod[..., 2*C:]

            modulated = gate_msa * _modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            modulated = modulated + pose_tokens   # residual

            # Apply trunk blocks
            for j, block in enumerate(self.trunk):
                kv = self.kv_cache[i][j] if causal_inference else None
                modulated = block(
                    modulated,
                    kv_cache=kv,
                    num_frame_per_block=num_frame_per_block,
                )

            # Pose delta
            delta = self.pose_branch(self.trunk_norm(modulated))   # [B, S, 9]
            if pred_pose_enc is None:
                pred_pose_enc = delta
            else:
                pred_pose_enc = pred_pose_enc + delta

            pred_pose_enc_list.append(
                activate_pose(pred_pose_enc)
            )

        # Advance frame counter for streaming
        if causal_inference:
            self.frame_idx += S

        return pred_pose_enc_list


# ---------------------------------------------------------------------------
# DPT head helpers
# ---------------------------------------------------------------------------

class _ResidualConvUnit(nn.Module):
    """Two conv layers with residual connection (mirrors ResidualConvUnit).

    PyTorch uses ReLU(inplace=True), which modifies the input in-place before
    the residual add, so the effective formula is:
        out = conv2(relu(conv1(relu(x)))) + relu(x)
    not conv2(...) + x.  We replicate that here.
    """

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        residual = nn.relu(x)          # mirrors in-place relu that overwrites x
        out = self.conv1(residual)
        out = nn.relu(out)
        out = self.conv2(out)
        return out + residual          # residual is relu(x), not x


class _FeatureFusionBlock(nn.Module):
    """Fuses two feature maps: residual unit + bilinear upsample + 1×1 conv."""

    def __init__(self, features: int, has_residual: bool = True):
        super().__init__()
        self.has_residual = has_residual
        if has_residual:
            self.resConfUnit1 = _ResidualConvUnit(features)
        self.resConfUnit2 = _ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, bias=True)

    def __call__(self, x: mx.array,
                 skip: Optional[mx.array] = None,
                 target_hw: Optional[Tuple[int, int]] = None) -> mx.array:
        if self.has_residual and skip is not None:
            x = x + self.resConfUnit1(skip)
        x = self.resConfUnit2(x)
        # Bilinear upsample
        if target_hw is not None:
            x = _bilinear(x, target_hw)
        else:
            x = _bilinear(x, (x.shape[1] * 2, x.shape[2] * 2))
        x = self.out_conv(x)
        return x


def _bilinear(x: mx.array, hw: Tuple[int, int], align_corners: bool = True) -> mx.array:
    """Bilinear resize [B, H, W, C] → [B, h, w, C]."""
    B, H, W, C = x.shape
    h, w = hw
    if (h, w) == (H, W):
        return x

    if align_corners:
        y_src = mx.arange(h, dtype=mx.float32) * ((H - 1) / max(h - 1, 1))
        x_src = mx.arange(w, dtype=mx.float32) * ((W - 1) / max(w - 1, 1))
    else:
        y_src = (mx.arange(h, dtype=mx.float32) + 0.5) * (H / h) - 0.5
        x_src = (mx.arange(w, dtype=mx.float32) + 0.5) * (W / w) - 0.5

    y0 = mx.clip(mx.floor(y_src).astype(mx.int32), 0, H - 1)
    y1 = mx.clip(y0 + 1, 0, H - 1)
    x0 = mx.clip(mx.floor(x_src).astype(mx.int32), 0, W - 1)
    x1 = mx.clip(x0 + 1, 0, W - 1)

    wy1 = (y_src - mx.floor(y_src)).reshape(1, h, 1, 1)
    wx1 = (x_src - mx.floor(x_src)).reshape(1, 1, w, 1)
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1

    q00 = x[:, y0, :, :][:, :, x0, :]   # [B, h, w, C]
    q01 = x[:, y0, :, :][:, :, x1, :]
    q10 = x[:, y1, :, :][:, :, x0, :]
    q11 = x[:, y1, :, :][:, :, x1, :]

    return q00 * wy0 * wx0 + q01 * wy0 * wx1 + q10 * wy1 * wx0 + q11 * wy1 * wx1


# ---------------------------------------------------------------------------
# DPT head
# ---------------------------------------------------------------------------

class DPTHeadMLX(nn.Module):
    """Dense Prediction Transformer head (channel-last for MLX).

    Mirrors DPTHead from lingbot_map/heads/dpt_head.py.

    Key differences from the PyTorch version:
    - All Conv2d operate on NHWC tensors (channel last).
    - ConvTranspose2d in resize_layers[0] and [1] are replaced by bilinear
      upsample + Conv2d (MLX ConvTranspose2d is available but bilinear is faster).
    - Positional embeddings are computed as sinusoidal grids on-the-fly.

    Parameters
    ----------
    dim_in : int
        Input token dimension (2*embed_dim = 2048 for ViT-L).
    patch_size : int
        Patch size (14).
    output_dim : int
        Output channels: 2 for depth (value+conf), 4 for points (xyz+conf).
    activation, conf_activation : str
        Activation types passed to activate_head.
    features : int
        DPT fusion feature channels (256).
    out_channels : list
        Per-layer projection channels (default [256, 512, 1024, 1024]).
    """

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = None,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = [256, 512, 1024, 1024]
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.out_channels = out_channels

        self.norm = LayerNorm(dim_in)

        # Token-to-spatial projection: one 1×1 Conv2d per DPT level
        self.projects = [
            nn.Conv2d(dim_in, oc, kernel_size=1)
            for oc in out_channels
        ]

        # Resize layers matching PyTorch DPTHead.resize_layers exactly:
        #   [0] ConvTranspose2d 4× upsample
        #   [1] ConvTranspose2d 2× upsample
        #   [2] identity (no parameters)
        #   [3] Conv2d stride-2 downsample
        # Weights remapped from checkpoint keys resize_layers.{0,1,3}.*
        self.resize_conv0 = nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4)
        self.resize_conv1 = nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2)
        self.resize_conv3 = nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)

        # Scratch-level readout convolutions (checkpoint keys: scratch.layer{1-4}_rn.weight)
        self.layer1_rn = nn.Conv2d(out_channels[0], features, kernel_size=3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, kernel_size=3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, kernel_size=3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, kernel_size=3, padding=1, bias=False)

        # Fusion blocks (checkpoint keys: scratch.refinenet{1-4}.*)
        self.refinenet4 = _FeatureFusionBlock(features, has_residual=False)
        self.refinenet3 = _FeatureFusionBlock(features, has_residual=True)
        self.refinenet2 = _FeatureFusionBlock(features, has_residual=True)
        self.refinenet1 = _FeatureFusionBlock(features, has_residual=True)

        # Output convolutions (checkpoint keys: scratch.output_conv1/2.0/2.*)
        self.output_conv1  = nn.Conv2d(features, features // 2, kernel_size=3, padding=1)
        self.output_conv2a = nn.Conv2d(features // 2, 32, kernel_size=3, padding=1)
        self.output_conv2b = nn.Conv2d(32, output_dim, kernel_size=1)

        # Pos-embed cache
        self._pos_embed_cache: Dict[tuple, mx.array] = {}

    # ------------------------------------------------------------------

    def __call__(
        self,
        aggregated_tokens_list: List[mx.array],
        images: mx.array,
        patch_start_idx: int,
    ) -> Tuple[mx.array, mx.array]:
        """
        aggregated_tokens_list: List of [B, S, P, 2C].
        images: [B, S, 3, H, W] (PyTorch channel-first convention).
        Returns: (preds, conf) each [B, S, H, W, C-1] and [B, S, H, W].
        """
        B, _, _, H, W = images.shape
        S = aggregated_tokens_list[0].shape[1]
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        out_feats = []
        for level, layer_idx in enumerate([0, 1, 2, 3]):
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]  # [B, S, N_patch, 2C]
            x = x.reshape(B * S, patch_h * patch_w, x.shape[-1])
            x = self.norm(x)

            # Reshape to spatial grid (channel-last)
            x = x.reshape(B * S, patch_h, patch_w, x.shape[-1])   # [B*S, H', W', 2C]

            # 1×1 projection
            x = self.projects[level](x)                              # [B*S, H', W', oc]

            # Optional positional embedding
            x = self._apply_pos_embed(x, W, H)

            # Resize
            if level == 0:
                x = self.resize_conv0(x)   # ConvTranspose2d: 4× upsample
            elif level == 1:
                x = self.resize_conv1(x)   # ConvTranspose2d: 2× upsample
            elif level == 2:
                pass  # identity
            elif level == 3:
                x = self.resize_conv3(x)   # Conv2d stride 2: 2× downsample

            out_feats.append(x)

        # Scratch fusion
        l1 = self.layer1_rn(out_feats[0])
        l2 = self.layer2_rn(out_feats[1])
        l3 = self.layer3_rn(out_feats[2])
        l4 = self.layer4_rn(out_feats[3])

        out = self.refinenet4(l4, target_hw=(l3.shape[1], l3.shape[2]))
        out = self.refinenet3(out, skip=l3, target_hw=(l2.shape[1], l2.shape[2]))
        out = self.refinenet2(out, skip=l2, target_hw=(l1.shape[1], l1.shape[2]))
        out = self.refinenet1(out, skip=l1)

        # Output head
        out = self.output_conv1(out)             # [B*S, H', W', features//2]
        # Final upsample to full patch resolution
        out = _bilinear(out, (patch_h * self.patch_size, patch_w * self.patch_size))
        out = self._apply_pos_embed(out, W, H)

        out = nn.relu(self.output_conv2a(out))   # [B*S, H, W, 32]
        out = self.output_conv2b(out)             # [B*S, H, W, output_dim]

        preds, conf = activate_head(out, self.activation, self.conf_activation)

        # Reshape back to [B, S, H, W, ...]
        preds = preds.reshape(B, S, *preds.shape[1:])
        conf  = conf.reshape(B, S, *conf.shape[1:])
        return preds, conf

    def _apply_pos_embed(self, x: mx.array, W: int, H: int, ratio: float = 0.1) -> mx.array:
        """Sinusoidal UV positional embedding matching PyTorch DPTHead._apply_pos_embed.

        Uses create_uv_grid + position_grid_to_embed logic from lingbot_map/heads/utils.py.
        x: [B, ph, pw, C] (channel-last).
        """
        ph, pw, C = x.shape[1], x.shape[2], x.shape[3]
        key = (pw, ph, W / H, C)
        if key not in self._pos_embed_cache:
            aspect = W / H
            diag = (aspect ** 2 + 1.0) ** 0.5
            span_x = aspect / diag
            span_y = 1.0 / diag

            # Bounds matching create_uv_grid (align_corners-like: (N-1)/N scaling)
            lx = -span_x * (pw - 1) / pw
            rx =  span_x * (pw - 1) / pw
            ty = -span_y * (ph - 1) / ph
            by =  span_y * (ph - 1) / ph

            # meshgrid xy-indexing → uu[i,j]=x[j], vv[i,j]=y[i], shape [ph, pw]
            x_c = np.linspace(lx, rx, pw, dtype=np.float32)
            y_c = np.linspace(ty, by, ph, dtype=np.float32)
            uu, vv = np.meshgrid(x_c, y_c)            # [ph, pw]
            uv = np.stack([uu, vv], axis=-1)          # [ph, pw, 2]
            pos = uv.reshape(-1, 2)                   # [ph*pw, 2]

            # position_grid_to_embed: half C for x, half for y; omega_0=100
            def _sincos(coords: np.ndarray, dim: int) -> np.ndarray:
                """[N] → [N, dim] sincos embedding."""
                omega = np.arange(dim // 2, dtype=np.float32) / (dim / 2.0)
                omega = 1.0 / (100.0 ** omega)           # [dim//2]
                out = np.outer(coords, omega)             # [N, dim//2]
                return np.concatenate([np.sin(out), np.cos(out)], axis=-1)

            half = C // 2
            emb_x = _sincos(pos[:, 0], half)             # [ph*pw, C//2]
            emb_y = _sincos(pos[:, 1], half)             # [ph*pw, C//2]
            emb = np.concatenate([emb_x, emb_y], axis=-1)  # [ph*pw, C]
            emb = emb.reshape(ph, pw, C).astype(np.float32) * ratio
            self._pos_embed_cache[key] = mx.array(emb[None])  # [1, ph, pw, C]

        emb = self._pos_embed_cache[key]
        return x + mx.broadcast_to(emb, x.shape)
