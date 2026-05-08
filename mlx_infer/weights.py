"""
PyTorch checkpoint → MLX weight conversion for GCTStreamMLX.

Usage
-----
    from mlx_infer.weights import load_checkpoint
    model = GCTStreamMLX(...)
    load_checkpoint(model, "path/to/checkpoint.pt")

Key remapping
-------------
The PyTorch checkpoint uses slightly different attribute paths than the MLX
model for a few structural reasons:

1. patch_embed.patch_embed.proj.* → patch_embed.patch_embed.*
   PyTorch VisionTransformer wraps the patch Conv2d in a PatchEmbed class with
   a .proj attribute.  Our ViTMLX stores the Conv2d directly.

2. camera_head.poseLN_modulation.1.* → camera_head.poseLN_linear.*
   PyTorch uses nn.Sequential([SiLU, Linear]); we store the Linear as a named
   attribute `poseLN_linear`.

3. {depth,point}_head.resize_layers.{0,1,3}.* → {depth,point}_head.resize_conv{0,1,3}.*
   PyTorch DPTHead stores resize convs in a ModuleList; ours use named attrs.

4. {depth,point}_head.scratch.* → {depth,point}_head.*
   PyTorch DPTHead groups fusion layers under a nested `scratch` module; ours
   are flat at the head level.

5. {depth,point}_head.scratch.output_conv2.0.* → {depth,point}_head.output_conv2a.*
   {depth,point}_head.scratch.output_conv2.2.* → {depth,point}_head.output_conv2b.*
   PyTorch uses nn.Sequential for the two output convs; ours are named attrs.

ConvTranspose2d weight layout
------------------------------
PyTorch ConvTranspose2d weight: [in_channels, out_channels, kH, kW]
MLX   ConvTranspose2d weight: [out_channels, kH, kW, in_channels]
→ transpose axes (1, 2, 3, 0)

Conv2d weight layout
--------------------
PyTorch Conv2d weight: [out_channels, in_channels, kH, kW]
MLX   Conv2d weight: [out_channels, kH, kW, in_channels]
→ transpose axes (0, 2, 3, 1)
"""

import re
from pathlib import Path
from typing import Dict, Any
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


# ---------------------------------------------------------------------------
# Key remapping rules
# ---------------------------------------------------------------------------

def _remap_key(k: str) -> str:
    """Transform a PyTorch checkpoint key to its MLX model equivalent."""

    # 1. patch_embed Conv2d: remove the extra .proj level
    k = re.sub(
        r"(aggregator\.patch_embed\.patch_embed)\.proj\.(weight|bias)$",
        r"\1.\2", k
    )

    # 2. camera_head poseLN_modulation sequential → named attr
    k = re.sub(
        r"camera_head\.poseLN_modulation\.1\.(weight|bias)$",
        r"camera_head.poseLN_linear.\1", k
    )

    # 3. DPT resize_layers ModuleList → named attributes
    for head in ("depth_head", "point_head"):
        for old, new in (("0", "0"), ("1", "1"), ("3", "3")):
            k = k.replace(
                f"{head}.resize_layers.{old}.",
                f"{head}.resize_conv{new}."
            )

    # 4. Strip DPT scratch.* namespace (also handles scratch.refinenet*, scratch.layer*_rn)
    for head in ("depth_head", "point_head"):
        k = k.replace(f"{head}.scratch.", f"{head}.")

    # 5. DPT output_conv2 Sequential indices → named attrs
    for head in ("depth_head", "point_head"):
        k = k.replace(f"{head}.output_conv2.0.", f"{head}.output_conv2a.")
        k = k.replace(f"{head}.output_conv2.2.", f"{head}.output_conv2b.")

    return k


# Keys to silently drop (no corresponding parameter in the MLX model)
_DROP_PATTERNS = [
    re.compile(r"\.mask_token$"),            # not used at inference
    re.compile(r"\.poseLN_modulation\.0\."),  # SiLU has no params
]


def _should_drop(k: str) -> bool:
    return any(p.search(k) for p in _DROP_PATTERNS)


# Keys whose weight layout needs ConvTranspose2d treatment.
# Checked on the ORIGINAL (pre-remap) key.
_CONV_TRANSPOSE_RE = re.compile(r"resize_layers\.[01]\.weight$")


def _is_conv_transpose(original_key: str) -> bool:
    return bool(_CONV_TRANSPOSE_RE.search(original_key))


# ---------------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------------

def _convert_tensor(original_key: str, t) -> mx.array:
    """Convert a single PyTorch tensor to an MLX array with layout fixes."""
    arr = np.asarray(t.float().cpu())   # always float32

    if arr.ndim == 4 and original_key.endswith(".weight"):
        if _is_conv_transpose(original_key):
            # PyTorch ConvTranspose2d: [I, O, kH, kW] → MLX [O, kH, kW, I]
            arr = arr.transpose(1, 2, 3, 0)
        else:
            # PyTorch Conv2d: [O, I, kH, kW] → MLX [O, kH, kW, I]
            arr = arr.transpose(0, 2, 3, 1)

    return mx.array(arr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_state_dict(state_dict: Dict[str, Any]) -> Dict[str, mx.array]:
    """Convert + remap a PyTorch state dict to MLX-compatible flat weight dict."""
    out: Dict[str, mx.array] = {}
    for k, v in state_dict.items():
        if _should_drop(k):
            continue
        mlx_key = _remap_key(k)
        try:
            out[mlx_key] = _convert_tensor(k, v)
        except Exception as e:
            print(f"[weights] skipping {k}: {e}")
    return out


def load_checkpoint(
    model: nn.Module,
    path: str,
    strict: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """Load a PyTorch .pt checkpoint into an MLX GCTStreamMLX model.

    Parameters
    ----------
    model : nn.Module
        The GCTStreamMLX instance to populate.
    path : str
        Path to the PyTorch checkpoint (.pt file).
    strict : bool
        If True, raise on missing or unexpected keys (after remapping).
    verbose : bool
        Print key statistics.

    Returns
    -------
    model (in-place modified).
    """
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    mlx_weights = convert_state_dict(state_dict)

    # Collect the model's parameter keys from the flattened tree
    model_keys = set(k for k, _ in tree_flatten(model.parameters()))
    ckpt_keys  = set(mlx_weights.keys())

    matched    = model_keys & ckpt_keys
    missing    = model_keys - ckpt_keys     # in model, not in ckpt
    unexpected = ckpt_keys - model_keys     # in ckpt, not in model

    if verbose:
        print(f"[weights] loaded {len(state_dict)} tensors from {Path(path).name}")
        print(f"[weights] matched {len(matched)} / {len(model_keys)} model params")
        if missing:
            print(f"[weights] missing  ({len(missing)}): "
                  + ", ".join(sorted(missing)[:6])
                  + (" ..." if len(missing) > 6 else ""))
        if unexpected:
            print(f"[weights] unexpected ({len(unexpected)}): "
                  + ", ".join(sorted(unexpected)[:6])
                  + (" ..." if len(unexpected) > 6 else ""))

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Strict load failed: {len(missing)} missing, {len(unexpected)} unexpected"
        )

    # Filter to only matched keys, then load — pass strict=False so MLX doesn't
    # raise for parameters that exist in the model but weren't in the checkpoint.
    filtered = {k: v for k, v in mlx_weights.items() if k in model_keys}
    model.load_weights(list(filtered.items()), strict=False)
    mx.eval(model.parameters())
    return model
