"""Inference-only nonpersistent BF16 parameter caches."""
import types
import weakref
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WeightCacheStats:
    blocks_patched: int
    buffers_registered: int


def _set_nonpersistent_buffer(module, name: str, value: torch.Tensor | None) -> bool:
    if name in module._buffers:
        module._buffers[name] = value
        module._non_persistent_buffers_set.add(name)
        return False
    module.register_buffer(name, value, persistent=False)
    return True


def _refresh_mlp_shadow_buffers(mlp) -> int:
    buffers_registered = 0
    for source_name, buffer_name in (
        ("fc1.weight", "fc1_weight_bf16"),
        ("fc1.bias", "fc1_bias_bf16"),
        ("fc2.weight", "fc2_weight_bf16"),
        ("fc2.bias", "fc2_bias_bf16"),
    ):
        owner_name, attr_name = source_name.split(".")
        source = getattr(getattr(mlp, owner_name), attr_name)
        shadow = None if source is None else source.detach().to(torch.bfloat16).contiguous()
        if _set_nonpersistent_buffer(mlp, buffer_name, shadow):
            buffers_registered += 1
    return buffers_registered


def _refresh_qkv_shadow_buffers(attn) -> int:
    qkv = getattr(attn, "qkv")
    buffers_registered = 0
    for source_name, buffer_name in (
        ("weight", "qkv_weight_bf16"),
        ("bias", "qkv_bias_bf16"),
    ):
        source = getattr(qkv, source_name)
        shadow = None if source is None else source.detach().to(torch.bfloat16).contiguous()
        if _set_nonpersistent_buffer(attn, buffer_name, shadow):
            buffers_registered += 1
    return buffers_registered


def _cached_qkv_linear(attn, x):
    return F.linear(x, attn.qkv_weight_bf16, attn.qkv_bias_bf16)


def _global_qkv_cached_forward(qkv, x):
    attn = qkv._qkv_cache_owner_ref()
    if attn is None:
        raise RuntimeError("Global QKV cache attention owner was released")
    return F.linear(x, attn.qkv_weight_bf16, attn.qkv_bias_bf16)


def _global_mlp_cached_forward(self, x, attn_out):
    x = self.attn_post(x, attn_out)
    y = self.norm2(x)
    mlp = self.mlp
    y = F.linear(y, mlp.fc1_weight_bf16, mlp.fc1_bias_bf16)
    y = mlp.act(y)
    y = mlp.drop(y)
    y = F.linear(y, mlp.fc2_weight_bf16, mlp.fc2_bias_bf16)
    y = mlp.drop(y)
    return x + self.ls2(y)


def _frame_mlp_cached_forward(
    self,
    x,
    pos=None,

    num_patches=None,
    num_special=None,
    num_frames=None,
    enable_3d_rope=False,
):
    if self.training:
        return self._frame_mlp_original_forward(
            x,
            pos=pos,
            num_patches=num_patches,
            num_special=num_special,
            num_frames=num_frames,
            enable_3d_rope=enable_3d_rope,
        )

    qkv = getattr(self.attn, "_frame_cached_qkv_linear", None)
    if qkv is None:
        attn_out = self.attn(
            self.norm1(x),
            pos=pos,
            num_patches=num_patches,
            num_special=num_special,
            num_frames=num_frames,
            enable_3d_rope=enable_3d_rope,
        )
    else:
        attn_out = self.attn.forward_with_qkv_linear(
            self.norm1(x),
            qkv_linear=qkv,
            pos=pos,
            num_patches=num_patches,
            num_special=num_special,
            num_frames=num_frames,
            enable_3d_rope=enable_3d_rope,
        )

    x = x + self.ls1(
        attn_out
    )

    mlp = self.mlp
    y = self.norm2(x)
    y = F.linear(y, mlp.fc1_weight_bf16, mlp.fc1_bias_bf16)
    y = mlp.act(y)
    y = mlp.drop(y)
    y = F.linear(y, mlp.fc2_weight_bf16, mlp.fc2_bias_bf16)
    y = mlp.drop(y)
    return x + self.ls2(y)


def _patch_mlp_cached_forward(
    self,
    x,
    pos=None,

    num_patches=None,
    num_special=None,
    num_frames=None,
    enable_3d_rope=False,
):
    if self.training:
        return self._patch_mlp_original_forward(
            x,
            pos=pos,
            num_patches=num_patches,
            num_special=num_special,
            num_frames=num_frames,
            enable_3d_rope=enable_3d_rope,
        )

    x = x + self.ls1(
        self.attn(
            self.norm1(x),
            pos=pos,
            num_patches=num_patches,
            num_special=num_special,
            num_frames=num_frames,
            enable_3d_rope=enable_3d_rope,
        )
    )

    mlp = self.mlp
    y = self.norm2(x)
    y = F.linear(y, mlp.fc1_weight_bf16, mlp.fc1_bias_bf16)
    y = mlp.act(y)
    y = mlp.drop(y)
    y = F.linear(y, mlp.fc2_weight_bf16, mlp.fc2_bias_bf16)
    y = mlp.drop(y)
    return x + self.ls2(y)


def _patch_qkv_cached_forward(
    self,
    x,
    pos=None,

    num_patches=None,
    num_special=None,
    num_frames=None,
    enable_3d_rope=False,
):
    if self.training:
        return self._patch_qkv_original_forward(
            x,
            pos=pos,
            num_patches=num_patches,
            num_special=num_special,
            num_frames=num_frames,
            enable_3d_rope=enable_3d_rope,
        )

    qkv = self.attn._patch_cached_qkv_linear
    x = x + self.ls1(
        self.attn.forward_with_qkv_linear(
            self.norm1(x),
            qkv_linear=qkv,
            pos=pos,
            num_patches=num_patches,
            num_special=num_special,
            num_frames=num_frames,
            enable_3d_rope=enable_3d_rope,
        )
    )

    if hasattr(self, "_patch_mlp_weight_cache_enabled"):
        mlp = self.mlp
        y = self.norm2(x)
        y = F.linear(y, mlp.fc1_weight_bf16, mlp.fc1_bias_bf16)
        y = mlp.act(y)
        y = mlp.drop(y)
        y = F.linear(y, mlp.fc2_weight_bf16, mlp.fc2_bias_bf16)
        y = mlp.drop(y)
        return x + self.ls2(y)

    return x + self.ls2(self.mlp(self.norm2(x)))


def _frame_qkv_cached_forward(
    self,
    x,
    pos=None,

    num_patches=None,
    num_special=None,
    num_frames=None,
    enable_3d_rope=False,
):
    if self.training:
        return self._frame_qkv_original_forward(
            x,
            pos=pos,
            num_patches=num_patches,
            num_special=num_special,
            num_frames=num_frames,
            enable_3d_rope=enable_3d_rope,
        )

    qkv = self.attn._frame_cached_qkv_linear
    x = x + self.ls1(
        self.attn.forward_with_qkv_linear(
            self.norm1(x),
            qkv_linear=qkv,
            pos=pos,
            num_patches=num_patches,
            num_special=num_special,
            num_frames=num_frames,
            enable_3d_rope=enable_3d_rope,
        )
    )

    if hasattr(self, "_frame_mlp_weight_cache_enabled"):
        mlp = self.mlp
        y = self.norm2(x)
        y = F.linear(y, mlp.fc1_weight_bf16, mlp.fc1_bias_bf16)
        y = mlp.act(y)
        y = mlp.drop(y)
        y = F.linear(y, mlp.fc2_weight_bf16, mlp.fc2_bias_bf16)
        y = mlp.drop(y)
        return x + self.ls2(y)

    return x + self.ls2(self.mlp(self.norm2(x)))


def _iter_patch_embed_blocks(model):
    agg = getattr(model, "aggregator", None)
    patch_embed = getattr(agg, "patch_embed", None)
    blocks = getattr(patch_embed, "blocks", None)
    if blocks is None:
        return []

    flattened = []
    for block in blocks:
        if hasattr(block, "mlp"):
            flattened.append(block)
        elif isinstance(block, torch.nn.ModuleList):
            flattened.extend(child for child in block if hasattr(child, "mlp"))
    return flattened


def apply_global_mlp_weight_cache(model) -> WeightCacheStats:
    """Cache BF16 MLP parameters in global attention blocks.

    The original fp32 parameters remain untouched.  Shadow buffers are marked
    ``persistent=False`` so they do not affect checkpoint or state_dict behavior.
    """
    agg = getattr(model, "aggregator", None)
    global_blocks = getattr(agg, "global_blocks", None)
    if global_blocks is None:
        return WeightCacheStats(blocks_patched=0, buffers_registered=0)

    blocks_patched = 0
    buffers_registered = 0
    for block in global_blocks:
        mlp = getattr(block, "mlp", None)
        if mlp is None or not hasattr(block, "attn_post_ffn"):
            continue
        if not (hasattr(mlp, "fc1") and hasattr(mlp, "fc2")):
            continue
        buffers_registered += _refresh_mlp_shadow_buffers(mlp)
        block.attn_post_ffn = types.MethodType(_global_mlp_cached_forward, block)
        block._global_mlp_weight_cache_enabled = True
        blocks_patched += 1

    return WeightCacheStats(
        blocks_patched=blocks_patched,
        buffers_registered=buffers_registered,
    )


def apply_frame_mlp_weight_cache(model) -> WeightCacheStats:
    """Cache BF16 MLP parameters in frame-attention blocks.

    The original fp32 parameters remain untouched.  Shadow buffers are marked
    ``persistent=False`` so they do not affect checkpoint or state_dict behavior.
    """
    agg = getattr(model, "aggregator", None)
    frame_blocks = getattr(agg, "frame_blocks", None)
    if frame_blocks is None:
        return WeightCacheStats(blocks_patched=0, buffers_registered=0)

    blocks_patched = 0
    buffers_registered = 0
    for block in frame_blocks:
        mlp = getattr(block, "mlp", None)
        if mlp is None or not (hasattr(mlp, "fc1") and hasattr(mlp, "fc2")):
            continue
        buffers_registered += _refresh_mlp_shadow_buffers(mlp)
        if not hasattr(block, "_frame_mlp_original_forward"):
            block._frame_mlp_original_forward = block.forward
        block.forward = types.MethodType(_frame_mlp_cached_forward, block)
        block._frame_mlp_weight_cache_enabled = True
        blocks_patched += 1

    return WeightCacheStats(
        blocks_patched=blocks_patched,
        buffers_registered=buffers_registered,
    )


def apply_patch_mlp_weight_cache(model) -> WeightCacheStats:
    """Cache BF16 MLP parameters in patch-embedding DINO/ViT blocks.

    The original fp32 parameters remain untouched.  Shadow buffers are marked
    ``persistent=False`` so they do not affect checkpoint or state_dict behavior.
    """
    blocks_patched = 0
    buffers_registered = 0
    for block in _iter_patch_embed_blocks(model):
        mlp = getattr(block, "mlp", None)
        if mlp is None or not (hasattr(mlp, "fc1") and hasattr(mlp, "fc2")):
            continue
        buffers_registered += _refresh_mlp_shadow_buffers(mlp)
        if not hasattr(block, "_patch_mlp_original_forward"):
            block._patch_mlp_original_forward = block.forward
        block.forward = types.MethodType(_patch_mlp_cached_forward, block)
        block._patch_mlp_weight_cache_enabled = True
        blocks_patched += 1

    return WeightCacheStats(
        blocks_patched=blocks_patched,
        buffers_registered=buffers_registered,
    )


def apply_frame_qkv_weight_cache(model) -> WeightCacheStats:
    """Cache BF16 QKV parameters in frame-attention blocks.

    Only frame blocks are patched. The original fp32 qkv parameters remain
    untouched. Shadow buffers are marked ``persistent=False`` so they do not
    affect checkpoint or state_dict behavior.
    """
    agg = getattr(model, "aggregator", None)
    frame_blocks = getattr(agg, "frame_blocks", None)
    if frame_blocks is None:
        return WeightCacheStats(blocks_patched=0, buffers_registered=0)

    blocks_patched = 0
    buffers_registered = 0
    for block in frame_blocks:
        attn = getattr(block, "attn", None)
        qkv = getattr(attn, "qkv", None)
        if attn is None or qkv is None:
            continue
        if not (hasattr(qkv, "weight") and hasattr(qkv, "bias")):
            continue
        buffers_registered += _refresh_qkv_shadow_buffers(attn)
        attn._frame_cached_qkv_linear = types.MethodType(_cached_qkv_linear, attn)
        if not hasattr(block, "_frame_qkv_original_forward"):
            block._frame_qkv_original_forward = block.forward
        block.forward = types.MethodType(_frame_qkv_cached_forward, block)
        block._frame_qkv_weight_cache_enabled = True
        blocks_patched += 1

    return WeightCacheStats(
        blocks_patched=blocks_patched,
        buffers_registered=buffers_registered,
    )


def apply_patch_qkv_weight_cache(model) -> WeightCacheStats:
    """Cache BF16 QKV parameters in patch-embedding blocks.

    Only patch_embed blocks are patched. The original fp32 qkv parameters
    remain untouched. Shadow buffers are marked ``persistent=False`` so they do
    not affect checkpoint or state_dict behavior.
    """
    blocks_patched = 0
    buffers_registered = 0
    for block in _iter_patch_embed_blocks(model):
        attn = getattr(block, "attn", None)
        qkv = getattr(attn, "qkv", None)
        if attn is None or qkv is None:
            continue
        if not (hasattr(qkv, "weight") and hasattr(qkv, "bias")):
            continue
        buffers_registered += _refresh_qkv_shadow_buffers(attn)
        attn._patch_cached_qkv_linear = types.MethodType(_cached_qkv_linear, attn)
        if not hasattr(block, "_patch_qkv_original_forward"):
            block._patch_qkv_original_forward = block.forward
        block.forward = types.MethodType(_patch_qkv_cached_forward, block)
        block._patch_qkv_weight_cache_enabled = True
        blocks_patched += 1

    return WeightCacheStats(
        blocks_patched=blocks_patched,
        buffers_registered=buffers_registered,
    )


def apply_global_qkv_weight_cache(model) -> WeightCacheStats:
    """Cache BF16 QKV parameters in global attention blocks.

    Only global blocks are patched.  The original fp32 qkv parameters remain
    untouched.  Shadow buffers are marked ``persistent=False`` so they do not
    affect checkpoint or state_dict behavior.

    The patch deliberately leaves ``FlashInferAttention.prepare_qkv`` intact:
    replacing only ``attn.qkv.forward`` preserves q/k norm, RoPE, FA4 handoff,
    and KV/cache semantics.
    """
    agg = getattr(model, "aggregator", None)
    global_blocks = getattr(agg, "global_blocks", None)
    if global_blocks is None:
        return WeightCacheStats(blocks_patched=0, buffers_registered=0)

    blocks_patched = 0
    buffers_registered = 0
    for block in global_blocks:
        attn = getattr(block, "attn", None)
        qkv = getattr(attn, "qkv", None)
        if attn is None or qkv is None:
            continue
        if not (hasattr(qkv, "weight") and hasattr(qkv, "bias")):
            continue
        buffers_registered += _refresh_qkv_shadow_buffers(attn)
        object.__setattr__(qkv, "_qkv_cache_owner_ref", weakref.ref(attn))
        if not hasattr(qkv, "_global_qkv_original_forward"):
            qkv._global_qkv_original_forward = qkv.forward
        qkv.forward = types.MethodType(_global_qkv_cached_forward, qkv)
        block._global_qkv_weight_cache_enabled = True
        blocks_patched += 1

    return WeightCacheStats(
        blocks_patched=blocks_patched,
        buffers_registered=buffers_registered,
    )
