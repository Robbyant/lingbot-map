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


def _candidate_qkv_linear(attn, x):
    return F.linear(x, attn.qkv_weight_bf16, attn.qkv_bias_bf16)


def _candidate002b3_qkv_forward(qkv, x):
    attn = qkv._candidate002b3_attn_owner_ref()
    if attn is None:
        raise RuntimeError("Candidate 002B-3 attention owner was released")
    return F.linear(x, attn.qkv_weight_bf16, attn.qkv_bias_bf16)


def _candidate_attn_post_ffn(self, x, attn_out):
    x = self.attn_post(x, attn_out)
    y = self.norm2(x)
    mlp = self.mlp
    y = F.linear(y, mlp.fc1_weight_bf16, mlp.fc1_bias_bf16)
    y = mlp.act(y)
    y = mlp.drop(y)
    y = F.linear(y, mlp.fc2_weight_bf16, mlp.fc2_bias_bf16)
    y = mlp.drop(y)
    return x + self.ls2(y)


def _candidate001c_frame_block_forward(
    self,
    x,
    pos=None,

    num_patches=None,
    num_special=None,
    num_frames=None,
    enable_3d_rope=False,
):
    if self.training:
        return self._candidate001c_original_forward(
            x,
            pos=pos,
            num_patches=num_patches,
            num_special=num_special,
            num_frames=num_frames,
            enable_3d_rope=enable_3d_rope,
        )

    qkv = getattr(self.attn, "_candidate002b1_qkv_linear", None)
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


def _candidate002a_patch_block_forward(
    self,
    x,
    pos=None,

    num_patches=None,
    num_special=None,
    num_frames=None,
    enable_3d_rope=False,
):
    if self.training:
        return self._candidate002a_original_forward(
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


def _candidate002b2_patch_block_forward(
    self,
    x,
    pos=None,

    num_patches=None,
    num_special=None,
    num_frames=None,
    enable_3d_rope=False,
):
    if self.training:
        return self._candidate002b2_original_forward(
            x,
            pos=pos,
            num_patches=num_patches,
            num_special=num_special,
            num_frames=num_frames,
            enable_3d_rope=enable_3d_rope,
        )

    qkv = self.attn._candidate002b2_qkv_linear
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

    if hasattr(self, "_candidate002a_patch_mlp_weight_cast_hoist"):
        mlp = self.mlp
        y = self.norm2(x)
        y = F.linear(y, mlp.fc1_weight_bf16, mlp.fc1_bias_bf16)
        y = mlp.act(y)
        y = mlp.drop(y)
        y = F.linear(y, mlp.fc2_weight_bf16, mlp.fc2_bias_bf16)
        y = mlp.drop(y)
        return x + self.ls2(y)

    return x + self.ls2(self.mlp(self.norm2(x)))


def _candidate002b1_frame_block_forward(
    self,
    x,
    pos=None,

    num_patches=None,
    num_special=None,
    num_frames=None,
    enable_3d_rope=False,
):
    if self.training:
        return self._candidate002b1_original_forward(
            x,
            pos=pos,
            num_patches=num_patches,
            num_special=num_special,
            num_frames=num_frames,
            enable_3d_rope=enable_3d_rope,
        )

    qkv = self.attn._candidate002b1_qkv_linear
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

    if hasattr(self, "_candidate001c_frame_mlp_weight_cast_hoist"):
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


def apply_candidate001_weight_cast_hoist(model) -> WeightCacheStats:
    """Patch GCA global blocks for Candidate 001B.

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
        block.attn_post_ffn = types.MethodType(_candidate_attn_post_ffn, block)
        block._candidate001_weight_cast_hoist = True
        blocks_patched += 1

    return WeightCacheStats(
        blocks_patched=blocks_patched,
        buffers_registered=buffers_registered,
    )


def apply_candidate001c_frame_mlp_weight_cast_hoist(model) -> WeightCacheStats:
    """Patch frame-attention block MLPs for Candidate 001C.

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
        if not hasattr(block, "_candidate001c_original_forward"):
            block._candidate001c_original_forward = block.forward
        block.forward = types.MethodType(_candidate001c_frame_block_forward, block)
        block._candidate001c_frame_mlp_weight_cast_hoist = True
        blocks_patched += 1

    return WeightCacheStats(
        blocks_patched=blocks_patched,
        buffers_registered=buffers_registered,
    )


def apply_candidate002a_patch_mlp_weight_cast_hoist(model) -> WeightCacheStats:
    """Patch patch_embed DINO/ViT block MLPs for Candidate 002A.

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
        if not hasattr(block, "_candidate002a_original_forward"):
            block._candidate002a_original_forward = block.forward
        block.forward = types.MethodType(_candidate002a_patch_block_forward, block)
        block._candidate002a_patch_mlp_weight_cast_hoist = True
        blocks_patched += 1

    return WeightCacheStats(
        blocks_patched=blocks_patched,
        buffers_registered=buffers_registered,
    )


def apply_candidate002b1_frame_qkv_weight_cast_hoist(model) -> WeightCacheStats:
    """Patch frame-attention QKV linears for Candidate 002B-1.

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
        attn._candidate002b1_qkv_linear = types.MethodType(_candidate_qkv_linear, attn)
        if not hasattr(block, "_candidate002b1_original_forward"):
            block._candidate002b1_original_forward = block.forward
        block.forward = types.MethodType(_candidate002b1_frame_block_forward, block)
        block._candidate002b1_frame_qkv_weight_cast_hoist = True
        blocks_patched += 1

    return WeightCacheStats(
        blocks_patched=blocks_patched,
        buffers_registered=buffers_registered,
    )


def apply_candidate002b2_patch_qkv_weight_cast_hoist(model) -> WeightCacheStats:
    """Patch patch_embed QKV linears for Candidate 002B-2.

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
        attn._candidate002b2_qkv_linear = types.MethodType(_candidate_qkv_linear, attn)
        if not hasattr(block, "_candidate002b2_original_forward"):
            block._candidate002b2_original_forward = block.forward
        block.forward = types.MethodType(_candidate002b2_patch_block_forward, block)
        block._candidate002b2_patch_qkv_weight_cast_hoist = True
        blocks_patched += 1

    return WeightCacheStats(
        blocks_patched=blocks_patched,
        buffers_registered=buffers_registered,
    )


def apply_candidate002b3_gca_qkv_weight_cast_hoist(model) -> WeightCacheStats:
    """Patch GCA/global QKV linears for Candidate 002B-3.

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
        object.__setattr__(qkv, "_candidate002b3_attn_owner_ref", weakref.ref(attn))
        if not hasattr(qkv, "_candidate002b3_original_forward"):
            qkv._candidate002b3_original_forward = qkv.forward
        qkv.forward = types.MethodType(_candidate002b3_qkv_forward, qkv)
        block._candidate002b3_gca_qkv_weight_cast_hoist = True
        blocks_patched += 1

    return WeightCacheStats(
        blocks_patched=blocks_patched,
        buffers_registered=buffers_registered,
    )
