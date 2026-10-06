"""Compile hot modules before capturing the static aggregator/depth stages."""
import torch


def compile_default(model):
    """Compile hot modules WITHOUT reduce-overhead so the inner torch.compile graphs
    don't fight our outer manual graph capture for ownership of the CUDA stream.
    Default mode still gets us inductor codegen / fusion; just not the inner CUDA-
    graph wrapping."""
    agg = model.aggregator
    for i, b in enumerate(agg.frame_blocks):
        agg.frame_blocks[i] = torch.compile(b)
    for i, b in enumerate(agg.patch_embed.blocks):
        agg.patch_embed.blocks[i] = torch.compile(b)
    for b in agg.global_blocks:
        if hasattr(b, 'attn_pre'):
            b.attn_pre = torch.compile(b.attn_pre)
        if hasattr(b, 'attn_post_ffn'):
            b.attn_post_ffn = torch.compile(b.attn_post_ffn)
    if model.depth_head is not None and hasattr(model.depth_head, '_forward_impl'):
        model.depth_head._forward_impl = torch.compile(model.depth_head._forward_impl)
    assert model.point_head is None, 'Thor captured runner does not support point_head'
