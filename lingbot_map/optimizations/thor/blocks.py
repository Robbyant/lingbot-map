from torch import Tensor
import torch.nn.functional as F
from lingbot_map.layers.attention import apply_rotary_emb

def attn_post(self, x: Tensor, attn_out: Tensor) -> Tensor:
    """Project NHD attention output and apply the layer-scaled residual."""
    attn_out = attn_out.reshape(x.shape[0], attn_out.shape[0],
                                self.attn.num_heads * self.attn.head_dim)
    attn_out = self.attn.proj(attn_out)
    return x + self.ls1(attn_out)

def attn_post_ffn(self, x: Tensor, attn_out: Tensor) -> Tensor:
    """Compile the post-attention projection and FFN residual as one region."""
    x = self.attn_post(x, attn_out)
    return self.ffn_residual(x)

def forward(
    self,
    x: Tensor,
    pos=None,

    num_patches=None,
    num_special=None,
    num_frames=None,
    enable_3d_rope=False,
    kv_cache=None,
    global_idx=0,
    num_frame_per_block=1,
    num_frame_for_scale=-1,
    num_register_tokens=4,
) -> Tensor:
    # Phase 2 (streaming): single-frame FlashInfer paged attention.
    # Keep norm1 and QKV preparation in one compilation region.
    is_streaming = (kv_cache is not None and (num_frames is None or num_frames <= 1))
    if is_streaming:
        manager = kv_cache
        if getattr(manager, '_skip_append', False) or getattr(manager, '_defer_eviction', False):
            raise RuntimeError("Thor streaming requires every frame to append; dynamic keyframes are unsupported")
        # Compiled: norm1 + qkv linear + reshape + q_norm + k_norm + RoPE + format
        q_nhd, k_nhd, v_nhd = self.attn_pre(x, pos=pos, enable_3d_rope=enable_3d_rope)

        # Graph-capture mode: every per-frame Python state mutation has been
        # hoisted to ``manager.prepare_frame_for_graph`` (called outside the
        # captured graph), and the writes here are tensor-driven so they're
        # safe to record into a ``torch.cuda.graph``.
        if getattr(manager, '_graph_mode', False):
            manager.append_frame_graph(global_idx, k_nhd, v_nhd)
            attn_x = manager.compute_attention_graph(global_idx, q_nhd)
            x = self.attn_post_ffn(x, attn_x)
            return x

        # Eager: write frame K/V to paged cache
        manager.append_frame(global_idx, k_nhd, v_nhd)
        # CPU-only: update eviction state (deque ops, no GPU kernel)
        manager.evict_frames(
            block_idx=global_idx,
            scale_frames=self.attn.kv_cache_scale_frames,
            sliding_window=self.attn.kv_cache_sliding_window,
            cross_frame_special=self.attn.kv_cache_cross_frame_special,
            include_scale_frames=self.attn.kv_cache_include_scale_frames,
            camera_only=self.attn.kv_cache_camera_only,
            num_register_tokens=num_register_tokens,
        )
        # Eager: FlashInfer BatchPrefillWithPagedKVCacheWrapper
        attn_x = manager.compute_attention(global_idx, q_nhd)

        # The installed runtime compiles this projection/residual/FFN region.
        x = self.attn_post_ffn(x, attn_x)
        return x  # ffn_residual already applied inside attn_post_ffn
    else:
        # Phase 1 (multi-frame scale pass) or non-streaming training path
        x = x + self.ls1(self.attn(
            self.norm1(x),
            pos=pos,
            num_patches=num_patches,
            num_special=num_special,
            num_frames=num_frames,
            enable_3d_rope=enable_3d_rope,
            kv_cache=kv_cache,
            global_idx=global_idx,
            num_frame_per_block=num_frame_per_block,
            num_frame_for_scale=num_frame_for_scale,
            num_register_tokens=num_register_tokens,
        ))
    x = self.ffn_residual(x)
    return x

def forward_with_qkv_linear(self, x: Tensor, qkv_linear, pos=None,  num_patches=None, num_special=None, num_frames=None, enable_3d_rope=False) -> Tensor:
    B, N, C = x.shape
    qkv = qkv_linear(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)


    if self.rope is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v


    x = x.transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def prepare_qkv(self, x: Tensor, pos=None, enable_3d_rope: bool = False) -> tuple:
    """Fused pre-attention ops for single-frame streaming (Phase 2).

    Computes q/k/v from x, applies q_norm/k_norm/RoPE, and converts to
    [tpf, H, D] format ready for append_frame + compute_attention.

    The preparation region is compiled before the outer aggregator graph is
    captured. The immediate q/k casts are part of this runtime's baseline.
    """
    B, N, C = x.shape
    # qkv linear runs in autocast dtype (bf16). v stays in bf16.
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # Each: [B, num_heads, N, head_dim], bf16
    # q_norm/k_norm are LayerNorm and upcast to fp32 under autocast.  Cast back
    # to v's dtype immediately so RoPE and the FlashInfer kernel see bf16 — this
    # avoids a fp32→bf16 cast that previously ran inside compute_attention()
    # outside any compiled graph (showed up as bfloat16_copy_kernel in nsys).
    q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
    if self.rope is not None and not enable_3d_rope:
        q = self.rope(q, pos)
        k = self.rope(k, pos)
    elif self.rope is not None:  # enable_3d_rope=True
        q = apply_rotary_emb(q, pos)
        k = apply_rotary_emb(k, pos)
    # Convert to [tpf, H, D] format for FlashInfer (B=1 in streaming mode)
    q_nhd = q.squeeze(0).permute(1, 0, 2).contiguous()
    k_nhd = k.squeeze(0).permute(1, 0, 2).contiguous()
    v_nhd = v.squeeze(0).permute(1, 0, 2).contiguous()
    return q_nhd, k_nhd, v_nhd
