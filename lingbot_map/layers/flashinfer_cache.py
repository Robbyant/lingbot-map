"""
FlashInfer KV Cache Manager — Two-Stream Paged Design.

Two logical streams sharing one physical page pool per layer:

  Patch stream (recyclable):
    - page_size = patches_per_frame  (256 for 224×224; 972 for 504×378)
    - Exactly 1 patch page per frame
    - Scale frames  → scale_patch_pages  (never evicted, maxlen=scale_frames)
    - Recent frames → live_window_patch_pages (evicted when > sliding_window)

  Special stream (append-only, never recycled):
    - num_special_tokens (6) special tokens per frame
    - Packed continuously: one special page holds floor(page_size/6) frames
      e.g. page_size=256 → 42 frames per special page, 4 slots wasted
    - Specials written for EVERY frame (including scale + window), not just evicted ones.

Physical layout per block:
    kv_caches[block_idx]: [max_num_pages, 2, page_size, H, D]
      Pages 0 .. max_patch_pages-1        : patch page pool (recyclable)
      Pages max_patch_pages .. max_pages-1: special page pool (append-only)
      dim 1: 0=K  1=V

Attention computation:
    visible = scale_patch_pages + live_window_patch_pages + all_special_pages
    Special pages placed LAST → paged_kv_last_page_len naturally describes
    the partial special-tail without a custom mask.

    plan() is called ONCE per frame step (when block_idx == 0).
    run() is called per layer, reusing the same plan.  All layers at the
    same frame step have identical page structures (same page IDs in same
    positions), so reusing the plan across layers is correct.

Public API is drop-in compatible with the previous FlashInferKVCacheManager:
    append_frame(block_idx, k, v)
    evict_frames(block_idx, scale_frames, sliding_window, ...)
    compute_attention(block_idx, q) -> out
    reset()
"""

import collections
import math
from typing import List

import torch
from torch import Tensor

try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False


class FlashInferKVCacheManager:
    """
    Two-stream paged KV cache: patch pages (recyclable) + special pages (append-only).

    Args:
        num_blocks:          Number of Transformer blocks (one cache per block).
        max_num_frames:      Maximum frames held in the KV window at once
                             (scale_frames + sliding_window + headroom).
        tokens_per_frame:    Total tokens per frame = patches + specials (e.g. 262).
        num_heads:           Number of KV heads (= QO heads; MHA assumed).
        head_dim:            Head dimension (64 for ViT-L).
        dtype:               Storage dtype (bfloat16 / float16).
        device:              CUDA device.
        num_special_tokens:  Special tokens per frame: camera + register×N + scale (6).
        scale_frames:        Number of always-resident scale frames (8).
        sliding_window:      Sliding window size (64).
        max_total_frames:    Upper bound on total frames ever processed; used to
                             pre-allocate the special page pool (default 2048).
    """

    def __init__(
        self,
        num_blocks: int,
        max_num_frames: int,
        tokens_per_frame: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        num_special_tokens: int = 6,
        scale_frames: int = 8,
        sliding_window: int = 64,
        max_total_frames: int = 2048,
        force_fp32: bool = False,
        fa3: bool = False,
    ):
        if not FLASHINFER_AVAILABLE:
            raise RuntimeError("FlashInfer is not available. Please install flashinfer.")

        self.num_blocks = num_blocks
        self.num_special_tokens = num_special_tokens         # 6
        self.patches_per_frame = tokens_per_frame - num_special_tokens  # 256 / 999 / ...
        # Use exact page_size = patches_per_frame to eliminate zero-padded slots.
        # FA2 (backend="fa2") supports non-power-of-2 page sizes.
        # FA3 (sm90) requires power-of-2 page sizes; use next_power_of_2 when fa3=True.
        p = self.patches_per_frame
        if fa3:
            # Round up to next power-of-2 for FA3 SM90 kernel requirement.
            # e.g. 999 → 1024 (25 zero-padded slots per patch page)
            self.page_size = 1 << (p - 1).bit_length()
        else:
            self.page_size = p  # exact: no zero padding in patch pages
        self.scale_frames = scale_frames                     # 8
        self.sliding_window = sliding_window                 # 64
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.tokens_per_frame = tokens_per_frame

        assert self.patches_per_frame > 0, (
            f"tokens_per_frame={tokens_per_frame} <= num_special_tokens={num_special_tokens}"
        )
        assert self.page_size > 0

        # force_fp32: bypass FlashInfer FA2 kernel (which only supports fp16/bf16) and
        # instead gather paged K/V into a dense tensor and use F.scaled_dot_product_attention
        # in fp32 for accuracy comparison.  Storage dtype is also kept as fp32 in this mode.
        self.force_fp32 = force_fp32
        if force_fp32:
            self.dtype = torch.float32
        else:
            if dtype == torch.float32:
                dtype = torch.bfloat16
            self.dtype = dtype
        self.device = device

        # ── Page pool sizing ─────────────────────────────────────────────────
        # Patch: scale + window + 16 headroom  (pages recycled → fixed count)
        max_patch_pages = scale_frames + sliding_window + 16   # e.g. 88
        # Special: enough for max_total_frames × 6 tokens, plus 16 headroom
        max_special_pages = (
            math.ceil(max_total_frames * num_special_tokens / self.page_size) + 16
        )
        self.max_patch_pages = max_patch_pages
        self.max_num_pages = max_patch_pages + max_special_pages

        # ── Physical paged KV caches ─────────────────────────────────────────
        # Shape per block: [max_num_pages, 2, page_size, H, D]   (NHD, K=dim0, V=dim1)
        self.kv_caches: List[Tensor] = [
            torch.zeros(
                self.max_num_pages, 2, self.page_size, num_heads, head_dim,
                dtype=dtype, device=device,
            )
            for _ in range(num_blocks)
        ]

        # ── Per-block state ──────────────────────────────────────────────────
        # Patch pages (IDs 0 .. max_patch_pages-1)
        self.scale_patch_pages: List[collections.deque] = [
            collections.deque() for _ in range(num_blocks)
        ]
        self.live_window_patch_pages: List[collections.deque] = [
            collections.deque() for _ in range(num_blocks)
        ]
        self.free_patch_pages: List[List[int]] = [
            list(range(max_patch_pages)) for _ in range(num_blocks)
        ]

        # Special pages (IDs max_patch_pages .. max_num_pages-1)
        self.all_special_pages: List[List[int]] = [[] for _ in range(num_blocks)]
        self.free_special_pages: List[List[int]] = [
            list(range(max_patch_pages, self.max_num_pages)) for _ in range(num_blocks)
        ]
        self.special_token_count: List[int] = [0] * num_blocks

        # Frame counter per block (determines scale vs window routing)
        self.frame_count: List[int] = [0] * num_blocks

        # Deferred eviction support for flow-based keyframe selection.
        # When True, evict_frames() becomes a no-op; caller must later call
        # execute_deferred_eviction() or rollback_last_frame().
        self._defer_eviction: bool = False

        # ── FlashInfer wrapper ───────────────────────────────────────────────
        # plan() is called once per frame step (block_idx == 0).
        # run() is called per layer, reusing the same aux structures.
        # backend: "fa2" (default) or "fa3" (SM90/H100, requires power-of-2 page_size).
        # FA2 supports non-power-of-2 page sizes and avoids a FA3 NaN bug seen in
        # FlashInfer 0.2.5 at 518×378 resolution.
        _fi_backend = "fa3" if fa3 else "fa2"
        self.workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            kv_layout="NHD",
            backend=_fi_backend,
        )

        # plan() inputs (indices/indptr built fresh each step; qo_indptr is fixed)
        self._qo_indptr = torch.tensor(
            [0, tokens_per_frame], dtype=torch.int32, device=device
        )

    # =========================================================================
    # Public API  (drop-in compatible with previous FlashInferKVCacheManager)
    # =========================================================================

    def append_frame(self, block_idx: int, k: Tensor, v: Tensor) -> None:
        """
        Append one frame's K/V tensors to the two-stream cache.

        Token layout must be: [camera, reg0, ..., regN, scale, patch0, ..., patchP-1]
        i.e. specials come first (matching stream.py's patch_start_idx convention).

        Args:
            block_idx: Block/layer index (0 … num_blocks-1).
            k: [tokens_per_frame, H, D]  NHD layout.
            v: [tokens_per_frame, H, D]  NHD layout.
        """
        n = self.num_special_tokens  # 6
        sp_k    = k[:n].to(self.dtype)      # [6,   H, D]
        patch_k = k[n:].to(self.dtype)     # [256, H, D]
        sp_v    = v[:n].to(self.dtype)
        patch_v = v[n:].to(self.dtype)

        assert patch_k.shape[0] == self.patches_per_frame, (
            f"block {block_idx}: expected {self.patches_per_frame} patch tokens, "
            f"got {patch_k.shape[0]} (tokens_per_frame={k.shape[0]})"
        )

        self._write_patch_page(block_idx, patch_k, patch_v)
        self._write_special_tokens(block_idx, sp_k, sp_v)
        self.frame_count[block_idx] += 1

    def evict_frames(
        self,
        block_idx: int,
        scale_frames: int,
        sliding_window: int,
        cross_frame_special: bool = True,
        include_scale_frames: bool = True,
        camera_only: bool = False,
        num_register_tokens: int = 4,
    ) -> None:
        """
        Evict old window patch pages (recycle to free list).

        Special pages are NEVER evicted.
        Scale pages are NEVER evicted.
        Only live_window_patch_pages beyond `sliding_window` are recycled.

        When ``_defer_eviction`` is True, this method is a no-op.  The caller
        is expected to later call ``execute_deferred_eviction()`` (keep frame)
        or ``rollback_last_frame()`` (discard frame).
        """
        if self._defer_eviction:
            return
        while len(self.live_window_patch_pages[block_idx]) > sliding_window:
            old_page = self.live_window_patch_pages[block_idx].popleft()
            self.free_patch_pages[block_idx].append(old_page)

    def execute_deferred_eviction(
        self,
        block_idx: int,
        scale_frames: int,
        sliding_window: int,
        **kwargs,
    ) -> None:
        """Run the eviction that was skipped while ``_defer_eviction`` was True."""
        while len(self.live_window_patch_pages[block_idx]) > sliding_window:
            old_page = self.live_window_patch_pages[block_idx].popleft()
            self.free_patch_pages[block_idx].append(old_page)

    def rollback_last_frame(self, block_idx: int) -> None:
        """Undo the most recent ``append_frame()`` for *block_idx*.

        This reverses all three sub-operations of ``append_frame``:
        patch page allocation, special-token write, and frame_count increment.
        It must be called **before** any eviction for that frame (i.e. while
        ``_defer_eviction`` is True or before ``evict_frames`` is called).
        """
        assert self.frame_count[block_idx] > 0, (
            f"block {block_idx}: cannot rollback, frame_count is 0"
        )

        # 1) Undo patch page ── pop from whichever deque it was routed to.
        if self.frame_count[block_idx] > self.scale_frames:
            page_id = self.live_window_patch_pages[block_idx].pop()
        else:
            page_id = self.scale_patch_pages[block_idx].pop()
        self.free_patch_pages[block_idx].append(page_id)

        # 2) Undo special tokens
        n = self.num_special_tokens
        new_count = self.special_token_count[block_idx] - n
        assert new_count >= 0, (
            f"block {block_idx}: special_token_count underflow "
            f"({self.special_token_count[block_idx]} - {n})"
        )
        new_num_pages = math.ceil(new_count / self.page_size) if new_count > 0 else 0
        while len(self.all_special_pages[block_idx]) > new_num_pages:
            freed = self.all_special_pages[block_idx].pop()
            self.free_special_pages[block_idx].append(freed)
        self.special_token_count[block_idx] = new_count

        # 3) Decrement frame count
        self.frame_count[block_idx] -= 1

    def get_cache_stats(self, block_idx: int = 0) -> dict:
        """Read-only snapshot of cache occupancy for one block.

        Useful for debugging keyframe / sliding-window behavior.

        Returns:
            dict with keys:
              - ``frame_count``   total frames ever appended (minus rollbacks)
              - ``scale_pages``   scale-region patch pages currently held
              - ``live_pages``    sliding-window patch pages currently held
              - ``free_pages``    patch pages on the free list
              - ``special_tokens`` running count of special tokens written
        """
        return {
            "frame_count":    int(self.frame_count[block_idx]),
            "scale_pages":    len(self.scale_patch_pages[block_idx]),
            "live_pages":     len(self.live_window_patch_pages[block_idx]),
            "free_pages":     len(self.free_patch_pages[block_idx]),
            "special_tokens": int(self.special_token_count[block_idx]),
        }

    def _gather_kv(self, block_idx: int):
        """
        Gather all visible K and V tokens from the paged cache into dense tensors.

        Used by force_fp32 mode to bypass the FlashInfer FA2 kernel (which only
        supports fp16/bf16) and instead run F.scaled_dot_product_attention in fp32.

        Returns:
            k_flat: [kv_len, H, D]  — all visible K tokens concatenated
            v_flat: [kv_len, H, D]  — all visible V tokens concatenated
        """
        visible  = self.build_visible_page_table(block_idx)
        last_len = self.compute_last_page_len(block_idx)
        P = self.page_size

        parts_k, parts_v = [], []
        for i, pid in enumerate(visible):
            n = last_len if (i == len(visible) - 1) else P
            parts_k.append(self.kv_caches[block_idx][pid, 0, :n])  # [n, H, D]
            parts_v.append(self.kv_caches[block_idx][pid, 1, :n])

        k_flat = torch.cat(parts_k, dim=0)  # [kv_len, H, D]
        v_flat = torch.cat(parts_v, dim=0)
        return k_flat, v_flat

    def compute_attention(self, block_idx: int, q: Tensor) -> Tensor:
        """
        Compute cross-frame attention using FlashInfer BatchPrefillWithPagedKVCacheWrapper.

        When self.force_fp32 is True, gathers all visible K/V into dense tensors
        and uses F.scaled_dot_product_attention in fp32 instead of the FA2 kernel.
        This is used for accuracy comparison since FlashInfer FA2 only supports fp16/bf16.

        plan() is called once per frame step (when block_idx == 0).
        All layers at the same step share the same visible page structure,
        so the plan is reused by calling run() with each layer's kv_cache.

        Args:
            block_idx: Block/layer index.
            q: [q_len, H, D]  NHD layout (q_len = tokens_per_frame = 262).

        Returns:
            out: [q_len, H, D]
        """
        if self.frame_count[block_idx] == 0:
            # No KV present yet (should not occur in normal usage after append_frame)
            return torch.zeros_like(q)

        if self.force_fp32:
            # ── fp32 gather+SDPA path ─────────────────────────────────────────
            # Gather visible K/V from paged cache and run SDPA in fp32.
            # This bypasses the FlashInfer FA2 kernel (fp16/bf16 only) for accuracy.
            # q_len, H, D → 1, H, q_len, D  (SDPA expects BHsD layout)
            import torch.nn.functional as F_nn
            k_flat, v_flat = self._gather_kv(block_idx)
            q_b = q.float().permute(1, 0, 2).unsqueeze(0)      # [1, H, q_len, D]
            k_b = k_flat.float().permute(1, 0, 2).unsqueeze(0) # [1, H, kv_len, D]
            v_b = v_flat.float().permute(1, 0, 2).unsqueeze(0) # [1, H, kv_len, D]
            out = F_nn.scaled_dot_product_attention(q_b, k_b, v_b)
            return out.squeeze(0).permute(1, 0, 2).to(q.dtype) # [q_len, H, D]

        if block_idx == 0:
            # ── Plan once per frame step ──────────────────────────────────────
            # Build visible page table from block 0's state.
            # All blocks have identical page structures, so this plan is valid
            # for all subsequent run() calls (block_idx = 1, 2, ...).
            visible  = self.build_visible_page_table(0)
            last_len = self.compute_last_page_len(0)

            assert visible, "visible page table is empty after append_frame"
            assert 1 <= last_len <= self.page_size, (
                f"block 0: last_page_len={last_len} out of [1, {self.page_size}]"
            )

            paged_kv_indices       = torch.tensor(visible, dtype=torch.int32, device=self.device)
            paged_kv_indptr        = torch.tensor([0, len(visible)], dtype=torch.int32, device=self.device)
            paged_kv_last_page_len = torch.tensor([last_len], dtype=torch.int32, device=self.device)

            self.prefill_wrapper.plan(
                self._qo_indptr,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                num_qo_heads      = self.num_heads,
                num_kv_heads      = self.num_heads,
                head_dim_qk       = self.head_dim,
                page_size         = self.page_size,
                causal            = False,          # custom page ordering; no causal mask
                pos_encoding_mode = "NONE",         # RoPE applied externally before append
                q_data_type       = self.dtype,
            )

        # ── Run attention for this layer ──────────────────────────────────────
        # Cast q to storage dtype (LayerNorm may upcast to float32 under autocast).
        return self.prefill_wrapper.run(
            q              = q.to(self.dtype).contiguous(),
            paged_kv_cache = self.kv_caches[block_idx],
        )  # → [q_len, H, D]

    def reset(self) -> None:
        """Reset all per-block state for a new sequence."""
        for i in range(self.num_blocks):
            self.scale_patch_pages[i].clear()
            self.live_window_patch_pages[i].clear()
            self.all_special_pages[i].clear()
            self.free_patch_pages[i]   = list(range(self.max_patch_pages))
            self.free_special_pages[i] = list(range(self.max_patch_pages, self.max_num_pages))
            self.special_token_count[i] = 0
            self.frame_count[i] = 0

    # =========================================================================
    # Helper methods
    # =========================================================================

    def build_visible_page_table(self, block_idx: int) -> List[int]:
        """
        Return page IDs in strict order: scale → window → special.

        Placing special pages last means only the final page may be partially
        full, so paged_kv_last_page_len = compute_last_page_len() is sufficient
        without a custom attention mask.
        """
        return (
            list(self.scale_patch_pages[block_idx])       +
            list(self.live_window_patch_pages[block_idx]) +
            list(self.all_special_pages[block_idx])
        )

    def compute_last_page_len(self, block_idx: int) -> int:
        """
        Valid token count in the last page of the visible sequence.

        - No special pages      → last page is a patch page.
                                  Returns patches_per_frame (real tokens written),
                                  which may be < page_size when page_size was rounded
                                  up to a power of 2.
        - Special tail partial  → special_token_count % page_size.
        - Special tail exactly full → page_size.
        """
        if not self.all_special_pages[block_idx]:
            # Last page is a patch page.  We wrote patches_per_frame tokens (0..P-1);
            # positions P..page_size-1 are zero padding.  Tell FlashInfer the true
            # valid count so it doesn't read beyond the real tokens.
            return self.patches_per_frame

        tail = self.special_token_count[block_idx] % self.page_size
        return self.page_size if tail == 0 else tail

    # ── Internal write helpers ────────────────────────────────────────────────

    def _write_patch_page(self, block_idx: int, patch_k: Tensor, patch_v: Tensor) -> int:
        """
        Allocate one free patch page and write patches_per_frame patch tokens.

        Direct tensor assignment to kv_caches[block_idx][page_id, 0/1] avoids
        the Python→C++/CUDA dispatch overhead of flashinfer.page.append_paged_kv_cache.
        kv_caches layout: [max_num_pages, 2, page_size, H, D]  (NHD, K=0, V=1).
        patch_k/v fill exactly one full page (patches_per_frame == page_size).

        Routes to scale_patch_pages if still filling scale quota,
        otherwise to live_window_patch_pages.

        Returns:
            page_id: Physical page index used.
        """
        assert self.free_patch_pages[block_idx], (
            f"block {block_idx}: patch page pool exhausted — "
            f"scale={len(self.scale_patch_pages[block_idx])}, "
            f"window={len(self.live_window_patch_pages[block_idx])}, "
            f"free={len(self.free_patch_pages[block_idx])}"
        )

        page_id = self.free_patch_pages[block_idx].pop()

        # Direct slice write: positions 0..patches_per_frame-1.
        # When page_size == patches_per_frame (power-of-2 aligned, e.g. 256 for 224×224),
        # this is equivalent to a full-page write.  When page_size > patches_per_frame
        # (rounded up for FA3 alignment, e.g. page_size=1024 for patches_per_frame=999),
        # positions patches_per_frame..page_size-1 remain zero (kv_caches is zero-init).
        P = self.patches_per_frame
        self.kv_caches[block_idx][page_id, 0, :P] = patch_k  # K
        self.kv_caches[block_idx][page_id, 1, :P] = patch_v  # V

        if len(self.scale_patch_pages[block_idx]) < self.scale_frames:
            self.scale_patch_pages[block_idx].append(page_id)
        else:
            self.live_window_patch_pages[block_idx].append(page_id)

        return page_id

    def _write_special_tokens(self, block_idx: int, sp_k: Tensor, sp_v: Tensor) -> None:
        """
        Append num_special_tokens (6) special tokens to the special stream.

        Direct tensor slice assignment to kv_caches[block_idx][tail_page, 0/1,
        tail_offset : tail_offset+write_n] avoids the Python→C++/CUDA dispatch
        overhead of flashinfer.page.append_paged_kv_cache.

        Handles page-boundary crossing: if 6 tokens straddle two pages, performs
        two slice writes (rare — page_size=256 >> 6).
        """
        remaining = self.num_special_tokens   # 6
        written   = 0

        while remaining > 0:
            tail_offset = self.special_token_count[block_idx] % self.page_size

            if tail_offset == 0:
                # Current tail page is full (or no page exists) — allocate a new one
                assert self.free_special_pages[block_idx], (
                    f"block {block_idx}: special page pool exhausted at "
                    f"special_token_count={self.special_token_count[block_idx]}. "
                    f"Increase max_total_frames."
                )
                new_page = self.free_special_pages[block_idx].pop()
                self.all_special_pages[block_idx].append(new_page)

            tail_page = self.all_special_pages[block_idx][-1]
            space     = self.page_size - tail_offset   # free slots in tail page
            write_n   = min(remaining, space)

            # Direct slice write: kv_caches[block_idx][tail_page, 0/1, offset:offset+n]
            # shape: [page_size, H, D];  slice [tail_offset:tail_offset+write_n, :, :]
            end = tail_offset + write_n
            self.kv_caches[block_idx][tail_page, 0, tail_offset:end] = sp_k[written:written + write_n]
            self.kv_caches[block_idx][tail_page, 1, tail_offset:end] = sp_v[written:written + write_n]

            self.special_token_count[block_idx] += write_n
            written   += write_n
            remaining -= write_n

    # ── Legacy property (used by stream.py) ──────────────────────────────────

    @property
    def num_frames(self) -> int:
        """Number of frames appended to block 0 (representative)."""
        return self.frame_count[0] if self.frame_count else 0


# =============================================================================
# Sanity check
# =============================================================================

def _sanity_check():
    """
    Minimal smoke test.
    Run with:  python -c "from lingbot_map.layers.flashinfer_cache import _sanity_check; _sanity_check()"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[sanity_check] CUDA not available — skipping.")
        return

    tokens_per_frame  = 262   # 256 patch + 6 special (224×224)
    num_special       = 6
    patches_per_frame = tokens_per_frame - num_special  # 256
    page_size         = patches_per_frame               # 256

    mgr = FlashInferKVCacheManager(
        num_blocks         = 2,
        max_num_frames     = 88,
        tokens_per_frame   = tokens_per_frame,
        num_heads          = 16,
        head_dim           = 64,
        dtype              = torch.bfloat16,
        device             = device,
        num_special_tokens = num_special,
        scale_frames       = 8,
        sliding_window     = 64,
        max_total_frames   = 200,
    )

    def make_kv():
        k = torch.randn(tokens_per_frame, 16, 64, dtype=torch.bfloat16, device=device)
        v = torch.randn(tokens_per_frame, 16, 64, dtype=torch.bfloat16, device=device)
        return k, v

    def make_q():
        return torch.randn(tokens_per_frame, 16, 64, dtype=torch.bfloat16, device=device)

    for block in range(2):
        for t in range(100):
            k, v = make_kv()
            mgr.append_frame(block, k, v)
            mgr.evict_frames(block, scale_frames=8, sliding_window=64)

        # ── Page count checks ───────────────────────────────────────────────
        n_scale  = len(mgr.scale_patch_pages[block])
        n_window = len(mgr.live_window_patch_pages[block])
        n_spec   = len(mgr.all_special_pages[block])
        sp_count = mgr.special_token_count[block]

        assert n_scale  == 8,  f"block {block}: scale pages = {n_scale},  expected 8"
        assert n_window == 64, f"block {block}: window pages = {n_window}, expected 64"
        # 100 frames × 6 specials = 600 tokens; ceil(600/256) = 3 pages
        expected_spec_pages = math.ceil(100 * num_special / page_size)
        assert n_spec == expected_spec_pages, (
            f"block {block}: special pages = {n_spec}, expected {expected_spec_pages}"
        )
        assert sp_count == 100 * num_special, (
            f"block {block}: special_token_count = {sp_count}, expected {100*num_special}"
        )

        # ── last_page_len ────────────────────────────────────────────────────
        last_len = mgr.compute_last_page_len(block)
        tail = sp_count % page_size
        expected_len = page_size if tail == 0 else tail
        assert last_len == expected_len, f"block {block}: last_len={last_len}, expected={expected_len}"

        # ── visible page table order ─────────────────────────────────────────
        visible = mgr.build_visible_page_table(block)
        assert len(visible) == n_scale + n_window + n_spec, "visible page count mismatch"
        for pid in visible[:n_scale + n_window]:
            assert pid < mgr.max_patch_pages, f"patch page {pid} out of patch range"
        for pid in visible[n_scale + n_window:]:
            assert pid >= mgr.max_patch_pages, f"special page {pid} not in special range"

        # ── forward pass: plan() once for block 0, run() for both blocks ─────
        if block == 1:
            # Simulate the actual calling pattern: plan on block 0, run on both
            q0 = make_q()
            out0 = mgr.compute_attention(0, q0)   # triggers plan()
            q1 = make_q()
            out1 = mgr.compute_attention(1, q1)   # reuses plan, different kv_cache
            assert out0.shape == (tokens_per_frame, 16, 64)
            assert out1.shape == (tokens_per_frame, 16, 64)

        print(f"[block {block}] PASS: scale={n_scale}, window={n_window}, "
              f"special_pages={n_spec}, special_tokens={sp_count}, "
              f"last_page_len={last_len}")

    mgr.reset()
    assert mgr.frame_count[0] == 0
    print("\n[sanity_check] All assertions passed.")


if __name__ == "__main__":
    _sanity_check()
