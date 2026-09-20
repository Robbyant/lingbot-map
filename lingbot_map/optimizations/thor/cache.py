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

Public API follows the previous FlashInferKVCacheManager for every-frame append:
    append_frame(block_idx, k, v)
    evict_frames(block_idx, scale_frames, sliding_window, ...)
    compute_attention(block_idx, q) -> out
    reset()

Dynamic keyframes and deferred appends are rejected before cache mutation.
"""

import collections
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import List, Optional

import torch
from torch import Tensor

from lingbot_map.layers.flashinfer_cache import (
    FlashInferKVCacheManager as BaseFlashInferKVCacheManager,
)

from .fa4_runtime import (
    candidate022_fa4_b15_enabled,
    prepare_candidate022_fa4_b15,
)
from .kv_append import (
    candidate021_direct_append_loaded_module_fingerprint,
    candidate021_direct_append_paged_kv_cache,
    validate_candidate021_direct_append_runtime,
)

try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False


CANDIDATE021_DIRECT_KV_APPEND_ROUTE = "direct_kv_append"
CANDIDATE021_PROJECTION_SHADOW_DIRECT_KV_APPEND_ROUTE = (
    "projection_shadow_direct_kv_append"
)
CANDIDATE021_FA4_VERSION = "4.0.0b14"
CANDIDATE021_ALLOWED_TOKENS_PER_FRAME = frozenset((783, 978, 1005, 1042))
CANDIDATE021_FA4_SOURCE_HASHES = {
    "interface.py": "27aa8985be0464bc872cf773d39c1b3266ce6f1b3dd2ed63e0f9829861b59c9b",
    "flash_fwd_sm100.py": "327dc9741515af112d09555fe5641ba855f45c018f7cddd61bf5c9caf704b2ed",
    "paged_kv.py": "2aafce6dc0bb6fdaa683c4067fc18c34d6232a25d05c090dcf2b9a9350efc009",
    "tile_scheduler.py": "a2f36e052733e7aa67a0f78fe1f9e4c414dcd0ddb554a22d471cafb5562ef489",
}


def _resolve_candidate021_route() -> Optional[str]:
    from .options import housekeeping_route
    return housekeeping_route()


def _candidate021_direct_kv_append_enabled(route: Optional[str]) -> bool:
    """Whether a route owns the exact Candidate 021C2 append capability."""
    return route in (
        CANDIDATE021_DIRECT_KV_APPEND_ROUTE,
        CANDIDATE021_PROJECTION_SHADOW_DIRECT_KV_APPEND_ROUTE,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FlashInferKVCacheManager(BaseFlashInferKVCacheManager):
    """
    Two-stream paged KV cache: patch pages (recyclable) + special pages (append-only).

    Retains the original manager type contract so scale attention populates this
    cache through the unmodified FlashInferAttention.forward dispatch.

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
        backend: Optional[str] = None,
    ):
        if not FLASHINFER_AVAILABLE:
            raise RuntimeError("FlashInfer is not available. Please install flashinfer.")

        self.num_blocks = num_blocks
        self.num_special_tokens = num_special_tokens         # 6
        self.patches_per_frame = tokens_per_frame - num_special_tokens  # 256 / 999 / ...
        # Use exact page_size = patches_per_frame to eliminate zero-padded slots.
        # The current FlashInfer fa3 backend is the SM90/Hopper path and requires
        # power-of-2 pages; FA4 and other paged backends can use the exact patch count.
        self.backend = backend if backend is not None else ("fa3" if fa3 else "fa2")
        self.use_fa4 = self.backend == "fa4"
        backend_may_select_sm90_fa3 = False
        if self.backend == "auto" and torch.cuda.is_available():
            backend_may_select_sm90_fa3 = torch.cuda.get_device_capability(device)[0] == 9
        p = self.patches_per_frame
        if self.backend == "fa3" or backend_may_select_sm90_fa3:
            # Round up to next power-of-2 for the current FA3 SM90 kernel requirement.
            # e.g. 999 → 1024 (25 zero-padded slots per patch page)
            self.page_size = 1 << (p - 1).bit_length()
        else:
            self.page_size = p  # exact: no zero padding in patch pages
        self.scale_frames = scale_frames                     # 8
        self.sliding_window = sliding_window                 # 64
        self.visible_window = sliding_window                 # opt-in attention budget
        self.visible_plan_calls = 0
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
        self.candidate021_route = _resolve_candidate021_route()
        self.candidate021_metadata = None
        self.candidate022_metadata = None
        self._candidate021_direct_runtime_contract = None
        self._candidate021_direct_loaded_module = None
        if candidate022_fa4_b15_enabled() and not self.use_fa4:
            raise RuntimeError(
                "Candidate 022 FA4 beta15 contract rejected: "
                f"backend={self.backend!r}, expected 'fa4'"
            )
        if self.candidate021_route is not None:
            candidate_device = torch.device(device)
            candidate_cc = (
                torch.cuda.get_device_capability(candidate_device)
                if candidate_device.type == "cuda" and torch.cuda.is_available()
                else None
            )
            contract_errors = []
            if not self.use_fa4:
                contract_errors.append(f"backend={self.backend!r}, expected 'fa4'")
            if self.force_fp32:
                contract_errors.append("force_fp32 must be false")
            if self.dtype != torch.bfloat16:
                contract_errors.append(f"dtype={self.dtype}, expected torch.bfloat16")
            if candidate_device.type != "cuda":
                contract_errors.append(f"device={candidate_device}, expected CUDA")
            if candidate_cc != (11, 0):
                contract_errors.append(f"compute_capability={candidate_cc}, expected (11, 0)")
            if self.num_blocks != 24:
                contract_errors.append(f"num_blocks={self.num_blocks}, expected 24")
            if self.num_heads != 16:
                contract_errors.append(f"num_heads={self.num_heads}, expected 16")
            if self.head_dim != 64:
                contract_errors.append(f"head_dim={self.head_dim}, expected 64")
            if self.num_special_tokens != 6:
                contract_errors.append(
                    f"num_special_tokens={self.num_special_tokens}, expected 6"
                )
            if self.tokens_per_frame not in CANDIDATE021_ALLOWED_TOKENS_PER_FRAME:
                contract_errors.append(
                    f"tokens_per_frame={self.tokens_per_frame}, expected one of "
                    f"{sorted(CANDIDATE021_ALLOWED_TOKENS_PER_FRAME)}"
                )
            if self.page_size != self.patches_per_frame:
                contract_errors.append(
                    f"page_size={self.page_size}, expected exact patch page "
                    f"{self.patches_per_frame}"
                )
            if contract_errors:
                raise RuntimeError(
                    "Candidate 021 cache contract rejected: "
                    + "; ".join(contract_errors)
                )
            if _candidate021_direct_kv_append_enabled(self.candidate021_route):
                self._candidate021_direct_runtime_contract = (
                    validate_candidate021_direct_append_runtime()
                )
                self._candidate021_direct_loaded_module = (
                    candidate021_direct_append_loaded_module_fingerprint()
                )

        # Candidate 022 validates and selects its process-wide FA4 source before
        # allocating the large cache and workspace buffers below.
        if self.use_fa4:
            self.candidate022_metadata = prepare_candidate022_fa4_b15()

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

        # pop() must allocate ascending IDs, matching the graph-mode page tables.
        self.all_special_pages: List[List[int]] = [[] for _ in range(num_blocks)]
        self.free_special_pages: List[List[int]] = [
            list(range(self.max_num_pages - 1, max_patch_pages - 1, -1))
            for _ in range(num_blocks)
        ]
        self.special_token_count: List[int] = [0] * num_blocks

        # Frame counter per block (determines scale vs window routing)
        self.frame_count: List[int] = [0] * num_blocks

        # Retain the upstream policy flag so unsupported appends can be rejected.
        self._defer_eviction: bool = False

        # ── Attention wrapper ────────────────────────────────────────────────
        # plan() is called once per frame step (block_idx == 0).
        # run() is called per layer, reusing the same aux structures.
        # backend examples: "fa2", "auto", "fa3" (current SM90/Hopper path),
        # "fa4" (official FlashAttention-4 CuTe paged path), or "trtllm-gen"
        # (paged Blackwell candidate in FlashInfer 0.6.x).
        _fi_backend = self.backend
        self.workspace_buffer = None
        self._qo_indptr_buf_gpu = None
        self._kv_indptr_buf_gpu = None
        self._kv_indices_buf_gpu = None
        self._kv_last_page_len_buf_gpu = None

        self.prefill_wrapper = None
        self._fa4_flash_attn_varlen_func = None
        self._fa4_cu_q_gpu = torch.tensor(
            [0, tokens_per_frame], dtype=torch.int32, device=device
        )
        self._fa4_seqused_k_gpu = torch.zeros(1, dtype=torch.int32, device=device)
        self._fa4_page_table_gpu = torch.empty(
            1, self.max_num_pages, dtype=torch.int32, device=device
        )
        self._fa4_page_table_gpu[0].copy_(
            torch.arange(self.max_num_pages, dtype=torch.int32, device=device)
        )
        if self.use_fa4:
            try:
                from flash_attn.cute import flash_attn_varlen_func
            except ImportError as exc:
                raise RuntimeError(
                    "backend='fa4' requires official flash-attn-4 "
                    "(install with `uv pip install --prerelease=allow "
                    "\"flash-attn-4[cu13]\"`)."
                ) from exc
            self._fa4_flash_attn_varlen_func = flash_attn_varlen_func
            if self.candidate021_route is not None:
                import flash_attn.cute.interface as fa4_interface

                cute_root = Path(fa4_interface.__file__).resolve().parent
                runtime_errors = []
                if self.candidate022_metadata is not None:
                    fa4_version = self.candidate022_metadata["version"]
                    observed_hashes = None
                else:
                    try:
                        fa4_version = importlib.metadata.version("flash-attn-4")
                    except importlib.metadata.PackageNotFoundError as exc:
                        raise RuntimeError(
                            "Candidate 021 requires the flash-attn-4 distribution metadata"
                        ) from exc
                    expected_source_hashes = CANDIDATE021_FA4_SOURCE_HASHES
                    observed_hashes = {
                        name: _sha256_file(cute_root / name)
                        for name in expected_source_hashes
                    }
                    if fa4_version != CANDIDATE021_FA4_VERSION:
                        runtime_errors.append(
                            f"flash-attn-4={fa4_version}, expected {CANDIDATE021_FA4_VERSION}"
                        )
                    for name, expected in expected_source_hashes.items():
                        observed = observed_hashes[name]
                        if observed != expected:
                            runtime_errors.append(
                                f"{name} sha256={observed}, expected {expected}"
                            )
                if runtime_errors:
                    raise RuntimeError(
                        "Candidate 021 FA4 runtime rejected: "
                        + "; ".join(runtime_errors)
                    )
                is_attention_kernel_route = self.candidate022_metadata is not None
                is_direct_kv_append_route = _candidate021_direct_kv_append_enabled(
                    self.candidate021_route
                )
                self.candidate021_metadata = {
                    "enabled": True,
                    "route": self.candidate021_route,
                    "kernel_route_applied": is_attention_kernel_route,
                    "kv_append_route_applied": is_direct_kv_append_route,
                    "kv_append_impl": (
                        "combined_flashinfer_append"
                        if is_direct_kv_append_route
                        else "accepted_split_append"
                    ),
                    "attention_impl": (
                        "public_varlen_beta15_qstage2"
                        if self.candidate022_metadata is not None
                        else "public_varlen"
                    ),
                    "expected_scheduler": "SingleTileVarlenScheduler",
                    "tile_mn": [128, 128],
                    "kv_stage": 12,
                    "compute_capability": list(
                        torch.cuda.get_device_capability(torch.device(self.device))
                    ),
                    "dtype": str(self.dtype),
                    "tokens_per_frame": self.tokens_per_frame,
                    "num_heads": self.num_heads,
                    "head_dim": self.head_dim,
                    "page_size": self.page_size,
                    "q_batch": 1,
                    "fa4_version": fa4_version,
                    "fa4_source_hashes": observed_hashes,
                    "candidate022_fa4_b15": self.candidate022_metadata,
                    "direct_kv_append_runtime": (
                        {
                            **self._candidate021_direct_runtime_contract,
                            "loaded_module": self._candidate021_direct_loaded_module,
                        }
                        if is_direct_kv_append_route
                        else None
                    ),
                }
                print(
                    "Candidate021 kernel route enabled: "
                    + json.dumps(self.candidate021_metadata, sort_keys=True)
                )
        else:
            # Only the FlashInfer wrapper uses this workspace and fixed-address
            # metadata. The public FA4 call uses its own inputs above.
            self.workspace_buffer = torch.zeros(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            self._qo_indptr_buf_gpu = torch.zeros(2, dtype=torch.int32, device=device)
            self._kv_indptr_buf_gpu = torch.zeros(2, dtype=torch.int32, device=device)
            self._kv_indices_buf_gpu = torch.zeros(self.max_num_pages, dtype=torch.int32, device=device)
            self._kv_last_page_len_buf_gpu = torch.zeros(1, dtype=torch.int32, device=device)
            self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer,
                kv_layout="NHD",
                backend=_fi_backend,
                use_cuda_graph=True,
                qo_indptr_buf=self._qo_indptr_buf_gpu,
                paged_kv_indptr_buf=self._kv_indptr_buf_gpu,
                paged_kv_indices_buf=self._kv_indices_buf_gpu,
                paged_kv_last_page_len_buf=self._kv_last_page_len_buf_gpu,
            )


        # ── plan() inputs (CPU pinned, no syncs) ─────────────────────────────
        # plan() does `qo_indptr.to("cpu")`, `paged_kv_indptr.to("cpu")` and
        # `paged_kv_last_page_len.to("cpu")` to derive host-side metadata.  When the
        # inputs already live on the host these `.to("cpu")` calls are no-ops, killing
        # three D→H syncs/frame.  plan()'s subsequent
        # ``self._*_buf.copy_(arg, non_blocking=True)`` then becomes an async pinned
        # H→D that pipelines behind the running GPU work — Python doesn't block on a
        # sync, GPU keeps draining the trunk.
        # qo_indptr is constant ([0, tpf]) so we fill once and never touch again.
        self._qo_indptr = torch.tensor(
            [0, tokens_per_frame], dtype=torch.int32, pin_memory=True
        )
        self._kv_indptr_buf = torch.zeros(2, dtype=torch.int32, pin_memory=True)
        self._kv_indices_buf = torch.zeros(self.max_num_pages, dtype=torch.int32, pin_memory=True)
        self._kv_last_page_len_buf = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        # Pass seq_lens explicitly so plan() skips its `get_seq_lens(...)` derivation
        # (which would otherwise force a sync on paged_kv_indptr/last_page_len even on
        # CPU inputs).  We already know the value on the host: see compute_attention.
        self._seq_lens_buf = torch.zeros(1, dtype=torch.int32, pin_memory=True)

        # FlashInfer run() accepts reusable output buffers; the FA4 call does not.
        self._attn_out_buffers: List[Tensor] = []
        if not self.use_fa4:
            self._attn_out_buffers = [
                torch.empty(
                    tokens_per_frame, num_heads, head_dim,
                    dtype=self.dtype, device=device,
                )
                for _ in range(num_blocks)
            ]

        # ── Graph-capture mode scaffolding (default OFF) ────────────────────
        # Set ``_graph_mode = True`` to switch ``append_frame``/``compute_attention``
        # to tensor-driven, Python-state-stable variants suitable for capture inside
        # ``torch.cuda.graph``.  In that mode the caller is responsible for invoking
        # ``prepare_frame_for_graph(frame_idx)`` OUTSIDE the captured graph each
        # frame to update Python state, refresh the host-pinned plan inputs, set the
        # CUDA index buffers consumed by the in-graph writes, and call ``plan()``.
        self._graph_mode: bool = False

        # Cyclic patch-page ID for the current frame.  Read by the in-graph
        # `index_copy_` that lands new K/V into the patch page pool.
        # ``index_copy_`` requires int64 indices (a torch limitation); using int32
        # raises ``RuntimeError: Expected a long tensor for index``.
        self._patch_write_page_id_buf = torch.zeros(1, dtype=torch.int64, device=device)
        # Padded scratch buffer shared across all 24 layers within one frame.  Each
        # layer overwrites positions [0, patches_per_frame) with its own K/V before
        # the index_copy_; positions [patches_per_frame, page_size) stay zero
        # forever (they get copied into kv_caches but are never read by FA3 because
        # patch pages are never the "last page" in the visible page table).
        self._patch_write_pad_buf = torch.zeros(
            1, 2, self.page_size, num_heads, head_dim,
            dtype=self.dtype, device=device,
        )

        # Special-stream `flashinfer.page.append_paged_kv_cache` inputs.  All host-
        # writable, all consumed by a single kernel invocation per layer × frame.
        # `_special_positions_buf[i]` = current_total_specials + i  for i ∈ [0, 6).
        # The buf is updated per frame via `torch.add(base, offset, out=buf)`,
        # avoiding a per-frame `torch.arange` allocation + slice copy.
        self._candidate021_direct_positions_buf = None
        if _candidate021_direct_kv_append_enabled(self.candidate021_route):
            # Preserve the accepted six-position buffer as a view while extending
            # the same stable storage with constant patch positions [0, P).
            self._candidate021_direct_positions_buf = torch.empty(
                tokens_per_frame, dtype=torch.int32, device=device,
            )
            self._candidate021_direct_positions_buf[:num_special_tokens].zero_()
            self._candidate021_direct_positions_buf[num_special_tokens:].copy_(
                torch.arange(self.patches_per_frame, dtype=torch.int32, device=device)
            )
            self._special_positions_buf = (
                self._candidate021_direct_positions_buf[:num_special_tokens]
            )
        else:
            self._special_positions_buf = torch.zeros(
                num_special_tokens, dtype=torch.int32, device=device,
            )
        self._special_positions_base = torch.arange(
            0, num_special_tokens, dtype=torch.int32, device=device,
        )
        # batch_indices is constant zeros (single-batch streaming inference).
        self._special_batch_indices_buf = torch.zeros(
            num_special_tokens, dtype=torch.int32, device=device,
        )
        # Special-only page table: separate from the wrapper's combined page table
        # (patches + specials) used by run().  append_paged_kv_cache reads this table.
        # Pre-baked once: graph mode allocates special pages monotonically from
        # max_patch_pages, so the kernel always reads the first n_sp_pages entries
        # which hold [max_pp, max_pp+1, ..., max_pp+n_sp_pages-1].  No per-frame
        # rewrite needed (saves a `torch.tensor(list, ...)` GPU allocation).
        self._special_kv_indices_buf = torch.arange(
            max_patch_pages, self.max_num_pages, dtype=torch.int32, device=device,
        )
        self._special_kv_indptr_buf = torch.zeros(2, dtype=torch.int32, device=device)
        self._special_kv_last_page_len_buf = torch.zeros(1, dtype=torch.int32, device=device)

        # Candidate 021C2 encodes patch and special writes as two logical
        # sequences while retaining the original [specials, patches] K/V order.
        # Every tensor has a stable address and the dynamic scalar fields are
        # updated outside replay by prepare_frame_for_graph().
        self._candidate021_direct_batch_indices_buf = None
        self._candidate021_direct_kv_indices_buf = None
        self._candidate021_direct_kv_indptr_buf = None
        self._candidate021_direct_kv_last_page_len_buf = None
        if _candidate021_direct_kv_append_enabled(self.candidate021_route):
            self._candidate021_direct_batch_indices_buf = torch.zeros(
                tokens_per_frame, dtype=torch.int32, device=device,
            )
            self._candidate021_direct_batch_indices_buf[:num_special_tokens].fill_(1)
            self._candidate021_direct_kv_indices_buf = torch.empty(
                1 + self._special_kv_indices_buf.numel(),
                dtype=torch.int32,
                device=device,
            )
            self._candidate021_direct_kv_indices_buf[0] = 0
            self._candidate021_direct_kv_indices_buf[1:].copy_(
                self._special_kv_indices_buf
            )
            self._candidate021_direct_kv_indptr_buf = torch.tensor(
                [0, 1, 1], dtype=torch.int32, device=device,
            )
            self._candidate021_direct_kv_last_page_len_buf = torch.tensor(
                [self.patches_per_frame, 0], dtype=torch.int32, device=device,
            )

        # Track whether ``_kv_indices_buf`` (host-pinned plan input) currently
        # holds the steady-state layout: ``[0..sf+sw-1, max_pp..max_pp+max_sp-1]``.
        # In steady state plan() reads ``_kv_indices_buf[:sf+sw+n_sp_pages]`` which
        # yields the correct visible page table without per-frame writes.  Any
        # warmup-phase / eager-path write to this buffer flips the flag to False;
        # the first subsequent steady-state graph-mode frame restores the layout.
        self._kv_indices_buf_in_steady_layout: bool = False

        self._candidate021_direct_capture_ready = False
        self._candidate021_direct_contract = None
        if _candidate021_direct_kv_append_enabled(self.candidate021_route):
            expected_batch = torch.cat(
                (
                    torch.ones(num_special_tokens, dtype=torch.int32, device=device),
                    torch.zeros(
                        self.patches_per_frame, dtype=torch.int32, device=device
                    ),
                )
            )
            expected_patch_positions = torch.arange(
                self.patches_per_frame, dtype=torch.int32, device=device
            )
            expected_special_pages = torch.arange(
                self.max_patch_pages,
                self.max_num_pages,
                dtype=torch.int32,
                device=device,
            )
            value_errors = []
            if not torch.equal(
                self._candidate021_direct_batch_indices_buf, expected_batch
            ):
                value_errors.append("batch order is not [special=1, patch=0]")
            if not torch.equal(
                self._candidate021_direct_positions_buf[num_special_tokens:],
                expected_patch_positions,
            ):
                value_errors.append("patch positions are not [0,P)")
            if not torch.equal(
                self._candidate021_direct_kv_indices_buf[1:],
                expected_special_pages,
            ):
                value_errors.append("special page IDs are not ascending")
            if value_errors:
                raise RuntimeError(
                    "Candidate 021C2 direct buffer initialization rejected: "
                    + "; ".join(value_errors)
                )
            self._candidate021_direct_contract = {
                "graph_only": True,
                "scale_eager_unchanged": True,
                "payload_order": "specials_then_patches",
                "logical_request_order": "patch_request_0_then_special_request_1",
                "special_page_policy": "ascending_from_max_patch_pages",
                "positions_alias_special_buffer": (
                    self._special_positions_buf.data_ptr()
                    == self._candidate021_direct_positions_buf.data_ptr()
                ),
                "batch_shape": list(
                    self._candidate021_direct_batch_indices_buf.shape
                ),
                "positions_shape": list(
                    self._candidate021_direct_positions_buf.shape
                ),
                "kv_indices_shape": list(
                    self._candidate021_direct_kv_indices_buf.shape
                ),
                "kv_indptr_shape": list(
                    self._candidate021_direct_kv_indptr_buf.shape
                ),
                "kv_last_page_len_shape": list(
                    self._candidate021_direct_kv_last_page_len_buf.shape
                ),
                "buffer_data_ptrs": {
                    "batch": self._candidate021_direct_batch_indices_buf.data_ptr(),
                    "positions": self._candidate021_direct_positions_buf.data_ptr(),
                    "kv_indices": self._candidate021_direct_kv_indices_buf.data_ptr(),
                    "kv_indptr": self._candidate021_direct_kv_indptr_buf.data_ptr(),
                    "kv_last_page_len": (
                        self._candidate021_direct_kv_last_page_len_buf.data_ptr()
                    ),
                },
                "cache_data_ptrs": [cache.data_ptr() for cache in self.kv_caches],
            }
            if not self._candidate021_direct_contract[
                "positions_alias_special_buffer"
            ]:
                raise RuntimeError(
                    "Candidate 021C2 special positions must alias direct positions"
                )
            self._candidate021_direct_capture_ready = True
            self.candidate021_metadata["direct_kv_append_contract"] = {
                key: value
                for key, value in self._candidate021_direct_contract.items()
                if key not in ("buffer_data_ptrs", "cache_data_ptrs")
            }

    def set_visible_window(self, visible_window: int) -> None:
        """Limit attention to recent window pages without changing eviction."""
        visible_window = int(visible_window)
        if not 1 <= visible_window <= self.sliding_window:
            raise ValueError(
                f"visible_window={visible_window} must be in "
                f"[1, {self.sliding_window}]"
            )
        self.visible_window = visible_window
        self.visible_plan_calls = 0
        self._kv_indices_buf_in_steady_layout = False

    def _validate_candidate021_direct_manager_buffers(self) -> None:
        """Validate stable storage without reading CUDA values or synchronizing."""
        if not _candidate021_direct_kv_append_enabled(self.candidate021_route):
            return
        errors = []
        contract = self._candidate021_direct_contract
        if not self._candidate021_direct_capture_ready or not isinstance(contract, dict):
            errors.append("capture-ready contract is missing")
        else:
            tensors = (
                (
                    "batch",
                    self._candidate021_direct_batch_indices_buf,
                    (self.tokens_per_frame,),
                ),
                (
                    "positions",
                    self._candidate021_direct_positions_buf,
                    (self.tokens_per_frame,),
                ),
                (
                    "kv_indices",
                    self._candidate021_direct_kv_indices_buf,
                    (1 + self.max_num_pages - self.max_patch_pages,),
                ),
                ("kv_indptr", self._candidate021_direct_kv_indptr_buf, (3,)),
                (
                    "kv_last_page_len",
                    self._candidate021_direct_kv_last_page_len_buf,
                    (2,),
                ),
            )
            for name, tensor, shape in tensors:
                if (
                    tensor is None
                    or tuple(tensor.shape) != shape
                    or tensor.dtype != torch.int32
                    or tensor.device != self.kv_caches[0].device
                    or not tensor.is_contiguous()
                ):
                    errors.append(f"{name} static tensor contract changed")
                elif tensor.data_ptr() != contract["buffer_data_ptrs"][name]:
                    errors.append(f"{name} storage address changed")
            if (
                self._special_positions_buf.data_ptr()
                != self._candidate021_direct_positions_buf.data_ptr()
            ):
                errors.append("special/direct positions alias changed")
            if [cache.data_ptr() for cache in self.kv_caches] != contract[
                "cache_data_ptrs"
            ]:
                errors.append("cache storage address changed")
        if errors:
            raise RuntimeError(
                "Candidate 021C2 manager buffer contract rejected: "
                + "; ".join(errors)
            )

    # =========================================================================
    # Public API  (every-frame append only)
    # =========================================================================

    def _validate_append_policy(self) -> None:
        if getattr(self, "_skip_append", False) or self._defer_eviction:
            raise RuntimeError(
                "Thor streaming requires every frame to append; dynamic keyframes are unsupported"
            )

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
        self._validate_append_policy()
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
        """Use upstream eviction only under the supported every-frame policy."""
        self._validate_append_policy()
        super().evict_frames(
            block_idx, scale_frames, sliding_window, cross_frame_special,
            include_scale_frames, camera_only, num_register_tokens,
        )

    def execute_deferred_eviction(
        self,
        block_idx: int,
        scale_frames: int,
        sliding_window: int,
        **kwargs,
    ) -> None:
        """Cyclic page writes cannot defer eviction of overwritten data."""
        raise RuntimeError("Deferred eviction is unsupported by the Thor cyclic KV cache")

    def rollback_last_frame(self, block_idx: int) -> None:
        """Reject rollback because cyclic writes may already replace live pages."""
        raise RuntimeError("Rollback is unsupported by the Thor cyclic KV cache")

    def _compute_fa4_attention(self, block_idx: int, q: Tensor) -> Tensor:
        """Run the public FA4 varlen call with the selected cache layout."""
        cache = self.kv_caches[block_idx]
        if self.candidate021_route is not None:
            expected_shape = (self.tokens_per_frame, self.num_heads, self.head_dim)
            contract_errors = []
            if tuple(q.shape) != expected_shape:
                contract_errors.append(f"q.shape={tuple(q.shape)}, expected {expected_shape}")
            if q.dtype != torch.bfloat16:
                contract_errors.append(f"q.dtype={q.dtype}, expected torch.bfloat16")
            if not q.is_contiguous():
                contract_errors.append(f"q.stride={tuple(q.stride())}, expected contiguous NHD")
            if q.device != cache.device:
                contract_errors.append(f"q.device={q.device}, cache.device={cache.device}")
            if cache.dtype != torch.bfloat16:
                contract_errors.append(
                    f"cache.dtype={cache.dtype}, expected torch.bfloat16"
                )
            if tuple(self._fa4_page_table_gpu.shape) != (1, self.max_num_pages):
                contract_errors.append(
                    f"page_table.shape={tuple(self._fa4_page_table_gpu.shape)}, "
                    f"expected {(1, self.max_num_pages)}"
                )
            if self._fa4_page_table_gpu.dtype != torch.int32:
                contract_errors.append(
                    f"page_table.dtype={self._fa4_page_table_gpu.dtype}, expected torch.int32"
                )
            if tuple(self._fa4_seqused_k_gpu.shape) != (1,):
                contract_errors.append(
                    f"seqused_k.shape={tuple(self._fa4_seqused_k_gpu.shape)}, expected (1,)"
                )
            if contract_errors:
                raise RuntimeError(
                    "Candidate 021 FA4 call rejected: "
                    + "; ".join(contract_errors)
                )

        q_arg = q.to(self.dtype).contiguous()
        out, _ = self._fa4_flash_attn_varlen_func(
            q_arg,
            cache[:, 0],
            cache[:, 1],
            cu_seqlens_q=self._fa4_cu_q_gpu,
            cu_seqlens_k=None,
            max_seqlen_q=self.tokens_per_frame,
            max_seqlen_k=None,
            seqused_k=self._fa4_seqused_k_gpu,
            page_table=self._fa4_page_table_gpu,
            causal=False,
        )
        return out

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

        if self.use_fa4:
            if block_idx == 0:
                visible = self.build_visible_page_table(0)
                last_len = self.compute_last_page_len(0)
                assert visible, "visible page table is empty after append_frame"
                n_visible = len(visible)
                kv_len = (n_visible - 1) * self.page_size + last_len
                self._fa4_page_table_gpu[0, :n_visible] = torch.tensor(
                    visible, dtype=torch.int32, device=self.device,
                )
                self._fa4_seqused_k_gpu[0] = kv_len

            return self._compute_fa4_attention(block_idx, q)

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

            # Update CPU-pinned plan() input buffers.  Writing a Python list straight
            # into a pinned CPU tensor stays on the host, so plan() sees its inputs
            # already on CPU and skips its three D→H syncs.  The wrapper's internal
            # GPU buffers (use_cuda_graph=True) get an async pinned H→D copy in plan().
            n_visible = len(visible)
            self._kv_indices_buf[:n_visible] = torch.tensor(visible, dtype=torch.int32)
            self._kv_indptr_buf[1] = n_visible
            self._kv_last_page_len_buf[0] = last_len
            # seq_lens = (n_visible - 1) * page_size + last_len, computed on host.
            self._seq_lens_buf[0] = (n_visible - 1) * self.page_size + last_len

            self.prefill_wrapper.plan(
                self._qo_indptr,
                self._kv_indptr_buf,
                self._kv_indices_buf[:n_visible],
                self._kv_last_page_len_buf,
                seq_lens          = self._seq_lens_buf,
                num_qo_heads      = self.num_heads,
                num_kv_heads      = self.num_heads,
                head_dim_qk       = self.head_dim,
                page_size         = self.page_size,
                causal            = False,          # custom page ordering; no causal mask
                pos_encoding_mode = "NONE",         # RoPE applied externally before append
                q_data_type       = self.dtype,
                non_blocking      = True,
            )

        # ── Run attention for this layer ──────────────────────────────────────
        # Cast q to storage dtype (LayerNorm may upcast to float32 under autocast).
        # Reuse a pre-allocated per-layer output buffer to skip the per-call
        # ``torch.zeros`` inside FlashInfer's run().
        # ``enable_pdl=True`` (Programmatic Dependent Launch, CUDA 12.5+) lets the
        # FA3 kernel start issuing reads before its predecessor's writes are
        # fully retired, hiding the kernel-to-kernel hand-off latency.
        return self.prefill_wrapper.run(
            q              = q.to(self.dtype).contiguous(),
            paged_kv_cache = self.kv_caches[block_idx],
            out            = self._attn_out_buffers[block_idx],
            enable_pdl     = True,
        )  # → [q_len, H, D]

    # =========================================================================
    # Graph-mode API  (used when ``_graph_mode`` is True, i.e. the per-frame
    # forward is being captured as one ``torch.cuda.graph``.  Drop-in for
    # ``append_frame`` + ``compute_attention`` from inside a captured region.)
    # =========================================================================

    def prepare_frame_for_graph(self, frame_idx: int) -> None:
        """Update host-side bookkeeping AND CUDA buffers for one streaming frame.

        Called OUTSIDE the captured graph each frame, before ``g.replay()``.
        Responsibilities (all paths):
          - cyclic patch-page ID for THIS frame → CUDA tensor.
          - special-token positions (n_s ints) → CUDA tensor (incremental add).
          - special-stream indptr / last_page_len for in-graph
            ``append_paged_kv_cache``.
          - host-pinned plan inputs (visible page table, indptr, last_page_len,
            seq_lens) and ``prefill_wrapper.plan(...)`` so the wrapper's
            internal CUDA buffers reflect THIS frame's visible page layout.

        Steady-state fast path: once ``frame_idx >= sf+sw`` and the
        ``_kv_indices_buf`` holds its pre-fill layout, the per-frame visible
        page table is naturally exposed by reading
        ``_kv_indices_buf[:sf+sw+n_sp_pages]`` — no per-frame patch-page-table
        rewrite, no ``torch.tensor(list, ...)`` allocation.
        ``_special_kv_indices_buf`` is similarly pre-baked once at init
        (sequential page IDs from ``max_patch_pages``) and never rewritten
        during streaming.
        """
        if not self._graph_mode:
            raise RuntimeError("prepare_frame_for_graph requires _graph_mode=True")
        self._validate_append_policy()
        self._validate_candidate021_direct_manager_buffers()

        sw  = self.sliding_window
        vw  = self.visible_window
        sf  = self.scale_frames
        n_s = self.num_special_tokens
        ps  = self.page_size

        if _candidate021_direct_kv_append_enabled(self.candidate021_route):
            route_errors = []
            if frame_idx < sf:
                route_errors.append(
                    f"frame_idx={frame_idx}, expected graph streaming frame >= {sf}"
                )
            if route_errors:
                raise RuntimeError(
                    "Candidate 021C2 graph prepare rejected: "
                    + "; ".join(route_errors)
                )

        # ── 1. Cyclic patch page ID for THIS frame ──────────────────────────
        if frame_idx < sf:
            patch_page_id = frame_idx                        # scale region
        else:
            patch_page_id = sf + ((frame_idx - sf) % sw)     # window cycle
        self._patch_write_page_id_buf[0] = patch_page_id     # async H→D
        if _candidate021_direct_kv_append_enabled(self.candidate021_route):
            self._candidate021_direct_kv_indices_buf[0] = patch_page_id

        # ── 2. Special positions: incremental write via in-place add ────────
        #    base = [0..n_s-1] (pre-baked); add scalar offset → buf.  Single
        #    tiny add kernel, no allocation, no per-frame `arange` call.
        special_total_before = frame_idx * n_s
        torch.add(
            self._special_positions_base,
            special_total_before,
            out=self._special_positions_buf,
        )

        # ── 3. Bookkeeping (block 0 only — graph mode is layer-lockstep) ────
        #    Other blocks' frame_count/all_special_pages aren't read by any
        #    graph-mode code path, so we skip the 24× loop.
        new_total = special_total_before + n_s
        needed_pages = (new_total + ps - 1) // ps
        if (
            _candidate021_direct_kv_append_enabled(self.candidate021_route)
            and needed_pages > self._special_kv_indices_buf.numel()
        ):
            raise RuntimeError(
                "Candidate 021C2 special page capacity exceeded: "
                f"needed_pages={needed_pages}, capacity="
                f"{self._special_kv_indices_buf.numel()}"
            )
        while len(self.all_special_pages[0]) < needed_pages:
            next_idx = len(self.all_special_pages[0])
            page_id  = self.max_patch_pages + next_idx
            self.all_special_pages[0].append(page_id)
        self.special_token_count[0] = new_total
        self.frame_count[0] = frame_idx + 1

        # ── 4. Special-only indptr / last_page_len ──────────────────────────
        #    _special_kv_indices_buf is pre-baked at init with sequential page
        #    IDs — kernel reads first n_sp_pages entries which always hold
        #    [max_pp, max_pp+1, ..., max_pp+n_sp_pages-1].
        n_sp_pages = needed_pages
        self._special_kv_indptr_buf[1] = n_sp_pages
        last_sp_len = new_total - (n_sp_pages - 1) * ps if n_sp_pages > 0 else 0
        if last_sp_len == 0 and n_sp_pages > 0:
            last_sp_len = ps
        self._special_kv_last_page_len_buf[0] = last_sp_len
        if _candidate021_direct_kv_append_enabled(self.candidate021_route):
            self._candidate021_direct_kv_indptr_buf[2] = 1 + n_sp_pages
            self._candidate021_direct_kv_last_page_len_buf[1] = last_sp_len

        # ── 5. Visible page table sizing ────────────────────────────────────
        n_active_scale = min(frame_idx + 1, sf)
        total_window_frames = max(0, frame_idx + 1 - sf)
        n_active_window = min(total_window_frames, sw, vw)
        n_visible = n_active_scale + n_active_window + n_sp_pages
        # last_page_len for the COMBINED page table follows the existing rule:
        #   - if specials exist, the last page is a special page → tail mod
        #   - else (only scale + maybe window): last is a patch page → P
        last_page_len = last_sp_len if n_sp_pages > 0 else self.patches_per_frame

        # ── 6. _kv_indices_buf: steady-state fast path / warmup slow path ───
        in_steady_state = (
            vw == sw
            and n_active_scale == sf
            and n_active_window == sw
        )
        if in_steady_state:
            if not self._kv_indices_buf_in_steady_layout:
                # First steady frame after warmup, eager phase, or reset() —
                # restore the pre-fill layout.  Done once per (re)entry.
                sf_sw = sf + sw
                self._kv_indices_buf[:sf_sw] = torch.arange(
                    0, sf_sw, dtype=torch.int32,
                )
                max_special_pages = self.max_num_pages - self.max_patch_pages
                self._kv_indices_buf[sf_sw:sf_sw + max_special_pages] = torch.arange(
                    self.max_patch_pages, self.max_num_pages, dtype=torch.int32,
                )
                self._kv_indices_buf_in_steady_layout = True
            # else: buffer already correct; no per-frame writes needed.
        else:
            # Preserve scale pages, select the newest visible window pages from
            # the physical cyclic pool, and keep all special pages.
            first_window_frame = total_window_frames - n_active_window
            window_pages = [
                sf + (window_frame % sw)
                for window_frame in range(
                    first_window_frame, total_window_frames
                )
            ]
            visible = list(range(n_active_scale)) + window_pages + [
                self.max_patch_pages + i for i in range(n_sp_pages)
            ]
            self._kv_indices_buf[:n_visible] = torch.tensor(visible, dtype=torch.int32)
            self._kv_indices_buf_in_steady_layout = False
            if vw < sw:
                self.visible_plan_calls += 1

        self._kv_indptr_buf[1] = n_visible
        self._kv_last_page_len_buf[0] = last_page_len
        self._seq_lens_buf[0] = (n_visible - 1) * ps + last_page_len
        if self.use_fa4:
            self._fa4_page_table_gpu[0, :n_visible].copy_(
                self._kv_indices_buf[:n_visible],
                non_blocking=True,
            )
            self._fa4_seqused_k_gpu[0] = self._seq_lens_buf[0]
            return

        # ── 7. plan() — populates wrapper's internal CUDA buffers in place ──
        self.prefill_wrapper.plan(
            self._qo_indptr,
            self._kv_indptr_buf,
            self._kv_indices_buf[:n_visible],
            self._kv_last_page_len_buf,
            seq_lens          = self._seq_lens_buf,
            num_qo_heads      = self.num_heads,
            num_kv_heads      = self.num_heads,
            head_dim_qk       = self.head_dim,
            page_size         = ps,
            causal            = False,
            pos_encoding_mode = "NONE",
            q_data_type       = self.dtype,
            non_blocking      = True,
        )

    def append_frame_graph(self, block_idx: int, k: Tensor, v: Tensor) -> None:
        """Tensor-driven, Python-state-stable variant of ``append_frame``.

        Safe to invoke from within a ``torch.cuda.graph`` capture: every
        write target is read from a CUDA tensor whose address is stable, with
        contents updated host-side by ``prepare_frame_for_graph``.

        Args:
            block_idx: layer index.
            k, v: [tokens_per_frame, H, D]  NHD layout.  Specials live at
                  positions [0, num_special_tokens); patches at the rest.
        """
        self._validate_append_policy()
        n = self.num_special_tokens
        P = self.patches_per_frame

        if _candidate021_direct_kv_append_enabled(self.candidate021_route):
            route_errors = []
            if torch.is_grad_enabled():
                route_errors.append("grad mode must be disabled before cache mutation")
            if not self._graph_mode:
                route_errors.append("_graph_mode must be true")
            if not self._candidate021_direct_capture_ready:
                route_errors.append("capture-ready contract is missing")
            if not isinstance(block_idx, int) or not (0 <= block_idx < self.num_blocks):
                route_errors.append(
                    f"block_idx={block_idx!r}, expected integer in [0,{self.num_blocks})"
                )
            if route_errors:
                raise RuntimeError(
                    "Candidate 021C2 graph append rejected: "
                    + "; ".join(route_errors)
                )
            candidate021_direct_append_paged_kv_cache(
                k,
                v,
                self.kv_caches[block_idx],
                self._candidate021_direct_batch_indices_buf,
                self._candidate021_direct_positions_buf,
                self._candidate021_direct_kv_indices_buf,
                self._candidate021_direct_kv_indptr_buf,
                self._candidate021_direct_kv_last_page_len_buf,
            )
            return

        # ── Patch K/V write: pad → index_copy_ at cyclic page ID ────────────
        patch_k = k[n:].to(self.dtype)
        patch_v = v[n:].to(self.dtype)
        # Slice writes are graph-compatible (Python-int slice baked into kernel).
        # The pad buffer is shared across layers within a frame; sequential writes
        # within the same graph are correctly ordered by tensor R/W dependencies.
        self._patch_write_pad_buf[0, 0, :P] = patch_k
        self._patch_write_pad_buf[0, 1, :P] = patch_v
        self.kv_caches[block_idx].index_copy_(
            0, self._patch_write_page_id_buf, self._patch_write_pad_buf,
        )

        # ── Special K/V write: append_paged_kv_cache (handles cross-page) ───
        sp_k = k[:n].to(self.dtype)
        sp_v = v[:n].to(self.dtype)
        flashinfer.page.append_paged_kv_cache(
            sp_k, sp_v,
            self._special_batch_indices_buf,
            self._special_positions_buf,
            self.kv_caches[block_idx],
            self._special_kv_indices_buf,
            self._special_kv_indptr_buf,
            self._special_kv_last_page_len_buf,
            kv_layout="NHD",
        )

    def compute_attention_graph(self, block_idx: int, q: Tensor) -> Tensor:
        """Graph-mode counterpart to ``compute_attention`` — ``run()`` only.

        ``plan()`` must have been called externally (``prepare_frame_for_graph``)
        before the captured graph is replayed.  Reads the wrapper's internal
        CUDA buffers and the page-K/V tensors; writes into the per-layer
        pre-allocated output buffer.
        """
        if self.use_fa4:
            return self._compute_fa4_attention(block_idx, q)

        return self.prefill_wrapper.run(
            q              = q.to(self.dtype).contiguous(),
            paged_kv_cache = self.kv_caches[block_idx],
            out            = self._attn_out_buffers[block_idx],
            enable_pdl     = True,
        )


    def reset(self) -> None:
        """Reset all per-block state for a new sequence."""
        for i in range(self.num_blocks):
            self.scale_patch_pages[i].clear()
            self.live_window_patch_pages[i].clear()
            self.all_special_pages[i].clear()
            self.free_patch_pages[i]   = list(range(self.max_patch_pages))
            self.free_special_pages[i] = list(
                range(self.max_num_pages - 1, self.max_patch_pages - 1, -1)
            )
            self.special_token_count[i] = 0
            self.frame_count[i] = 0
        # Eager `compute_attention` writes per-frame to `_kv_indices_buf`; the
        # graph-mode steady-state pre-fill must be re-established on the first
        # subsequent steady-state frame.
        self._kv_indices_buf_in_steady_layout = False

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
            list(self.live_window_patch_pages[block_idx])[-self.visible_window:] +
            list(self.all_special_pages[block_idx])
        )

    # ── Internal write helpers ────────────────────────────────────────────────

    def _write_patch_page(self, block_idx: int, patch_k: Tensor, patch_v: Tensor) -> int:
        """
        Write one frame's patch K/V to its **cyclic** page ID and update bookkeeping.

        Cyclic page-ID assignment (deterministic from frame_count, no free-list
        pops):
          - Scale  frame f (f < scale_frames):                         page f
          - Window frame f (f >= scale_frames):  scale_frames + ((f - scale_frames) % sliding_window)

        Cyclic IDs let the graph-capture path land K/V at predictable destinations
        without touching Python state, while keeping the eager (non-graph) path
        identical in behaviour: pages still get reused exactly when the window
        slides, just with a deterministic ID instead of a free-list pop.

        Returns:
            page_id: Physical page index written this call.
        """
        f  = self.frame_count[block_idx]                 # 0-indexed; pre-increment
        sw = self.sliding_window
        sf = self.scale_frames

        if f < sf:
            page_id = f
        else:
            page_id = sf + ((f - sf) % sw)

        # Direct slice write: positions 0..patches_per_frame-1.
        # When page_size > patches_per_frame (FA3 power-of-2 padding), positions
        # patches_per_frame..page_size-1 stay at their previous content.  For a
        # cyclic ID that gets reused, the trailing slot still holds the OLD K/V's
        # padding region — which was zero on first use (kv_caches is zero-init)
        # and stays zero across cycles because we only ever write [:P].
        P = self.patches_per_frame
        self.kv_caches[block_idx][page_id, 0, :P] = patch_k  # K
        self.kv_caches[block_idx][page_id, 1, :P] = patch_v  # V

        # Bookkeeping deques retain the same semantics as before: scale grows to
        # `scale_frames`, window grows to `sliding_window` then cycles by
        # popleft-on-append.  Cyclic IDs mean the new page_id may equal a popped
        # one — the deque just rotates the visible-page-table order.
        if f < sf:
            self.scale_patch_pages[block_idx].append(page_id)
        else:
            if len(self.live_window_patch_pages[block_idx]) >= sw:
                # Steady state: drop the oldest before appending the new one so
                # the deque stays at length `sw`.  No free-list bookkeeping needed.
                self.live_window_patch_pages[block_idx].popleft()
            self.live_window_patch_pages[block_idx].append(page_id)

        return page_id
