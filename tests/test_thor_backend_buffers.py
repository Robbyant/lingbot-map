"""Constructor allocation contracts, without allocating GPU storage."""
from contextlib import ExitStack
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import torch

from lingbot_map.optimizations.thor import cache


class ThorBackendBuffersTest(unittest.TestCase):
    def make_cache(self, backend):
        # Meta tensors retain shape/dtype while avoiding the 128 MiB workspace.
        def meta_factory(factory):
            def allocate(*args, **kwargs):
                kwargs.pop("pin_memory", None)
                kwargs["device"] = "meta"
                return factory(*args, **kwargs)
            return allocate

        cute = ModuleType("flash_attn.cute")
        cute.flash_attn_varlen_func = Mock()
        flashinfer = Mock()
        with ExitStack() as stack:
            stack.enter_context(patch.object(cache, "FLASHINFER_AVAILABLE", True))
            stack.enter_context(patch.object(cache, "flashinfer", flashinfer, create=True))
            stack.enter_context(patch.object(cache, "_resolve_housekeeping_route", return_value=None))
            stack.enter_context(patch.object(cache, "fa4_overlay_enabled", return_value=False))
            stack.enter_context(patch.object(cache, "prepare_fa4_overlay", return_value=None))
            stack.enter_context(patch.dict(sys.modules, {"flash_attn.cute": cute}))
            for name in ("empty", "zeros", "tensor", "arange"):
                factory = getattr(torch, name)
                stack.enter_context(patch.object(torch, name, meta_factory(factory)))
            manager = cache.FlashInferKVCacheManager(
                num_blocks=2, max_num_frames=3, tokens_per_frame=13,
                num_heads=1, head_dim=2, dtype=torch.bfloat16,
                device=torch.device("meta"), scale_frames=1, sliding_window=2,
                max_total_frames=7, backend=backend,
            )
        return manager, flashinfer

    def test_fa4_does_not_allocate_unused_wrapper_buffers(self):
        manager, flashinfer = self.make_cache("fa4")
        self.assertIsNone(manager.workspace_buffer)
        self.assertEqual(manager._attn_out_buffers, [])
        for name in ("_qo_indptr_buf_gpu", "_kv_indptr_buf_gpu",
                     "_kv_indices_buf_gpu", "_kv_last_page_len_buf_gpu"):
            self.assertIsNone(getattr(manager, name), name)
        flashinfer.BatchPrefillWithPagedKVCacheWrapper.assert_not_called()
        self.assertEqual(tuple(manager._fa4_page_table_gpu.shape), (1, manager.max_num_pages))
        self.assertEqual(manager._fa4_cu_q_gpu.dtype, torch.int32)
        self.assertIsNotNone(manager._fa4_flash_attn_varlen_func)

    def test_fa2_keeps_workspace_metadata_and_output_buffers(self):
        manager, flashinfer = self.make_cache("fa2")
        self.assertEqual(manager.workspace_buffer.numel(), 128 * 1024 * 1024)
        self.assertEqual(manager.workspace_buffer.dtype, torch.uint8)
        self.assertEqual(len(manager._attn_out_buffers), manager.num_blocks)
        for output in manager._attn_out_buffers:
            self.assertEqual(tuple(output.shape), (13, 1, 2))
            self.assertEqual(output.dtype, torch.bfloat16)
        flashinfer.BatchPrefillWithPagedKVCacheWrapper.assert_called_once_with(
            manager.workspace_buffer, kv_layout="NHD", backend="fa2", use_cuda_graph=True,
            qo_indptr_buf=manager._qo_indptr_buf_gpu,
            paged_kv_indptr_buf=manager._kv_indptr_buf_gpu,
            paged_kv_indices_buf=manager._kv_indices_buf_gpu,
            paged_kv_last_page_len_buf=manager._kv_last_page_len_buf_gpu,
        )


if __name__ == "__main__":
    unittest.main()
