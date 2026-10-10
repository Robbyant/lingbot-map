"""Short correctness tests; no benchmark or profiler calls."""
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock

import torch


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() == (11, 0), "SM110 required")
class ThorGpuTest(unittest.TestCase):
    def _run_captured_cache_case(self, merge_append):
        from lingbot_map.optimizations.thor.options import FLAGS, VISIBLE_ENV

        # Backend selection is process-wide; do not reuse imports across routes.
        for tokens in (783, 978, 1005):
            with self.subTest(tokens_per_frame=tokens):
                env = {
                    name: value for name, value in os.environ.items()
                    if not name.startswith("LINGBOT_CANDIDATE")
                    and name not in (*FLAGS.values(), VISIBLE_ENV)
                }
                env[FLAGS["merge_kv_append"]] = str(int(merge_append))
                env["THOR_TEST_TOKENS_PER_FRAME"] = str(tokens)
                result = subprocess.run(
                    [sys.executable, "-m", "unittest", "discover", "-s", "tests",
                     "-p", "thor_cache_handoff_case.py", "-v"],
                    cwd=Path(__file__).resolve().parents[1], env=env,
                    capture_output=True, text=True, timeout=180,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_captured_split_append_preserves_history(self):
        self._run_captured_cache_case(merge_append=False)

    def test_captured_combined_append_preserves_history(self):
        self._run_captured_cache_case(merge_append=True)

    @torch.no_grad()
    def test_combined_append_rejects_unaudited_shapes_and_layouts(self):
        from lingbot_map.optimizations.thor.kv_append import (
            validate_direct_append_call,
        )

        def arguments(tokens):
            return [
                torch.empty(tokens, 16, 64, device="cuda", dtype=torch.bfloat16),
                torch.empty(tokens, 16, 64, device="cuda", dtype=torch.bfloat16),
                torch.empty(3, 2, tokens - 6, 16, 64, device="cuda", dtype=torch.bfloat16),
                torch.empty(tokens, device="cuda", dtype=torch.int32),
                torch.empty(tokens, device="cuda", dtype=torch.int32),
                torch.empty(3, device="cuda", dtype=torch.int32),
                torch.empty(3, device="cuda", dtype=torch.int32),
                torch.empty(2, device="cuda", dtype=torch.int32),
            ]

        for tokens in (1004, 1006):
            with self.subTest(tokens=tokens):
                with self.assertRaisesRegex(RuntimeError, "tokens_per_frame"):
                    validate_direct_append_call(*arguments(tokens))
        args = arguments(1005)
        validate_direct_append_call(*args)
        for invalid in (
            torch.empty(1005, 16, 64, device="cuda", dtype=torch.float32),
            torch.empty(1005, 16, 128, device="cuda", dtype=torch.bfloat16)[..., ::2],
        ):
            with self.subTest(dtype=invalid.dtype, stride=invalid.stride()):
                with self.assertRaisesRegex(RuntimeError, r"k\.(dtype|stride)"):
                    validate_direct_append_call(invalid, *args[1:])

    @torch.no_grad()
    def test_public_fa4_dispatch_preserves_tensor_contract(self):
        from lingbot_map.optimizations.thor.cache import FlashInferKVCacheManager
        q = torch.randn(783, 16, 64, device="cuda", dtype=torch.bfloat16)
        cache = torch.empty(2, 2, 777, 16, 64, device="cuda", dtype=torch.bfloat16)
        result = torch.empty_like(q)
        for route in (None, "projection_shadow", "direct_kv_append",
                      "projection_shadow_direct_kv_append"):
            call = Mock(return_value=(result, None))
            manager = types.SimpleNamespace(
                housekeeping_mode=route, tokens_per_frame=783, num_heads=16,
                head_dim=64, dtype=torch.bfloat16, max_num_pages=2,
                kv_caches=[cache], _fa4_flash_attn_varlen_func=call,
                _fa4_cu_q_gpu=torch.tensor([0, 783], dtype=torch.int32, device="cuda"),
                _fa4_page_table_gpu=torch.tensor([[0, 1]], dtype=torch.int32, device="cuda"),
                _fa4_seqused_k_gpu=torch.tensor([1554], dtype=torch.int32, device="cuda"),
            )
            self.assertIs(FlashInferKVCacheManager._compute_fa4_attention(manager, 0, q), result)
            call.assert_called_once()
            args, kwargs = call.call_args
            self.assertEqual(args[0].data_ptr(), q.data_ptr())
            for index in (0, 1):
                self.assertEqual(args[index + 1].data_ptr(), cache[:, index].data_ptr())
                self.assertEqual(args[index + 1].stride(), cache[:, index].stride())
            self.assertIs(kwargs["cu_seqlens_q"], manager._fa4_cu_q_gpu)
            self.assertIs(kwargs["seqused_k"], manager._fa4_seqused_k_gpu)
            self.assertIs(kwargs["page_table"], manager._fa4_page_table_gpu)
            self.assertEqual(kwargs["max_seqlen_q"], 783)
            self.assertIsNone(kwargs["max_seqlen_k"])
            self.assertFalse(kwargs["causal"])

    @torch.no_grad()
    def test_visible_window_keeps_scale_and_special_pages(self):
        from lingbot_map.optimizations.thor.cache import FlashInferKVCacheManager
        manager = FlashInferKVCacheManager(
            num_blocks=1, max_num_frames=88, tokens_per_frame=978,
            num_heads=16, head_dim=64, dtype=torch.bfloat16,
            device=torch.device("cuda"), sliding_window=64,
            scale_frames=8, max_total_frames=160, backend="fa4",
        )
        manager.set_visible_window(56)
        manager._graph_mode = True
        for frame in (8, 65, 71, 72, 135, 143):
            manager.prepare_frame_for_graph(frame)
            count = int(manager._kv_indptr_buf[1])
            visible = manager._kv_indices_buf[:count].tolist()
            window_count = min(frame + 1 - 8, 56)
            window_end = frame + 1 - 8
            expected_window = [8 + i % 64 for i in range(window_end - window_count, window_end)]
            self.assertEqual(visible[:8], list(range(8)))
            self.assertEqual(visible[8:8 + window_count], expected_window)
            special_count = ((frame + 1) * 6 + 971) // 972
            self.assertEqual(visible[8 + window_count:], list(range(manager.max_patch_pages, manager.max_patch_pages + special_count)))
            self.assertEqual(manager.sliding_window, 64)

    @torch.no_grad()
    def test_eager_to_graph_handoff_preserves_special_page_payload(self):
        """Regression: graph-mode reads must retain the eager scale K/V payload."""
        from lingbot_map.optimizations.thor.cache import FlashInferKVCacheManager

        manager = FlashInferKVCacheManager(
            num_blocks=1,
            max_num_frames=88,
            tokens_per_frame=978,
            num_heads=16,
            head_dim=64,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
            sliding_window=64,
            scale_frames=8,
            max_total_frames=160,
            backend="fa4",
        )

        special = manager.num_special_tokens
        expected = torch.empty(
            (2, 8 * special, 16, 64), dtype=torch.bfloat16, device="cuda",
        )
        for frame in range(8):
            k = torch.full(
                (978, 16, 64), frame + 1,
                dtype=torch.bfloat16, device="cuda",
            )
            v = torch.full(
                (978, 16, 64), frame + 101,
                dtype=torch.bfloat16, device="cuda",
            )
            manager.append_frame(0, k, v)
            expected[0, frame * special:(frame + 1) * special].fill_(frame + 1)
            expected[1, frame * special:(frame + 1) * special].fill_(frame + 101)

        eager_special_page = manager.all_special_pages[0][0]
        history_tokens = expected.shape[1]
        self.assertEqual(manager.special_token_count[0], history_tokens)
        self.assertTrue(torch.equal(
            manager.kv_caches[0][eager_special_page, :, :history_tokens], expected,
        ), "eager setup did not store the known scale K/V payload")

        manager._graph_mode = True
        manager.prepare_frame_for_graph(8)
        # Eight scale patch pages and one current patch page precede specials.
        graph_special_page = int(manager._fa4_page_table_gpu[0, 9].item())
        k = torch.full(
            (978, 16, 64), 9,
            dtype=torch.bfloat16, device="cuda",
        )
        v = torch.full(
            (978, 16, 64), 109,
            dtype=torch.bfloat16, device="cuda",
        )
        manager.append_frame_graph(0, k, v)

        current = manager.kv_caches[0][
            graph_special_page, :, history_tokens:history_tokens + special,
        ]
        self.assertTrue(torch.equal(current[0], k[:special]))
        self.assertTrue(torch.equal(current[1], v[:special]))
        actual = manager.kv_caches[0][graph_special_page, :, :history_tokens]
        self.assertTrue(
            torch.equal(expected, actual),
            "scale special-token payload was lost at graph handoff "
            f"(eager page {eager_special_page}, graph read page {graph_special_page})",
        )

    @torch.no_grad()
    def test_cached_mlp_matches_bf16_autocast(self):
        from lingbot_map.layers.block import FlashInferBlock
        from lingbot_map.optimizations.thor import blocks, weight_cache
        block = FlashInferBlock(1024, 16, init_values=1e-5).eval().cuda()
        block.attn_post = types.MethodType(blocks.attn_post, block)
        x = torch.randn(1, 783, 1024, device="cuda", dtype=torch.bfloat16)
        out = torch.randn(783, 16, 64, device="cuda", dtype=torch.bfloat16)
        state_before = {k: v.clone() for k, v in block.state_dict().items()}
        weight_cache._refresh_mlp_shadow_buffers(block.mlp)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            expected = blocks.attn_post_ffn(block, x, out)
            actual = weight_cache._global_mlp_cached_forward(block, x, out)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(set(state_before), set(block.state_dict()))
        for name, tensor in block.state_dict().items():
            self.assertTrue(torch.equal(state_before[name], tensor))

    @torch.no_grad()
    def test_combined_append_matches_payload_across_wraps(self):
        from lingbot_map.optimizations.thor.kv_append import (
            append_paged_kv_cache, validate_direct_append_runtime,
        )
        validate_direct_append_runtime()
        page, special, window, scale = 777, 6, 2, 8
        cache = torch.zeros(15, 2, page, 16, 64, dtype=torch.bfloat16, device="cuda")
        expected = torch.zeros_like(cache)
        batch = torch.zeros(page + special, dtype=torch.int32, device="cuda")
        batch[:special] = 1
        positions = torch.empty(page + special, dtype=torch.int32, device="cuda")
        positions[special:] = torch.arange(page, device="cuda", dtype=torch.int32)
        indices = torch.tensor([0, 10, 11], device="cuda", dtype=torch.int32)
        indptr = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
        last = torch.tensor([page, special], device="cuda", dtype=torch.int32)
        for frame in range(140):
            patch_id = frame if frame < scale else scale + (frame - scale) % window
            indices[0] = patch_id
            positions[:special] = torch.arange(special, device="cuda", dtype=torch.int32) + frame * special
            special_total = (frame + 1) * special
            indptr[2] = 1 + (special_total + page - 1) // page
            last[1] = special_total % page or page
            k = torch.randn(page + special, 16, 64, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            append_paged_kv_cache(k, v, cache, batch, positions, indices, indptr, last)
            expected[patch_id, 0].copy_(k[special:])
            expected[patch_id, 1].copy_(v[special:])
            for slot in range(special):
                position = frame * special + slot
                expected[10 + position // page, 0, position % page].copy_(k[slot])
                expected[10 + position // page, 1, position % page].copy_(v[slot])
        self.assertTrue(torch.equal(expected, cache))


if __name__ == "__main__":
    unittest.main()
