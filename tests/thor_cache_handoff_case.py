"""Isolated GPU cases launched by test_thor_gpu; no weights or timing needed."""
import copy
import os
import types
import unittest

import torch


class CacheHandoffCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lingbot_map.optimizations.thor.options import ThorOptions

        cls.options = ThorOptions.from_env()
        cls.options.activate()
        from lingbot_map.optimizations.thor.cache import FlashInferKVCacheManager

        tokens = int(os.environ["THOR_TEST_TOKENS_PER_FRAME"])
        cls.manager = FlashInferKVCacheManager(
            num_blocks=24, max_num_frames=88, tokens_per_frame=tokens,
            num_heads=16, head_dim=64, dtype=torch.bfloat16,
            device=torch.device("cuda"), scale_frames=8, sliding_window=64,
            max_total_frames=180, backend="fa4",
        )

    def setUp(self):
        self.manager._graph_mode = False
        self.manager.reset()

    def test_fa4_omits_unused_flashinfer_buffers(self):
        self.assertIsNone(self.manager.workspace_buffer)
        self.assertEqual(self.manager._attn_out_buffers, [])

    @torch.no_grad()
    def test_scale_attention_writes_every_cache_block(self):
        from lingbot_map.layers.attention import FlashInferAttention

        manager = self.manager
        attention = FlashInferAttention(dim=1024, num_heads=16).eval().cuda()
        generator = torch.Generator(device="cuda").manual_seed(12)
        x = torch.randn(
            1, 8 * manager.tokens_per_frame, 1024,
            device="cuda", generator=generator,
        )
        # Independent tensor oracle for identity q/k norms and no RoPE in this case.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            qkv = attention.qkv(x).reshape(8, manager.tokens_per_frame, 3, 16, 64)
            k, v = qkv[:, :, 1].contiguous(), qkv[:, :, 2].contiguous()
            for block in range(manager.num_blocks):
                attention(x, num_frames=8, kv_cache=manager, global_idx=block)
                self.assertEqual(manager.frame_count[block], 8)
                self.assertEqual(manager.special_token_count[block], 48)
                self.assertEqual(list(manager.scale_patch_pages[block]), list(range(8)))
                cache = manager.kv_caches[block]
                self.assertTrue(torch.equal(cache[:8, 0], k[:, 6:]))
                self.assertTrue(torch.equal(cache[:8, 1], v[:, 6:]))
                special_page = manager.all_special_pages[block][0]
                self.assertTrue(torch.equal(
                    cache[special_page, 0, :48], k[:, :6].reshape(48, 16, 64),
                ))
                self.assertTrue(torch.equal(
                    cache[special_page, 1, :48], v[:, :6].reshape(48, 16, 64),
                ))

    @torch.no_grad()
    def test_captured_history_wrap_and_reset(self):
        from lingbot_map.optimizations.thor.projection import validate_housekeeping_runtime

        manager = self.manager
        model = types.SimpleNamespace(
            aggregator=types.SimpleNamespace(kv_cache_manager=manager),
        )
        validate_housekeeping_runtime(model)
        page, special = manager.page_size, manager.num_special_tokens
        # Check the first and last layer, including every token/head/component.
        checked_blocks = (0, manager.num_blocks - 1)
        generator = torch.Generator(device="cuda").manual_seed(13)
        shape = (manager.num_blocks, manager.tokens_per_frame, 16, 64)
        base_k = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        base_v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        k, v = torch.empty_like(base_k), torch.empty_like(base_v)
        expected = {
            block: torch.empty_like(manager.kv_caches[block]) for block in checked_blocks
        }
        crossed_page_frame = page // special
        last_frame = max(crossed_page_frame + 2, 137)
        checkpoints = {
            8, 71, 72, 135, 136, crossed_page_frame, crossed_page_frame + 1, last_frame,
        }

        def update_inputs(frame, sequence):
            torch.add(base_k, sequence + frame / 128, out=k)
            torch.sub(base_v, sequence + frame / 128, out=v)

        def update_expected(frame):
            patch_id = frame if frame < 8 else 8 + (frame - 8) % 64
            for block, cache in expected.items():
                cache[patch_id, 0].copy_(k[block, special:])
                cache[patch_id, 1].copy_(v[block, special:])
                for slot in range(special):
                    position = frame * special + slot
                    page_id = manager.max_patch_pages + position // page
                    cache[page_id, 0, position % page].copy_(k[block, slot])
                    cache[page_id, 1, position % page].copy_(v[block, slot])

        def append_all():
            for block in range(manager.num_blocks):
                manager.append_frame_graph(block, k[block], v[block])

        graph = None
        for sequence in range(2):
            with self.subTest(sequence=sequence):
                manager._graph_mode = False
                manager.reset()
                # Preserve stale bytes across reset; newly visible data must replace them.
                for block, cache in expected.items():
                    cache.copy_(manager.kv_caches[block])
                for frame in range(8):
                    update_inputs(frame, sequence)
                    for block in range(manager.num_blocks):
                        manager.append_frame(block, k[block], v[block])
                    update_expected(frame)

                manager._graph_mode = True
                if graph is None:
                    update_inputs(8, sequence)
                    manager.prepare_frame_for_graph(8)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        append_all()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        append_all()
                for frame in range(8, last_frame + 1):
                    update_inputs(frame, sequence)
                    manager.prepare_frame_for_graph(frame)
                    graph.replay()
                    update_expected(frame)
                    if frame in checkpoints:
                        count = int(manager._kv_indptr_buf[1])
                        visible = manager._fa4_page_table_gpu[0, :count].tolist()
                        n_patch = 8 + min(frame + 1 - 8, 64)
                        n_special_pages = ((frame + 1) * special + page - 1) // page
                        self.assertEqual(visible[n_patch:], list(range(
                            manager.max_patch_pages, manager.max_patch_pages + n_special_pages,
                        )))
                        for block, cache in expected.items():
                            self.assertTrue(
                                torch.equal(manager.kv_caches[block], cache),
                                f"KV payload mismatch: sequence={sequence}, "
                                f"frame={frame}, block={block}",
                            )
        torch.cuda.synchronize()

    @torch.no_grad()
    def test_rollback_and_deferred_eviction_reject_without_mutation(self):
        manager = self.manager
        block = manager.num_blocks - 1
        count = manager.page_size // manager.num_special_tokens + 1
        k = torch.ones(manager.tokens_per_frame, 16, 64, dtype=torch.bfloat16, device="cuda")
        v = -k
        for _ in range(count):
            manager.append_frame(block, k, v)
        before = manager.kv_caches[block].clone()
        state = {
            name: copy.deepcopy(getattr(manager, name))
            for name in ("frame_count", "special_token_count", "scale_patch_pages",
                         "live_window_patch_pages", "free_patch_pages",
                         "all_special_pages", "free_special_pages")
        }
        for operation, args in ((manager.rollback_last_frame, (block,)),
                                (manager.execute_deferred_eviction, (block, 8, 64))):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(RuntimeError, "unsupported"):
                    operation(*args)
                self.assertTrue(torch.equal(manager.kv_caches[block], before))
                for name, expected in state.items():
                    self.assertEqual(getattr(manager, name), expected, name)


if __name__ == "__main__":
    unittest.main()
