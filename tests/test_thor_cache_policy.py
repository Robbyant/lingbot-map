"""CPU regressions for the Thor cache's every-frame append policy."""
import collections
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from lingbot_map.optimizations.thor import blocks
from lingbot_map.optimizations.thor.cache import FlashInferKVCacheManager


class ThorCachePolicyTest(unittest.TestCase):
    def make_cache(self, frames=3):
        # Exercise real bookkeeping with small CPU pages, without a GPU wrapper.
        manager = FlashInferKVCacheManager.__new__(FlashInferKVCacheManager)
        manager.num_special_tokens = 6
        manager.patches_per_frame = manager.page_size = 7
        manager.tokens_per_frame = 13
        manager.scale_frames = 1
        manager.sliding_window = manager.visible_window = 2
        manager.dtype = torch.bfloat16
        manager.frame_count = [0]
        manager.special_token_count = [0]
        manager.scale_patch_pages = [collections.deque()]
        manager.live_window_patch_pages = [collections.deque()]
        manager.free_patch_pages = [list(range(4))]
        manager.all_special_pages = [[]]
        manager.free_special_pages = [list(range(9, 3, -1))]
        manager.kv_caches = [torch.zeros(10, 2, 7, 1, 2, dtype=manager.dtype)]
        manager._skip_append = manager._defer_eviction = manager._graph_mode = False
        manager.housekeeping_mode = None
        for frame in range(frames):
            k, v = self.frame_tensors(frame)
            manager.append_frame(0, k, v)
        return manager

    def frame_tensors(self, frame):
        k = torch.full((13, 1, 2), frame + 1, dtype=torch.bfloat16)
        return k, -k

    def snapshot(self, manager):
        names = (
            "frame_count", "special_token_count", "scale_patch_pages",
            "live_window_patch_pages", "free_patch_pages", "all_special_pages",
            "free_special_pages", "_skip_append", "_defer_eviction",
        )
        return (
            manager.kv_caches[0].clone(),
            {name: copy.deepcopy(getattr(manager, name)) for name in names},
        )

    def assert_unchanged(self, manager, snapshot):
        payload, state = snapshot
        self.assertTrue(torch.equal(manager.kv_caches[0], payload))
        for name, expected in state.items():
            self.assertEqual(getattr(manager, name), expected, name)

    def test_eager_rejects_dynamic_keyframes_without_mutation(self):
        for frames in (0, 1, 3, 5):
            for flag in ("_skip_append", "_defer_eviction"):
                with self.subTest(frames=frames, flag=flag):
                    manager = self.make_cache(frames)
                    setattr(manager, flag, True)
                    before = self.snapshot(manager)
                    k, v = self.frame_tensors(frames)
                    with self.assertRaisesRegex(RuntimeError, "dynamic keyframes are unsupported"):
                        manager.append_frame(0, k, v)
                    self.assert_unchanged(manager, before)
                    setattr(manager, flag, False)
                    manager.append_frame(0, k, v)
                    self.assertEqual(manager.frame_count, [frames + 1])

    def test_rollback_and_deferred_eviction_reject_without_mutation(self):
        for frames in (0, 1, 3, 5):
            for graph_mode in (False, True):
                for defer_eviction in (False, True):
                    for operation in ("rollback_last_frame", "execute_deferred_eviction"):
                        with self.subTest(frames=frames, graph=graph_mode,
                                          deferred=defer_eviction, operation=operation):
                            manager = self.make_cache(frames)
                            manager._graph_mode = graph_mode
                            manager._defer_eviction = defer_eviction
                            before = self.snapshot(manager)
                            args = (0,) if operation == "rollback_last_frame" else (0, 1, 2)
                            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                                getattr(manager, operation)(*args)
                            self.assert_unchanged(manager, before)

    def test_evict_rejects_dynamic_keyframes_without_mutation(self):
        for flag in ("_skip_append", "_defer_eviction"):
            with self.subTest(flag=flag):
                manager = self.make_cache(5)
                setattr(manager, flag, True)
                before = self.snapshot(manager)
                with self.assertRaisesRegex(RuntimeError, "dynamic keyframes are unsupported"):
                    manager.evict_frames(0, 1, 2)
                self.assert_unchanged(manager, before)

    def test_graph_prepare_rejects_before_buffer_updates(self):
        for route in (None, "direct_kv_append"):
            for flag in ("_skip_append", "_defer_eviction"):
                with self.subTest(route=route, flag=flag):
                    manager = self.make_cache()
                    manager._graph_mode = True
                    manager.housekeeping_mode = route
                    manager._validate_direct_append_buffers = Mock()
                    manager._patch_write_page_id_buf = torch.tensor([-1])
                    setattr(manager, flag, True)
                    before = self.snapshot(manager)
                    with self.assertRaisesRegex(RuntimeError, "dynamic keyframes are unsupported"):
                        manager.prepare_frame_for_graph(3)
                    self.assert_unchanged(manager, before)
                    self.assertEqual(manager._patch_write_page_id_buf.tolist(), [-1])
                    manager._validate_direct_append_buffers.assert_not_called()

    def test_graph_append_rejects_before_cache_writes(self):
        for route in (None, "direct_kv_append"):
            for flag in ("_skip_append", "_defer_eviction"):
                with self.subTest(route=route, flag=flag):
                    manager = self.make_cache()
                    manager._graph_mode = True
                    manager.housekeeping_mode = route
                    setattr(manager, flag, True)
                    before = self.snapshot(manager)
                    k, v = self.frame_tensors(3)
                    with self.assertRaisesRegex(RuntimeError, "dynamic keyframes are unsupported"):
                        manager.append_frame_graph(0, k, v)
                    self.assert_unchanged(manager, before)

    def test_block_rejects_before_qkv_or_cache_work(self):
        for graph_mode in (False, True):
            for flag in ("_skip_append", "_defer_eviction"):
                with self.subTest(graph_mode=graph_mode, flag=flag):
                    manager = self.make_cache()
                    manager._graph_mode = graph_mode
                    setattr(manager, flag, True)
                    before = self.snapshot(manager)
                    block = SimpleNamespace(attn_pre=Mock(), attn_post_ffn=Mock())
                    with self.assertRaisesRegex(RuntimeError, "dynamic keyframes are unsupported"):
                        blocks.forward(block, None, kv_cache=manager)
                    block.attn_pre.assert_not_called()
                    block.attn_post_ffn.assert_not_called()
                    self.assert_unchanged(manager, before)

    def test_regular_append_preserves_scale_window_and_special_payloads(self):
        manager = self.make_cache(0)
        for frame in range(7):
            k, v = self.frame_tensors(frame)
            manager.append_frame(0, k, v)
            window_frames = range(max(1, frame - 1), frame + 1)
            self.assertEqual(manager.frame_count, [frame + 1])
            self.assertEqual(list(manager.scale_patch_pages[0]), [0])
            self.assertEqual(list(manager.live_window_patch_pages[0]), [
                1 + (index - 1) % 2 for index in window_frames
            ])
            for index in (0, *window_frames):
                page_id = 0 if index == 0 else 1 + (index - 1) % 2
                expected_k, expected_v = self.frame_tensors(index)
                self.assertTrue(torch.equal(manager.kv_caches[0][page_id, 0], expected_k[6:]))
                self.assertTrue(torch.equal(manager.kv_caches[0][page_id, 1], expected_v[6:]))
            count = (frame + 1) * 6
            self.assertEqual(manager.special_token_count, [count])
            for position in range(count):
                page_id = manager.all_special_pages[0][position // 7]
                value = position // 6 + 1
                self.assertTrue(torch.all(manager.kv_caches[0][page_id, 0, position % 7] == value))
                self.assertTrue(torch.all(manager.kv_caches[0][page_id, 1, position % 7] == -value))

    def test_supported_block_paths_keep_operation_order(self):
        for graph_mode in (False, True):
            with self.subTest(graph_mode=graph_mode):
                calls = []
                q, k, v, x, out, result = (object() for _ in range(6))
                manager = SimpleNamespace(
                    _graph_mode=graph_mode, _skip_append=False, _defer_eviction=False,
                    append_frame=Mock(side_effect=lambda *a: calls.append("append")),
                    evict_frames=Mock(side_effect=lambda **kw: calls.append("evict")),
                    compute_attention=Mock(side_effect=lambda *a: calls.append("attention") or out),
                    append_frame_graph=Mock(side_effect=lambda *a: calls.append("append_graph")),
                    compute_attention_graph=Mock(side_effect=lambda *a: calls.append("attention_graph") or out),
                )
                block = SimpleNamespace(
                    attn_pre=Mock(side_effect=lambda *a, **kw: calls.append("qkv") or (q, k, v)),
                    attn_post_ffn=Mock(side_effect=lambda *a: calls.append("post") or result),
                    attn=SimpleNamespace(
                        kv_cache_scale_frames=1, kv_cache_sliding_window=2,
                        kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                        kv_cache_camera_only=False,
                    ),
                )
                self.assertIs(blocks.forward(block, x, kv_cache=manager, global_idx=4), result)
                expected = ["append_graph", "attention_graph"] if graph_mode else ["append", "evict", "attention"]
                self.assertEqual(calls, ["qkv", *expected, "post"])
                append = manager.append_frame_graph if graph_mode else manager.append_frame
                append.assert_called_once_with(4, k, v)
                block.attn_post_ffn.assert_called_once_with(x, out)


if __name__ == "__main__":
    unittest.main()
