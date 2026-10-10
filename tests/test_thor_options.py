import importlib.util
import itertools
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from lingbot_map.optimizations.thor.options import FLAGS, ThorOptions, housekeeping_route


class ThorOptionsTest(unittest.TestCase):
    def test_import_is_inert(self):
        code = "import os,sys; before=dict(os.environ); from lingbot_map.optimizations.thor import ThorOptions; assert before==dict(os.environ); assert 'torch' not in sys.modules; assert 'flash_attn' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_defaults_and_invalid_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ThorOptions.from_env(), ThorOptions())
            self.assertFalse(any(ThorOptions().to_dict().values()))
            for value in ("true", "2", "yes"):
                with patch.dict(os.environ, {FLAGS["merge_kv_append"]: value}):
                    with self.assertRaises(ValueError):
                        ThorOptions.from_env()

    def test_housekeeping_independent(self):
        for projection, append in itertools.product((False, True), repeat=2):
            with patch.dict(os.environ, {
                FLAGS["cache_projection_weights"]: str(int(projection)),
                FLAGS["merge_kv_append"]: str(int(append)),
            }, clear=True):
                opts = ThorOptions.from_env()
                self.assertFalse(opts.cache_mlp_weights)
                self.assertFalse(opts.cache_qkv_weights)
                expected = {(False, False): None, (True, False): "projection_shadow",
                            (False, True): "direct_kv_append", (True, True): "projection_shadow_direct_kv_append"}
                self.assertEqual(housekeeping_route(), expected[projection, append])

    def test_fa4_has_only_real_dependency(self):
        ThorOptions(fa4_query_staging=True).validate()
        ThorOptions(fa4_query_staging=True, paged_kv_affine=True).validate()
        with self.assertRaises(ValueError):
            ThorOptions(paged_kv_affine=True).validate()

    def test_approximate_gates(self):
        with self.assertRaises(ValueError):
            ThorOptions(visible_window=56).validate()
        ThorOptions(visible_window=56, allow_approximate=True).validate()
        for value in (0, 65, True):
            with self.assertRaises(ValueError):
                ThorOptions(visible_window=value, allow_approximate=True).validate()

    def test_process_immutable(self):
        from lingbot_map.optimizations.thor import options
        with patch.object(options, "_ACTIVE", None), patch.dict(os.environ, {}, clear=True):
            ThorOptions().activate()
            with self.assertRaises(RuntimeError):
                ThorOptions(cache_qkv_weights=True).activate()
            os.environ[FLAGS["cache_qkv_weights"]] = "1"
            with self.assertRaises(RuntimeError):
                ThorOptions().activate()

    def test_legacy_env_rejected(self):
        with patch.dict(os.environ, {"LINGBOT_CANDIDATE003A_GCA_ROPE_LAYOUT_FUSION": "1"}):
            with self.assertRaises(ValueError):
                ThorOptions().activate()

    def test_vendor_resources(self):
        root = Path(importlib.util.find_spec("lingbot_map").origin).parent
        overlay = root / "_vendor/flash_attn_4_0_0b15"
        self.assertEqual((overlay / "VERSION").read_text().strip(), "4.0.0b15")
        self.assertTrue((overlay / "LICENSE").is_file())
        self.assertIn("Tri Dao", (overlay / "AUTHORS").read_text())
        self.assertTrue((overlay / "flash_attn/__init__.py").is_file())

    def test_cache_and_append_shape_contracts_agree(self):
        from lingbot_map.optimizations.thor.cache import SUPPORTED_TOKENS_PER_FRAME
        from lingbot_map.optimizations.thor.kv_append import ALLOWED_TOKENS_PER_FRAME

        self.assertEqual(SUPPORTED_TOKENS_PER_FRAME, ALLOWED_TOKENS_PER_FRAME)
        self.assertIn(1005, ALLOWED_TOKENS_PER_FRAME)
        self.assertNotIn(1004, ALLOWED_TOKENS_PER_FRAME)
        self.assertNotIn(1006, ALLOWED_TOKENS_PER_FRAME)

    def test_thor_cache_is_used_by_scale_attention_dispatch(self):
        """Regression: the scale path must recognize the Thor cache manager."""
        import torch
        from unittest.mock import Mock

        from lingbot_map.layers.attention import FlashInferAttention
        from lingbot_map.layers.flashinfer_cache import FlashInferKVCacheManager as OriginalManager
        from lingbot_map.optimizations.thor.cache import FlashInferKVCacheManager as ThorManager

        attention = FlashInferAttention(dim=64, num_heads=1).eval()
        x = torch.randn(1, 16, 64)
        for manager_class in (OriginalManager, ThorManager):
            with self.subTest(manager=manager_class.__module__):
                # Exercise the real attention dispatch without GPU cache allocation.
                manager = manager_class.__new__(manager_class)
                manager.tokens_per_frame = 2
                manager.append_frame = Mock()
                manager.evict_frames = Mock()

                with torch.no_grad():
                    attention(
                        x,
                        num_frames=8,
                        kv_cache=manager,
                        global_idx=3,
                        num_frame_for_scale=8,
                        num_frame_per_block=8,
                    )

                self.assertEqual(
                    manager.append_frame.call_count,
                    8,
                    "scale attention must append K/V for all eight scale frames",
                )
                self.assertEqual(manager.evict_frames.call_count, 8)
                for call in manager.append_frame.call_args_list:
                    block_idx, k, v = call.args
                    self.assertEqual(block_idx, 3)
                    self.assertEqual(tuple(k.shape), (2, 1, 64))
                    self.assertEqual(tuple(v.shape), (2, 1, 64))


if __name__ == "__main__":
    unittest.main()
