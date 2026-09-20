"""Weight-cache installers keep parameters and serialized statistics stable."""
import types
import unittest

import torch

from lingbot_map.layers.block import Block, FlashInferBlock
from lingbot_map.optimizations.thor import blocks, weight_cache


INSTALLERS = (
    (weight_cache.apply_candidate001_weight_cast_hoist, 4),
    (weight_cache.apply_candidate001c_frame_mlp_weight_cast_hoist, 4),
    (weight_cache.apply_candidate002a_patch_mlp_weight_cast_hoist, 4),
    (weight_cache.apply_candidate002b1_frame_qkv_weight_cast_hoist, 2),
    (weight_cache.apply_candidate002b2_patch_qkv_weight_cast_hoist, 2),
    (weight_cache.apply_candidate002b3_gca_qkv_weight_cast_hoist, 2),
)


class ThorWeightCacheTest(unittest.TestCase):
    def make_model(self):
        model = torch.nn.Module()
        agg = model.aggregator = torch.nn.Module()
        agg.frame_blocks = torch.nn.ModuleList([Block(dim=8, num_heads=2)])
        agg.patch_embed = torch.nn.Module()
        agg.patch_embed.blocks = torch.nn.ModuleList([Block(dim=8, num_heads=2)])
        agg.global_blocks = torch.nn.ModuleList([FlashInferBlock(dim=8, num_heads=2)])
        for block in agg.global_blocks:
            block.attn_post_ffn = types.MethodType(blocks.attn_post_ffn, block)
        return model.eval()

    def test_installers_share_stats_schema_and_preserve_parameters(self):
        for install, buffer_count in INSTALLERS:
            with self.subTest(installer=install.__name__):
                model = self.make_model()
                original = {name: param.clone() for name, param in model.state_dict().items()}
                parameter_ids = {name: id(param) for name, param in model.named_parameters()}
                stats = install(model)
                self.assertIsInstance(stats, weight_cache.WeightCacheStats)
                self.assertEqual(vars(stats), {"blocks_patched": 1, "buffers_registered": buffer_count})
                self.assertEqual(vars(install(model)), {"blocks_patched": 1, "buffers_registered": 0})
                self.assertEqual({name: id(param) for name, param in model.named_parameters()}, parameter_ids)
                self.assertEqual(model.state_dict().keys(), original.keys())
                for name, param in model.state_dict().items():
                    self.assertEqual(param.dtype, torch.float32)
                    self.assertTrue(torch.equal(param, original[name]), name)

    def test_missing_blocks_return_empty_stats(self):
        for install, _ in INSTALLERS:
            with self.subTest(installer=install.__name__):
                stats = install(torch.nn.Module())
                self.assertEqual(stats, weight_cache.WeightCacheStats(0, 0))


if __name__ == "__main__":
    unittest.main()
