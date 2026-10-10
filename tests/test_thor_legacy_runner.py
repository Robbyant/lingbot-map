"""CPU-only checks for benchmark setup, FP32 positions and isolated sweeps."""
import os
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from lingbot_map.optimizations.thor.options import FLAGS, ThorOptions
from tools.thor_legacy_1005 import position, run, setup, sweep


ROOT = Path(__file__).resolve().parents[1]


def reference_sincos(dim, positions, omega_0=100):
    # Archived heads/utils.py uses FP32 before the outer product and sin/cos.
    exponent = torch.arange(dim // 2, dtype=torch.float32, device="cpu") / (dim / 2)
    frequencies = 1.0 / omega_0**exponent
    angles = torch.outer(positions.reshape(-1).float(), frequencies)
    return torch.cat((angles.sin(), angles.cos()), dim=1)


class LegacyRunnerTest(unittest.TestCase):
    def test_initialization_generates_cuda_fp32_then_bf16_before_random_model(self):
        events = []
        bf16_images = object()
        fp32_images = Mock()

        def cast(dtype):
            events.append(("cast", dtype))
            return bf16_images

        fp32_images.to.side_effect = cast
        fake_torch = SimpleNamespace(
            float32="fp32", bfloat16="bf16",
            manual_seed=Mock(side_effect=lambda seed: events.append(("seed", seed))),
            randn=Mock(side_effect=lambda *a, **k: events.append(("images", a, k)) or fp32_images),
        )
        model = Mock()
        model.eval.side_effect = lambda: events.append(("eval",)) or model
        model.cuda.side_effect = lambda: events.append(("cuda",)) or model
        model_class = Mock(side_effect=lambda **k: events.append(("model", k)) or model)
        actual_model, images = run.initialize_workload(fake_torch, model_class, 1000)
        self.assertIs(actual_model, model)
        self.assertIs(images, bf16_images)
        self.assertEqual([event[0] for event in events], ["seed", "images", "cast", "model", "eval", "cuda"])
        fake_torch.manual_seed.assert_called_once_with(42)
        fake_torch.randn.assert_called_once_with(1, 1000, 3, 378, 518, device="cuda", dtype="fp32")
        fp32_images.to.assert_called_once_with("bf16")
        model_class.assert_called_once_with(
            img_size=518, patch_size=14, enable_3d_rope=True,
            max_frame_num=1100, kv_cache_sliding_window=64, kv_cache_scale_frames=8,
            camera_num_iterations=4, use_sdpa=False,
        )
        model.load_state_dict.assert_not_called()

    def test_parser_defaults_all_off_even_with_enabled_environment(self):
        with patch.dict(os.environ, {name: "1" for name in FLAGS.values()}):
            args = run.build_parser().parse_args(["--out", "unused.json"])
        options = ThorOptions(**{name: getattr(args, name) for name in FLAGS},
                              visible_window=args.visible_window)
        self.assertEqual(options, ThorOptions())
        self.assertFalse(args.benchmark)
        self.assertEqual(args.frames, 19)

    def test_import_does_not_start_model_gpu_or_benchmark(self):
        script = """
import importlib.abc
import os
import sys
class BlockBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'flashinfer', 'flash_attn', 'triton'):
            raise AssertionError('Unexpected backend import: ' + fullname)
sys.meta_path.insert(0, BlockBackends())
before = dict(os.environ)
from tools.thor_legacy_1005 import run, sweep
from tools import compare_thor_runs
assert dict(os.environ) == before
assert not any(name.startswith('lingbot_map.models') for name in sys.modules)
print('import-only')
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
            check=True, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.stdout.strip(), "import-only")

    def test_shared_manager_receives_rectangular_1005_token_shape_and_capacity(self):
        manager = Mock()
        factory = Mock(return_value=manager)
        module = ModuleType("lingbot_map.optimizations.thor.cache")
        module.FlashInferKVCacheManager = factory
        aggregator = SimpleNamespace(
            kv_cache_manager=None, img_size=518, patch_size=14, num_special_tokens=6,
            _thor_options=ThorOptions(), depth=24, embed_dim=1024,
            kv_cache_scale_frames=8, kv_cache_sliding_window=64, max_frame_num=1100,
        )
        with patch.dict(sys.modules, {module.__name__: module}):
            first = setup._get_manager(aggregator, "cpu", torch.bfloat16, tokens_per_frame=1005)
            second = setup._get_manager(aggregator, "cpu", torch.bfloat16, tokens_per_frame=1005)
        self.assertIs(first, manager)
        self.assertIs(second, manager)
        factory.assert_called_once_with(
            num_blocks=24, max_num_frames=88, tokens_per_frame=1005, num_heads=16,
            head_dim=64, dtype=torch.bfloat16, device="cpu", num_special_tokens=6,
            scale_frames=8, sliding_window=64, max_total_frames=1200, backend="fa4",
        )
        manager.set_visible_window.assert_not_called()

    def test_invalid_arguments_exit_before_workload_initialization(self):
        for argv in (["--frames", "18", "--out", "unused.json"],
                     ["--out", "existing.json"]):
            with self.subTest(argv=argv), patch.object(Path, "exists", return_value=True), \
                 patch.object(run, "initialize_workload") as initialize, \
                 patch("sys.stderr", new=Mock()), self.assertRaises(SystemExit):
                run.main(argv)
            initialize.assert_not_called()


class HistoricalPositionTest(unittest.TestCase):
    def test_sincos_matches_archived_fp32_formula(self):
        for dtype in (torch.float32, torch.bfloat16):
            for dim in (16, 128):
                positions = torch.tensor([-1.25, 0, 0.3, 17, 1000], dtype=dtype, device="cpu")
                with self.subTest(dtype=dtype, dim=dim):
                    result = position.sincos(dim, positions)
                    self.assertEqual(result.dtype, torch.float32)
                    torch.testing.assert_close(result, reference_sincos(dim, positions), rtol=0, atol=0)

    def test_depth_position_grid_matches_fp32_reference_without_mutating_input(self):
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.ones(2, 16, 3, 5, dtype=dtype, device="cpu")
            original = x.clone()
            ratio, aspect = 0.1, 518 / 378
            diagonal = (aspect**2 + 1) ** 0.5
            u = torch.linspace(-aspect / diagonal * 4 / 5, aspect / diagonal * 4 / 5,
                               5, dtype=dtype, device="cpu")
            v = torch.linspace(-1 / diagonal * 2 / 3, 1 / diagonal * 2 / 3,
                               3, dtype=dtype, device="cpu")
            uu, vv = torch.meshgrid(u, v, indexing="xy")
            embedding = torch.cat((reference_sincos(8, uu), reference_sincos(8, vv)), dim=-1)
            expected = x + (embedding.view(3, 5, 16) * ratio).permute(2, 0, 1)[None]
            with self.subTest(dtype=dtype):
                actual = position.apply_pos_embed(None, x, 518, 378, ratio=ratio)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, original, rtol=0, atol=0)
                self.assertEqual(actual.dtype, torch.float32)

    def test_install_uses_fp32_rope_and_patches_only_target_instance(self):
        def model():
            return SimpleNamespace(
                aggregator=SimpleNamespace(rope3d=SimpleNamespace(
                    max_seq_len=1100, fhw_dim=(20, 22, 22), freqs=object())),
                depth_head=SimpleNamespace(_apply_pos_embed=object()),
            )

        target, untouched = model(), model()
        original_predictor = untouched.depth_head._apply_pos_embed
        original_frequencies = untouched.aggregator.rope3d.freqs
        position.install_historical_positions(target)
        expected = []
        for dim in (20, 22, 22):
            frequencies = 1.0 / 10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim)
            angles = torch.outer(torch.arange(1100, device="cpu"), frequencies)
            expected.append(torch.polar(torch.ones_like(angles), angles))
        self.assertEqual(target.aggregator.rope3d.freqs.dtype, torch.complex64)
        torch.testing.assert_close(target.aggregator.rope3d.freqs, torch.cat(expected, dim=1), rtol=0, atol=0)
        self.assertIs(target.depth_head._apply_pos_embed.__func__, position.apply_pos_embed)
        self.assertIs(target.depth_head._apply_pos_embed.__self__, target.depth_head)
        self.assertIs(untouched.depth_head._apply_pos_embed, original_predictor)
        self.assertIs(untouched.aggregator.rope3d.freqs, original_frequencies)


class LegacySweepTest(unittest.TestCase):
    def test_validation_loader_rejects_stale_runner_contract(self):
        record = {"runner_contract_id": "old", "frames": 19}
        path = Mock()
        path.read_text.return_value = json.dumps(record)
        with self.assertRaises(ValueError):
            sweep.load_validation_record(path, 19)

    def test_validation_loader_rejects_wrong_frame_count(self):
        try:
            from .test_thor_comparison import make_record
        except ImportError:
            from test_thor_comparison import make_record

        path = Mock()
        path.read_text.return_value = json.dumps(make_record(19))
        with self.assertRaisesRegex(RuntimeError, "frame-count mismatch"):
            sweep.load_validation_record(path, 200)

    def test_rows_are_cumulative_lossless_flags_with_all_off_baseline(self):
        self.assertEqual([name for name, _ in sweep.ROWS], [
            "all_off", "mlp_qkv_cache", "projection_append", "query_staging", "single_page_affine",
        ])
        expected = [set(), {"--cache-mlp-weights", "--cache-qkv-weights"},
                    {"--cache-projection-weights", "--merge-kv-append"},
                    {"--fa4-query-staging"}, {"--paged-kv-affine"}]
        previous = set()
        for (_, flags), additions in zip(sweep.ROWS, expected, strict=True):
            self.assertEqual(set(flags), previous | additions)
            args = run.build_parser().parse_args(["--out", "unused.json", *flags])
            options = ThorOptions(**{name: getattr(args, name) for name in FLAGS},
                                  visible_window=args.visible_window)
            options.validate()
            self.assertFalse(options.allow_approximate)
            self.assertIsNone(options.visible_window)
            previous = set(flags)

    def test_environment_is_cleaned_without_mutating_parent(self):
        original = {"LINGBOT_CANDIDATE003A": "1", "LINGBOT_THOR_CACHE_MLP": "1",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TORCH_LOGS": "all",
                    "PYTHONPATH": "/old", "CUDA_VISIBLE_DEVICES": "", "PATH": "/bin"}
        with patch.dict(os.environ, original, clear=True):
            result = sweep.clean_env()
            self.assertEqual(dict(os.environ), original)
        self.assertEqual(result, {"CUDA_VISIBLE_DEVICES": "", "PATH": "/bin",
                                  "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1", "PYTHONUNBUFFERED": "1"})

    def test_existing_result_or_log_prevents_any_launch_or_write(self):
        for suffix in (".json", ".log"):
            with self.subTest(suffix=suffix), \
                 patch.object(Path, "exists", autospec=True, side_effect=lambda p: p.suffix == suffix), \
                 patch.object(sweep, "write_new") as write, \
                 patch.object(sweep.subprocess, "run") as launch:
                with self.assertRaisesRegex(ValueError, "already exists"):
                    sweep.run(Path("unused"), "all_off", (), 19)
                launch.assert_not_called()
                write.assert_not_called()

    def test_record_writer_uses_exclusive_creation(self):
        path = Mock()
        path.open.side_effect = FileExistsError("exists")
        with self.assertRaises(FileExistsError):
            sweep.write_new(path, {})
        path.open.assert_called_once_with("x")


if __name__ == "__main__":
    unittest.main()
