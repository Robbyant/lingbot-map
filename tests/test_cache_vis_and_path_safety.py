"""Regression tests for device-keyed PositionGetter cache, headless vis imports,
GLB skyseg parity, exception-safe _set_skip_append, and Unicode/glob path safety.
"""

import importlib
import os
import sys
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import torch
from PIL import Image


class TestCacheVisAndPathSafety(unittest.TestCase):
    def test_position_getter_keys_by_device(self):
        from lingbot_map.layers.rope import PositionGetter, RotaryPositionEmbedding2D

        pg = PositionGetter()
        rope2d = RotaryPositionEmbedding2D()

        pos_cpu = pg(1, 8, 10, torch.device("cpu"))
        self.assertEqual(pos_cpu.device.type, "cpu")
        self.assertEqual(tuple(pos_cpu.shape), (1, 80, 2))

        accel = None
        if torch.cuda.is_available():
            accel = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accel = torch.device("mps")

        if accel is not None:
            pos_accel = pg(1, 8, 10, accel)
            self.assertEqual(pos_accel.device.type, accel.type)
            tokens = torch.randn(1, 4, 80, 32, device=accel)
            out = rope2d(tokens, pos_accel)
            self.assertEqual(out.device.type, accel.type)
            self.assertEqual(tuple(out.shape), tuple(tokens.shape))

    def test_vis_package_imports_without_viser(self):
        saved = {}
        for mod_name in list(sys.modules):
            if mod_name == "viser" or mod_name.startswith("viser.") or mod_name.startswith("lingbot_map.vis"):
                saved[mod_name] = sys.modules.pop(mod_name)
        try:
            with mock.patch.dict(sys.modules, {"viser": None, "viser.transforms": None}):
                vis_mod = importlib.import_module("lingbot_map.vis")
                self.assertIsNone(vis_mod.PointCloudViewer)
                self.assertIsNone(vis_mod.viser_wrapper)
                self.assertTrue(callable(vis_mod.predictions_to_glb))
                self.assertTrue(callable(vis_mod.load_or_create_sky_masks))
        finally:
            for mod_name in list(sys.modules):
                if mod_name.startswith("lingbot_map.vis"):
                    sys.modules.pop(mod_name, None)
            sys.modules.update(saved)

    def test_glb_export_run_skyseg_and_segment_sky_parity(self):
        from lingbot_map.vis import glb_export

        seen_shapes = []

        class UniformSession:
            def get_inputs(self):
                return [type("In", (), {"name": "input"})()]

            def get_outputs(self):
                return [type("Out", (), {"name": "output"})()]

            def run(self, _outputs, feed):
                seen_shapes.append(feed["input"].shape)
                return [np.full((1, 1, 240, 320), 0.42, dtype=np.float32)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mask_u8 = glb_export.run_skyseg(
                UniformSession(), (320, 240), np.zeros((100, 120, 3), dtype=np.uint8)
            )
            self.assertEqual(seen_shapes[-1], (1, 3, 240, 320))
            self.assertEqual(mask_u8.dtype, np.uint8)
            self.assertEqual(len(caught), 0)

        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "frame.png")
            Image.new("RGB", (64, 48), (80, 120, 200)).save(src)
            old_cwd = os.getcwd()
            os.chdir(td)
            try:
                conf = glb_export.segment_sky(src, UniformSession(), "bare_mask.png")
                self.assertEqual(conf.shape, (48, 64))
                self.assertTrue(os.path.isfile(os.path.join(td, "bare_mask.png")))
            finally:
                os.chdir(old_cwd)

    def test_predictions_to_glb_degenerate_single_point_cloud(self):
        from lingbot_map.vis.glb_export import predictions_to_glb

        preds = {
            "world_points": np.zeros((2, 4, 4, 3), dtype=np.float32),
            "world_points_conf": np.ones((2, 4, 4), dtype=np.float32),
            "images": np.full((2, 3, 4, 4), 0.5, dtype=np.float32),
            "extrinsic": np.repeat(np.eye(3, 4, dtype=np.float32)[None], 2, axis=0),
            "intrinsic": np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        }
        scene = predictions_to_glb(preds, conf_thres=0.0, show_cam=True)
        self.assertGreater(len(scene.geometry), 1)

    def test_sky_segmentation_unicode_and_bracket_paths(self):
        from lingbot_map.vis.sky_segmentation import (
            _default_sky_mask_dir,
            _list_image_files,
            segment_sky,
        )

        class GradientSession:
            def get_inputs(self):
                return [type("In", (), {"name": "input"})()]

            def get_outputs(self):
                return [type("Out", (), {"name": "output"})()]

            def run(self, _outputs, _feed):
                arr = np.linspace(0.0, 1.0, 320 * 320, dtype=np.float32).reshape(1, 1, 320, 320)
                return [arr]

        with tempfile.TemporaryDirectory() as td:
            folder = os.path.join(td, "مشهد_اختبار_[2026]")
            os.makedirs(folder)
            img_path = os.path.join(folder, "صورة_000.png")
            out_mask = os.path.join(folder, "أقنعة", "قناع_000.png")
            Image.new("RGB", (40, 30), (100, 150, 220)).save(img_path)

            listed = _list_image_files(folder)
            self.assertEqual(listed, [img_path])

            mask = segment_sky(img_path, GradientSession(), output_path=out_mask)
            self.assertEqual(mask.shape, (30, 40))
            self.assertTrue(os.path.isfile(out_mask))

            dot_dir = _default_sky_mask_dir(".")
            self.assertFalse(os.path.basename(dot_dir).startswith("."))

    def test_skip_append_reset_on_forward_exception(self):
        from lingbot_map.models.gct_stream import GCTStream

        dummy = object.__new__(GCTStream)
        torch.nn.Module.__init__(dummy)
        dummy.num_frame_for_scale = 2
        dummy.pred_normalization = False
        dummy.aggregator = type("Agg", (), {"kv_cache": {"_skip_append": False}, "kv_cache_manager": None})()
        dummy.camera_head = type("Cam", (), {"kv_cache": [{"_skip_append": False}]})()
        dummy.clean_kv_cache = lambda: None
        dummy._dummy_param = torch.nn.Parameter(torch.zeros(1))

        calls = {"n": 0}

        def fake_forward(images, **_kwargs):
            calls["n"] += 1
            s = images.shape[1]
            if calls["n"] == 2:
                raise RuntimeError("simulated transient failure on non-keyframe")
            return {"pose_enc": torch.zeros(1, s, 9)}

        dummy.forward = fake_forward
        imgs = torch.zeros(4, 3, 28, 28)
        with self.assertRaisesRegex(RuntimeError, "simulated transient failure"):
            dummy.inference_streaming(imgs, num_scale_frames=2, keyframe_interval=2)

        self.assertFalse(dummy.aggregator.kv_cache["_skip_append"])
        self.assertFalse(dummy.camera_head.kv_cache[0]["_skip_append"])


if __name__ == "__main__":
    unittest.main()
