"""CPU-only output-contract regressions; run with unittest discovery.

Execute the production functions without importing the GPU/rendering stack.
Inference and input discovery are stubbed; prediction NPZ writes are real.
Set LINGBOT_TEST_ROOT to run this harness against another source checkout.
"""

import argparse
import ast
import glob
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np


ROOT = Path(os.environ.get("LINGBOT_TEST_ROOT", Path(__file__).resolve().parents[1]))


def load_batch_functions():
    source = ROOT / "demo_render/batch_demo.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    module = types.ModuleType("batch_outputs_under_test")
    module.__dict__.update(
        argparse=argparse,
        glob=glob,
        json=json,
        os=os,
        shutil=shutil,
        sys=sys,
        time=time,
        np=np,
        SCRIPT_DIR=str(source.parent),
        torch=types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False)
        ),
    )
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        module.__dict__,
    )
    return module


class BatchPredictionOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input"
        self.input.mkdir()
        self.output = self.root / "output"
        self.output.mkdir()
        self.batch = load_batch_functions()
        self.parser = self.batch.build_parser()
        self.args = self.parser.parse_args(
            [
                "--input_folder",
                str(self.input),
                "--output_folder",
                str(self.output),
                "--save_predictions",
                "--no_render",
                "--skip_existing",
            ]
        )
        self.batch.find_scenes = mock.Mock(return_value=[("scene", str(self.input), 2)])
        self.batch._get_filtered_image_paths = mock.Mock(
            return_value=["a.png", "b.png"]
        )
        self.batch.load_images_from_paths = mock.Mock(
            return_value=np.ones((2, 3, 2, 2))
        )
        self.predictions = {"depth": np.ones((2, 2, 2)), "scale": np.array(1.0)}
        self.batch.run_inference = mock.Mock(return_value=self.predictions)
        self.batch.get_sky_artifact_dirs = mock.Mock(return_value=(None, None))
        self.batch.render_with_pipeline = mock.Mock(return_value=True)
        self.device = types.SimpleNamespace(type="cpu")
        self.saved = self.output / "scene"
        self.legacy = self.output / "scene.npz"
        self.marker = self.saved / "predictions_complete.json"
        self.addCleanup(mock.patch.stopall)
        mock.patch("sys.stdout", new=io.StringIO()).start()

    def discover(self):
        return self.batch._discover_scenes(self.args, self.parser)[0]

    def process(self):
        return self.batch.process_scene(
            self.args, "scene", str(self.input), object(), self.device
        )

    def save(self):
        return self.batch.save_predictions_npz(self.predictions, str(self.legacy))

    def test_second_batch_skips_complete_scene_without_rewriting_frames(self):
        for name, folder, _ in self.discover():
            result = self.batch.process_scene(
                self.args, name, folder, object(), self.device
            )
            self.assertTrue(result["success"], result["error"])
        frame = self.saved / "frame_000000.npz"
        os.utime(frame, ns=(1_000_000_000, 1_000_000_000))
        before = (frame.read_bytes(), frame.stat().st_mtime_ns)
        for name, folder, _ in self.discover():
            self.batch.process_scene(self.args, name, folder, object(), self.device)
        self.assertEqual(self.batch.run_inference.call_count, 1)
        self.assertEqual((frame.read_bytes(), frame.stat().st_mtime_ns), before)

    def test_result_reports_the_actual_saved_directory(self):
        result = self.process()
        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["output_npz"], str(self.saved))
        self.assertTrue(Path(result["output_npz"]).is_dir())
        with np.load(self.saved / "frame_000001.npz") as frame:
            np.testing.assert_array_equal(frame["depth"], self.predictions["depth"][1])

    def test_scalar_fallback_output_is_complete(self):
        self.predictions = {"scale": np.array(1.0)}
        self.save()
        self.assertEqual(self.discover(), [])

    def test_shorter_rewrite_without_metadata_removes_stale_outputs(self):
        self.save()
        self.predictions = {"depth": np.ones((1, 2, 2))}
        self.save()
        self.assertFalse((self.saved / "frame_000001.npz").exists())
        self.assertFalse((self.saved / "meta.npz").exists())
        self.assertEqual(self.discover(), [])

    def test_literal_brackets_in_scene_name_support_skip_and_rewrite(self):
        self.saved = self.output / "scene[1]"
        self.legacy = self.output / "scene[1].npz"
        self.batch.find_scenes.return_value = [("scene[1]", str(self.input), 2)]
        self.save()
        self.assertEqual(self.discover(), [])
        self.predictions = {"depth": np.ones((1, 2, 2))}
        self.save()
        self.assertFalse((self.saved / "frame_000001.npz").exists())
        self.assertEqual(self.discover(), [])

    def test_prediction_directory_does_not_skip_other_requested_outputs(self):
        self.save()
        self.args.no_render = False
        self.assertEqual(len(self.discover()), 1)
        self.args.no_render = True
        self.args.save_glb = True
        self.assertEqual(len(self.discover()), 1)

    def test_failed_render_can_be_retried_after_predictions_are_saved(self):
        self.args.no_render = False
        self.batch.render_with_pipeline.return_value = False
        result = self.process()
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Video rendering failed")
        self.assertEqual(len(self.discover()), 1)
        self.batch.render_with_pipeline.return_value = True
        self.assertTrue(self.process()["success"])

    def test_failed_glb_can_be_retried_after_predictions_are_saved(self):
        self.args.save_glb = True
        self.batch.export_glb = mock.Mock(side_effect=OSError("GLB write failed"))
        result = self.process()
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "GLB write failed")
        self.assertEqual(len(self.discover()), 1)
        self.batch.export_glb.side_effect = None
        self.assertTrue(self.process()["success"])

    def test_missing_empty_and_unmarked_outputs_are_not_skipped(self):
        self.assertEqual(len(self.discover()), 1)
        self.saved.mkdir()
        self.assertEqual(len(self.discover()), 1)
        np.savez(self.saved / "frame_000000.npz", depth=np.ones((2, 2)))
        self.assertEqual(len(self.discover()), 1)

    def test_missing_frame_or_metadata_is_not_skipped(self):
        for filename in ("frame_000001.npz", "meta.npz"):
            with self.subTest(filename=filename):
                self.save()
                (self.saved / filename).unlink()
                self.assertEqual(len(self.discover()), 1)

    def test_empty_or_extra_frame_is_not_skipped(self):
        self.save()
        (self.saved / "frame_000001.npz").write_bytes(b"")
        self.assertEqual(len(self.discover()), 1)
        self.save()
        np.savez(self.saved / "frame_000002.npz", depth=np.ones((2, 2)))
        self.assertEqual(len(self.discover()), 1)

    def test_invalid_completion_marker_is_not_skipped(self):
        self.save()
        for marker in (
            "{",
            "[]",
            "{}",
            '{"version": 99}',
            '{"version": 1, "frames": true, "metadata": false}',
        ):
            with self.subTest(marker=marker):
                self.marker.write_text(marker, encoding="utf-8")
                self.assertEqual(len(self.discover()), 1)

    def test_failed_rewrite_invalidates_previous_completion(self):
        self.save()
        original = np.savez

        def fail_second_frame(path, **values):
            if str(path).endswith("frame_000001.npz"):
                raise OSError("simulated interrupted frame write")
            return original(path, **values)

        with mock.patch.object(np, "savez", side_effect=fail_second_frame):
            result = self.process()
        self.assertFalse(result["success"])
        self.assertFalse(self.marker.exists())
        self.assertEqual(len(self.discover()), 1)

    def test_failed_metadata_write_does_not_complete_scene(self):
        original = np.savez

        def fail_metadata(path, **values):
            if str(path).endswith("meta.npz"):
                raise OSError("simulated metadata write failure")
            return original(path, **values)

        with mock.patch.object(np, "savez", side_effect=fail_metadata):
            self.assertFalse(self.process()["success"])
        self.assertFalse(self.marker.exists())
        self.assertEqual(len(self.discover()), 1)

    def test_existing_legacy_npz_file_still_skips(self):
        np.savez(self.legacy, **self.predictions)
        self.assertEqual(self.discover(), [])

    def test_existing_video_still_skips_rendering(self):
        self.args.no_render = False
        self.args.save_predictions = False
        (self.output / f"scene{self.args.video_suffix}.mp4").write_bytes(
            b"existing video"
        )
        self.assertEqual(self.discover(), [])

    def test_render_receives_directory_and_temporary_predictions_are_removed(self):
        self.args.no_render = False
        self.args.save_predictions = False
        result = self.process()
        self.assertTrue(result["success"], result["error"])
        self.assertIsNone(result["output_npz"])
        self.assertEqual(
            self.batch.render_with_pipeline.call_args.args[0], str(self.saved)
        )
        self.assertFalse(self.saved.exists())

    def test_skip_flag_and_sky_visualization_override_saved_predictions(self):
        self.save()
        self.args.skip_existing = False
        self.assertEqual(len(self.discover()), 1)
        self.args.skip_existing = True
        self.args.visualize_sky_mask_only = True
        self.assertEqual(len(self.discover()), 1)


if __name__ == "__main__":
    unittest.main()
