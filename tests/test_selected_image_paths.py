"""Focused CPU-only regression for demo -> viewer -> sky-cache path identity.

Run with `python -m unittest discover -s tests -v`. This harness executes the
real viewer constructor, prediction processing, demo viewer call and sky-cache
logic. Only rendering, neural inference and codec operations are test doubles;
it does not require PyTorch, Viser, OpenCV, ONNX or downloaded model weights.
"""
import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
from typing import Dict, List, Optional, Tuple

import numpy as np


ROOT = Path(os.environ.get("LINGBOT_TEST_ROOT", Path(__file__).resolve().parents[1]))


def read_mask(path, flags=0):
    try:
        with open(path, "rb") as stream:
            return np.load(stream)
    except (OSError, ValueError):
        return None


def write_mask(path, data):
    with open(path, "wb") as stream:
        np.save(stream, data)
    return True


def load_cache_module():
    """Import unchanged cache implementation with local codec/inference doubles."""
    doubles = {
        "cv2": types.SimpleNamespace(
            imread=read_mask, imwrite=write_mask, IMREAD_GRAYSCALE=0,
        ),
        "onnxruntime": types.SimpleNamespace(InferenceSession=lambda _: object()),
        "tqdm": types.ModuleType("tqdm"),
        "tqdm.auto": types.SimpleNamespace(tqdm=lambda iterable: iterable),
    }
    path = ROOT / "lingbot_map/vis/sky_segmentation.py"
    with mock.patch.dict(sys.modules, doubles):
        spec = importlib.util.spec_from_file_location("sky_cache_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    # Every synthetic frame has one known non-sky confidence value. This isolates
    # cache identity from the neural model's classification quality.
    module.segment_sky_from_array = mock.Mock(
        side_effect=lambda image, session, h, w: image[:, :, 0].astype(np.float32) / 255,
    )
    return module


def load_viewer_class(sky):
    """Execute complete unmodified methods without importing GPU/GUI libraries."""
    source = ROOT / "lingbot_map/vis/point_cloud_viewer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PointCloudViewer")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in {"__init__", "_process_pred_dict"}]
    server = lambda **kwargs: types.SimpleNamespace(
        gui=types.SimpleNamespace(configure_theme=lambda **kwargs: None),
        on_client_connect=lambda callback: None,
    )
    namespace = dict(
        np=np, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple,
        viser=types.SimpleNamespace(ViserServer=server), torch=object(),
        apply_sky_segmentation=sky.apply_sky_segmentation,
        unproject_depth_map_to_point_map=lambda depth, extrinsic, intrinsic: np.zeros((*depth.shape[:3], 3)),
        closed_form_inverse_se3=lambda matrices: matrices,
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    viewer = namespace["PointCloudViewer"]
    viewer.read_data = lambda self, pc, colors, conf, edge: (pc, list(range(len(pc))))
    viewer._setup_gui = lambda self: None
    viewer._connect_client = lambda self, client: None
    return viewer


def demo_viewer_call():
    source = ROOT / "demo.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "PointCloudViewer")
    return compile(ast.Expression(body=call), str(source), "eval")


class SelectedImagePathsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "images"
        self.folder.mkdir()
        self.paths = [str(self.folder / f"{i:06d}.png") for i in range(4)]
        for path in self.paths:
            Path(path).touch()
        self.model = self.root / "model.onnx"
        self.model.touch()  # Prevent any download attempt.
        self.images = np.stack([np.full((3, 4, 4), value, np.float32) for value in (1, 0, 1, 0)])
        self.sky = load_cache_module()
        self.viewer_class = load_viewer_class(self.sky)
        self.call = demo_viewer_call()

    def run_demo_call(self, indices, mask_sky=True, direct=False):
        images = self.images[indices]
        n = len(indices)
        predictions = {
            "images": images, "depth": np.ones((n, 4, 4, 1)),
            "depth_conf": np.ones((n, 4, 4)),
            "extrinsic": np.tile(np.eye(4), (n, 1, 1)),
            "intrinsic": np.tile(np.eye(3), (n, 1, 1)),
        }
        masks = []
        original = self.sky.load_or_create_sky_masks

        def record(*args, **kwargs):
            value = original(*args, **kwargs)
            masks.append(value)
            return value

        args = types.SimpleNamespace(
            port=8080, conf_threshold=1.5, downsample_factor=10, point_size=0.00001,
            mask_sky=mask_sky, sky_model=str(self.model),
            sky_mask_dir=str(self.root / "cache"), sky_mask_visualization_dir=None,
        )
        with mock.patch.object(self.sky, "load_or_create_sky_masks", side_effect=record), contextlib.redirect_stdout(io.StringIO()):
            if direct:
                # Existing callers that only supply an image folder still work.
                self.viewer_class(pred_dict=predictions, mask_sky=mask_sky,
                                  image_folder=str(self.folder), skyseg_model_path=str(self.model),
                                  sky_mask_dir=args.sky_mask_dir)
            else:
                eval(self.call, dict(
                    PointCloudViewer=self.viewer_class, args=args,
                    prepare_for_visualization=lambda predictions, images: predictions,
                    predictions=predictions, images_cpu=images,
                    resolved_image_folder=str(self.folder), paths=[self.paths[i] for i in indices],
                ))
        return masks[0][:, 0, 0].tolist() if masks else None

    def test_cached_full_run_then_stride(self):
        self.assertEqual(self.run_demo_call([0, 1, 2, 3]), [1, 0, 1, 0])
        self.assertEqual(self.run_demo_call([0, 2]), [1, 1])

    def test_stride_run_does_not_poison_full_run(self):
        self.assertEqual(self.run_demo_call([0, 2]), [1, 1])
        self.assertEqual(self.run_demo_call([0, 1, 2, 3]), [1, 0, 1, 0])

    def test_nonprefix_selection_preserves_order(self):
        self.run_demo_call([0, 1, 2, 3])
        self.assertEqual(self.run_demo_call([3, 0, 2]), [0, 1, 1])

    def test_folder_only_call_remains_compatible(self):
        self.assertEqual(self.run_demo_call([0, 1, 2, 3], direct=True), [1, 0, 1, 0])

    def test_disabled_sky_mask_does_not_segment(self):
        self.assertIsNone(self.run_demo_call([0, 2], mask_sky=False))
        self.sky.segment_sky_from_array.assert_not_called()


if __name__ == "__main__":
    unittest.main()
