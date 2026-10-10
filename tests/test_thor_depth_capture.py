"""CPU control-flow checks of static graphs and dynamic streaming state."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import torch

from tools.thor_legacy_1005 import execution


class FakeGraph:
    def __init__(self):
        self.operations = []
        self.replays = 0

    def replay(self):
        self.replays += 1
        for _, operation in self.operations:
            operation()


class DepthCaptureTest(unittest.TestCase):
    def setUp(self):
        self.previous = torch.are_deterministic_algorithms_enabled()
        self.warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(False, warn_only=False)
        self.addCleanup(torch.use_deterministic_algorithms, self.previous, warn_only=self.warn_only)
        self.compiled = object()
        self.recording = None
        self.graphs = []
        self.flags = []
        self.positions_seen = []
        self.shapes_seen = []
        self.feature = torch.zeros(1, device="cpu")
        self.depth_output = torch.zeros(1, device="cpu")
        self.pose_output = torch.zeros(1, device="cpu")
        self.image = torch.ones(1, 1, 3, 378, 518, device="cpu")
        self.manager = SimpleNamespace(_graph_mode=True, _nvtx_profile=True,
                                       prepare_frame_for_graph=Mock())
        aggregator = SimpleNamespace(
            total_frames_processed=18, _nvtx_profile=True, kv_cache_manager=self.manager,
            enable_3d_rope=True, rope3d=object(), _cached_pos3d=None,
            _get_3d_positions_streaming=lambda n, h, w, device, start, end:
                torch.tensor([start], dtype=torch.float32, device="cpu"),
        )
        camera = SimpleNamespace(frame_idx=18, kv_cache=[{"history": list(range(18))}])

        def aggregate(images, **kwargs):
            self.flags.append(("aggregate", torch.are_deterministic_algorithms_enabled()))
            self.shapes_seen.append(tuple(images.shape))
            position = aggregator._get_3d_positions_streaming(
                1, *images.shape[-2:], images.device,
                aggregator.total_frames_processed, aggregator.total_frames_processed + 1,
            )
            self.positions_seen.append(float(position[0]))
            if self.recording is not None:
                self.feature.fill_(float("nan"))
            self.launch("aggregate", lambda: self.feature.copy_(images.sum().reshape(1) + position))
            aggregator.total_frames_processed += images.shape[1]
            return [self.feature], 6

        def predict_camera(features, **kwargs):
            self.flags.append(("camera", torch.are_deterministic_algorithms_enabled()))
            history_length = len(camera.kv_cache[0]["history"])
            self.launch("camera", lambda: self.pose_output.copy_(features[0] + history_length))
            camera.kv_cache[0]["history"].append(camera.frame_idx)
            camera.frame_idx += 1
            return {"pose_enc": self.pose_output}

        def depth(features, images, start, **kwargs):
            self.flags.append(("depth", torch.are_deterministic_algorithms_enabled()))
            self.launch("depth", lambda: self.depth_output.copy_(features[0] * 2))
            return {"depth": self.depth_output}

        self.model = SimpleNamespace(
            training=False, depth_head=SimpleNamespace(_forward_impl=self.compiled),
            aggregator=aggregator, camera_head=camera, _aggregate_features=aggregate,
            _predict_camera=predict_camera, _predict_depth=depth,
            parameters=lambda: iter([self.feature]), clean_kv_cache=Mock(),
        )
        self.original_depth = depth
        self.original_positions = aggregator._get_3d_positions_streaming

    def launch(self, name, operation):
        if self.recording is None:
            operation()
        else:
            # Replay executes recorded tensor work, not the Python model methods.
            self.recording.operations.append((name, operation))

    @contextmanager
    def capture(self, graph):
        self.graphs.append(graph)
        self.recording = graph
        try:
            yield
        finally:
            self.recording = None

    @contextmanager
    def fake_cuda(self):
        with patch.object(torch.cuda, "CUDAGraph", side_effect=FakeGraph), \
             patch.object(torch.cuda, "graph", side_effect=self.capture), \
             patch.object(torch.cuda, "synchronize"), \
             patch.object(torch.cuda, "current_stream", return_value=Mock()), \
             patch.object(torch.cuda, "stream", side_effect=lambda *a: nullcontext()), \
             patch.object(torch.amp, "autocast", side_effect=lambda *a, **k: nullcontext()):
            yield

    def run_capture(self):
        return execution.prime_and_capture_depth_route(
            self.model, self.image, capture_frame=18, scale_frames=8,
            side_stream=Mock(), dtype=torch.bfloat16,
            compiled_depth_forward_impl=self.compiled,
        )

    def test_static_graphs_exclude_camera_and_capture_strict_depth(self):
        with self.fake_cuda():
            step, output, contract = self.run_capture()
        self.assertEqual(self.graphs, [step.aggregator_graph, step.depth_graph])
        self.assertEqual([name for name, _ in step.aggregator_graph.operations], ["aggregate"])
        self.assertEqual([name for name, _ in step.depth_graph.operations], ["depth"])
        self.assertEqual(self.flags, [("aggregate", False)] * 2 + [("depth", True)] * 2)
        self.assertEqual(self.manager.prepare_frame_for_graph.call_args_list, [call(18), call(18)])
        self.assertEqual(self.shapes_seen, [(1, 1, 3, 378, 518)] * 2)
        self.assertEqual(378 // 14 * (518 // 14) + 6, 1005)
        self.assertEqual(set(output), {"depth", "images"})
        self.assertIs(self.model._predict_depth, self.original_depth)
        execution.validate_capture_depth_contract(contract)

    def test_prime_and_capture_preserve_unprocessed_frame_and_camera_history(self):
        with self.fake_cuda():
            self.run_capture()
        self.assertEqual(self.positions_seen, [18, 18])
        self.assertEqual(self.model.aggregator.total_frames_processed, 18)
        self.assertEqual(self.model.camera_head.frame_idx, 18)
        self.assertEqual(self.model.camera_head.kv_cache[0]["history"], list(range(18)))
        self.assertIs(self.model.aggregator._get_3d_positions_streaming, self.original_positions)

    def test_replay_updates_temporal_position_and_camera_history(self):
        with self.fake_cuda():
            step, output, _ = self.run_capture()
            position_address = step.positions.data_ptr()
            pointers = {name: tensor.data_ptr() for name, tensor in output.items()}
            for frame in (18, 19, 20):
                self.image.fill_(frame)
                step.prepare(frame)
                step.replay()
                expected = self.image.sum().reshape(1) + frame
                self.assertTrue(torch.equal(output["depth"], expected * 2))
                self.assertTrue(torch.equal(output["pose_enc"], expected + frame))
                self.assertEqual({name: output[name].data_ptr() for name in pointers}, pointers)
                self.assertEqual(step.positions.data_ptr(), position_address)
                self.assertEqual(float(step.positions[0]), frame)
                self.assertEqual(self.model.aggregator.total_frames_processed, frame + 1)
                self.assertEqual(self.model.camera_head.frame_idx, frame + 1)
                self.assertEqual(self.model.camera_head.kv_cache[0]["history"], list(range(frame + 1)))
        self.assertEqual(step.aggregator_graph.replays, 4)  # Includes depth-priming execution.
        self.assertEqual(step.depth_graph.replays, 3)
        self.assertEqual(self.positions_seen, [18, 18])
        self.assertEqual(self.flags[-3:], [("camera", False)] * 3)

    def test_depth_prime_receives_executed_aggregator_buffers(self):
        def check(features, *args, **kwargs):
            self.assertTrue(torch.isfinite(features[0]).all())
            self.assertTrue(torch.equal(features[0], self.image.sum().reshape(1) + 18))
            return self.original_depth(features, *args, **kwargs)

        self.model._predict_depth = check
        with self.fake_cuda():
            self.run_capture()

    def test_missing_duplicate_or_out_of_order_prepare_fails(self):
        with self.fake_cuda():
            step, _, _ = self.run_capture()
            with self.assertRaisesRegex(RuntimeError, "Prepare"):
                step.replay()
            with self.assertRaisesRegex(RuntimeError, "in sequence"):
                step.prepare(19)
            step.prepare(18)
            with self.assertRaisesRegex(RuntimeError, "once"):
                step.prepare(18)

    def test_camera_failure_invalidates_step(self):
        with self.fake_cuda():
            step, _, _ = self.run_capture()
            step.prepare(18)
            self.model._predict_camera = Mock(side_effect=RuntimeError("camera failed"))
            with self.assertRaisesRegex(RuntimeError, "camera failed"):
                step.replay()
            with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
                step.replay()
            with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
                step.prepare(18)

    def test_camera_counter_must_advance(self):
        with self.fake_cuda():
            step, _, _ = self.run_capture()
            step.prepare(18)
            self.model._predict_camera = Mock(return_value={"pose_enc": self.pose_output})
            with self.assertRaisesRegex(RuntimeError, "advance exactly one"):
                step.replay()
            self.assertTrue(step.failed)

    def test_aggregate_failure_restores_positions_and_counter(self):
        def fail(*args, **kwargs):
            self.model.aggregator.total_frames_processed += 1
            raise RuntimeError("aggregate failed")

        self.model._aggregate_features = fail
        with self.fake_cuda(), self.assertRaisesRegex(RuntimeError, "aggregate failed"):
            self.run_capture()
        self.assertEqual(self.model.aggregator.total_frames_processed, 18)
        self.assertIs(self.model.aggregator._get_3d_positions_streaming, self.original_positions)

    def test_compile_during_capture_or_replay_rejected(self):
        before = execution.compile_counter_snapshot()
        after = {**before, "frames.total": before["frames.total"] + 1}
        with self.fake_cuda(), patch.object(execution, "compile_counter_snapshot", side_effect=[before, after]):
            with self.assertRaisesRegex(RuntimeError, "compiled or recompiled"):
                self.run_capture()
        with patch.object(execution, "compile_counter_snapshot", side_effect=[before, after]):
            with self.assertRaisesRegex(RuntimeError, "compiled or recompiled"):
                with execution.replay_without_recompile():
                    pass

    def test_replay_compile_guard_restores_config(self):
        previous = torch._dynamo.config.error_on_recompile
        with execution.replay_without_recompile() as evidence:
            self.assertTrue(torch._dynamo.config.error_on_recompile)
            self.assertTrue(torch._dynamo.config.fail_on_recompile_limit_hit)
        self.assertFalse(any(evidence.values()))
        self.assertEqual(torch._dynamo.config.error_on_recompile, previous)

    def test_capture_rejects_deterministic_entry_and_wrong_compiled_callable(self):
        with self.fake_cuda():
            torch.use_deterministic_algorithms(True)
            with self.assertRaisesRegex(RuntimeError, "deterministic algorithms disabled"):
                self.run_capture()
            torch.use_deterministic_algorithms(False)
            self.model.depth_head._forward_impl = object()
            with self.assertRaisesRegex(RuntimeError, "compiled depth implementation"):
                self.run_capture()
        self.manager.prepare_frame_for_graph.assert_not_called()

    def test_strict_wrapper_is_depth_only_and_restores_predictor(self):
        with execution.use_strict_deterministic_compiled_depth_only(
            self.model, compiled_depth_forward_impl=self.compiled, expected_depth_calls=1,
        ) as route:
            features, start = self.model._aggregate_features(self.image)
            self.model._predict_camera(features)
            self.model._predict_depth(features, self.image, start)
            self.assertFalse(torch.are_deterministic_algorithms_enabled())
        self.assertEqual(self.flags, [("aggregate", False), ("camera", False), ("depth", True)])
        self.assertEqual((route["depth_calls"], route["scope_enters"], route["scope_restores"]), (1, 1, 1))
        self.assertIs(self.model._predict_depth, self.original_depth)
        self.assertIs(self.model.depth_head._forward_impl, self.compiled)

    def test_strict_wrapper_restores_on_predictor_exception(self):
        torch.use_deterministic_algorithms(False, warn_only=True)

        def fail(*args, **kwargs):
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            raise RuntimeError("depth failed")

        self.model._predict_depth = fail
        with self.assertRaisesRegex(RuntimeError, "depth failed"):
            with execution.use_strict_deterministic_compiled_depth_only(
                self.model, compiled_depth_forward_impl=self.compiled, expected_depth_calls=1,
            ) as route:
                self.model._predict_depth()
        self.assertEqual(route["scope_enters"], route["scope_restores"])
        self.assertIs(self.model._predict_depth, fail)
        self.assertIs(self.model.depth_head._forward_impl, self.compiled)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_missing_depth_call_restores_wrapper(self):
        with self.assertRaisesRegex(RuntimeError, "calls, expected"):
            with execution.use_strict_deterministic_compiled_depth_only(
                self.model, compiled_depth_forward_impl=self.compiled, expected_depth_calls=1,
            ):
                pass
        self.assertIs(self.model._predict_depth, self.original_depth)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())

    def test_prewarm_exception_restores_wrapper_modes_and_cleans_cache(self):
        images = self.image.expand(1, 19, 3, 378, 518)

        def fail(*args, **kwargs):
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            raise RuntimeError("prewarm failed")

        self.model._predict_depth = fail
        self.model.forward = lambda *a, **k: self.model._predict_depth()
        zeros = {f"{section}.{name}": 0 for section, name in execution.COMPILE_COUNTER_KEYS}
        with self.fake_cuda(), patch.dict("os.environ", {}, clear=True), \
             patch.object(execution, "compile_counter_snapshot", return_value=zeros), \
             self.assertRaisesRegex(RuntimeError, "prewarm failed"):
            execution.prewarm_compiled_strict_depth_only_formal_route(
                self.model, images, torch.bfloat16, compiled_depth_forward_impl=self.compiled,
            )
        self.assertIs(self.model._predict_depth, fail)
        self.assertIs(self.model.depth_head._forward_impl, self.compiled)
        self.assertTrue(self.manager._graph_mode)
        self.assertTrue(self.manager._nvtx_profile)
        self.assertTrue(self.model.aggregator._nvtx_profile)
        self.assertEqual(self.model.clean_kv_cache.call_count, 2)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())

    def test_cpu_digest_reports_nonfinite_values(self):
        digest = execution.tensor_digest(torch.tensor([1.0, float("nan")], device="cpu"))
        self.assertIs(digest["finite"], False)
        self.assertEqual(len(digest["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
