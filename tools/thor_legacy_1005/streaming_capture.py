"""Capture static stages while retaining the camera's dynamic history."""
from contextlib import contextmanager

import torch


@contextmanager
def position_buffer(aggregator, positions):
    """Bind capture to a live RoPE tensor, not one Python frame index."""
    name = "_get_3d_positions_streaming"
    previous = aggregator.__dict__.get(name)
    had_override = name in aggregator.__dict__

    def get_positions(num_frames, height, width, device, f_start, f_end):
        if num_frames != 1:
            raise ValueError("Captured aggregator requires single-frame positions")
        return positions

    setattr(aggregator, name, get_positions)
    try:
        yield
    finally:
        if had_override:
            setattr(aggregator, name, previous)
        else:
            delattr(aggregator, name)


class CapturedStreamingStep:
    """Aggregator graph, dynamic camera, then strict-depth graph."""

    def __init__(
        self,
        model,
        static_input,
        *,
        capture_frame,
        scale_frames,
        side_stream,
        dtype,
        compiled_depth_forward_impl,
        use_nvtx=False,
    ):
        from . import execution

        execution.assert_compiled_nondeterministic_route(
            model, compiled_depth_forward_impl, phase="streaming capture entry",
        )
        self.model = model
        self.static_input = static_input
        self.scale_frames = scale_frames
        self.side_stream = side_stream
        self.dtype = dtype
        self.use_nvtx = use_nvtx
        self.next_frame = capture_frame
        self.prepared_frame = None
        self.failed = False
        aggregator = model.aggregator
        manager = aggregator.kv_cache_manager
        camera = model.camera_head
        if (
            aggregator.total_frames_processed != capture_frame
            or camera.frame_idx != capture_frame
        ):
            raise RuntimeError(
                "Prime must start from the next unprocessed frame in both stages"
            )
        if not aggregator.enable_3d_rope or aggregator.rope3d is None:
            raise ValueError("Thor capture requires the supported 3D RoPE route")
        self.get_positions = aggregator._get_3d_positions_streaming
        self.height, self.width = static_input.shape[-2:]
        self.positions = self._positions_for(capture_frame).clone()
        previous_pos = aggregator._cached_pos3d
        camera_cache = camera.kv_cache
        camera_frame = camera.frame_idx
        self.aggregator_graph = torch.cuda.CUDAGraph()
        self.depth_graph = torch.cuda.CUDAGraph()

        def aggregate():
            return model._aggregate_features(
                static_input,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=1,
            )

        try:
            with position_buffer(aggregator, self.positions):
                manager.prepare_frame_for_graph(capture_frame)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                    aggregate()
                torch.cuda.synchronize()
                aggregator.total_frames_processed = capture_frame
                manager.prepare_frame_for_graph(capture_frame)
                before = execution.compile_counter_snapshot()
                with torch._dynamo.config.patch(
                    error_on_recompile=True,
                    fail_on_recompile_limit_hit=True,
                ):
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                        with torch.cuda.graph(self.aggregator_graph):
                            self.features, self.patch_start_idx = aggregate()
                torch.cuda.synchronize()
                self.aggregator_compile_delta = execution.compile_counter_delta(
                    before,
                    execution.compile_counter_snapshot(),
                )
                execution.validate_zero_compile_counter_delta(
                    self.aggregator_compile_delta,
                    phase="aggregator capture",
                )
                self.aggregator_graph.replay()
                torch.cuda.synchronize()
        finally:
            aggregator.total_frames_processed = capture_frame
            aggregator._cached_pos3d = previous_pos

        def depth():
            return model._predict_depth(
                self.features,
                static_input,
                self.patch_start_idx,
                gather_outputs=True,
            )

        with execution.use_strict_deterministic_compiled_depth_only(
            model,
            compiled_depth_forward_impl=compiled_depth_forward_impl,
            expected_depth_calls=1,
        ) as prime_route:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                depth()
        torch.cuda.synchronize()
        before = execution.compile_counter_snapshot()
        with (
            torch._dynamo.config.patch(
                error_on_recompile=True,
                fail_on_recompile_limit_hit=True,
            ),
            execution.use_strict_deterministic_compiled_depth_only(
                model,
                compiled_depth_forward_impl=compiled_depth_forward_impl,
                expected_depth_calls=1,
            ) as capture_route,
        ):
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                with torch.cuda.graph(self.depth_graph):
                    self.depth_output = depth()
        torch.cuda.synchronize()
        if camera.kv_cache is not camera_cache or camera.frame_idx != camera_frame:
            raise RuntimeError("Camera history changed during static-stage capture")
        self.contract = {
            "prime": dict(prime_route),
            "capture": dict(capture_route),
            "compile_counter_delta": execution.compile_counter_delta(
                before,
                execution.compile_counter_snapshot(),
            ),
            "replay_uses_captured_strict_depth": True,
            "host_deterministic_flag_during_replay": False,
            "capture_boundary": "aggregator_and_depth_only",
            "camera_execution": "uncaptured_original_dynamic_history",
            "temporal_positions": "original_rope_copied_each_frame",
            "aggregator_compile_counter_delta": self.aggregator_compile_delta,
            "prime_consumes_camera_history": False,
        }
        execution.validate_capture_depth_contract(self.contract)
        self.output = {**self.depth_output, "images": static_input}

    def _positions_for(self, frame):
        return self.get_positions(
            1,
            self.height,
            self.width,
            self.static_input.device,
            frame,
            frame + 1,
        )

    def prepare(self, frame):
        if self.failed:
            raise RuntimeError(
                "Failed streaming step cannot be reused; start a fresh sequence"
            )
        if frame != self.next_frame or self.prepared_frame is not None:
            raise RuntimeError("Replay frames must be prepared once, in sequence")
        if self.model.camera_head.frame_idx != frame:
            raise RuntimeError("Camera history is out of step with replay")
        if self.model.aggregator.total_frames_processed != frame:
            raise RuntimeError("Aggregator position is out of step with replay")
        self.positions.copy_(self._positions_for(frame))
        self.model.aggregator.kv_cache_manager.prepare_frame_for_graph(frame)
        self.prepared_frame = frame

    def replay(self):
        if self.failed:
            raise RuntimeError(
                "Failed streaming step cannot be reused; start a fresh sequence"
            )
        if self.prepared_frame != self.next_frame:
            raise RuntimeError("Prepare the current frame before replay")
        try:
            return self._run_frame()
        except Exception:
            self.failed = True
            raise

    def _run_frame(self):
        model = self.model
        main_stream = torch.cuda.current_stream()
        self.aggregator_graph.replay()
        model.aggregator.total_frames_processed = self.next_frame + 1
        model.aggregator._cached_pos3d = self.positions
        self.side_stream.wait_stream(main_stream)
        with (
            torch.cuda.stream(self.side_stream),
            torch.no_grad(),
            torch.amp.autocast("cuda", dtype=self.dtype),
        ):
            camera_output = model._predict_camera(
                self.features,
                mask=None,
                causal_inference=True,
                num_frame_for_scale=self.scale_frames,
                sliding_window_size=None,
                num_frame_per_block=1,
                gather_outputs=True,
            )
        main_stream.wait_stream(self.side_stream)
        self.depth_graph.replay()
        if model.camera_head.frame_idx != self.next_frame + 1:
            raise RuntimeError("Camera call did not advance exactly one frame")
        self.output.update(camera_output)
        self.next_frame += 1
        self.prepared_frame = None
        return self.output
